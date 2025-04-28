import os
import torch
import clip
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch import nn
import torch.nn.functional as nnf
from enum import Enum
from typing import Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class MLP(nn.Module):
    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention

class TransformerLayer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:
                x = layer(x, y)
            elif self.enc_dec:  
                x = layer(x, x, mask)
            else:  
                x = layer(x, y, mask)
        return x

class TransformerMapper(nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device):
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(self, tokens, prefix, mask=None, labels=None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, 
                prefix_size: int = 640, num_layers: int = 8, 
                mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP(
                (prefix_size, (self.gpt_embedding_size * prefix_length) // 2, 
                self.gpt_embedding_size * prefix_length)
            )
        else:
            if clip_length is None:
                clip_length = 10
            self.clip_project = TransformerMapper(
                prefix_size, self.gpt_embedding_size, prefix_length,
                clip_length, num_layers
            )

def generate2(model, tokenizer, tokens=None, prompt=None, embed=None, entry_count=1,
             entry_length=30, top_p=0.8, temperature=1.0, stop_token: str = "."):
    model.eval()
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def evaluate_test_set(model_path, test_dir, device='cpu'):
    """Evaluating model performance on test sets"""
    
    print("Loading...")
    device = torch.device(device)
    clip_model, preprocess = clip.load('RN50x4', device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    mapping_type = MappingType.Transformer
    model = ClipCaptionModel(
        prefix_length=40,
        clip_length=40,
        prefix_size=640,
        mapping_type=mapping_type
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    model.to(device)
    
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend([f for f in os.listdir(test_dir) if f.lower().endswith(ext[1:])])
    
    print(f"Find {len(test_images)} images")
    generated_captions = []
    
    for img_file in tqdm(test_images, desc="Generated captions:"):
        img_path = os.path.join(test_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prefix = clip_model.encode_image(image_input).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, 40, -1)
        caption = generate2(
            model, tokenizer,
            embed=prefix_embed,
            entry_length=30,
            temperature=1.0
        )
        
        generated_captions.append({
            'image': img_file,
            'caption': caption
        })
    
    print("\n Assessment of indicators:")
    print("-" * 30)
    
    avg_length = np.mean([len(item['caption'].split()) for item in generated_captions])
    print(f"Average caption length: {avg_length:.2f} words")
    
    all_words = []
    for item in generated_captions:
        all_words.extend(item['caption'].lower().split())
    vocab_diversity = len(set(all_words)) / len(all_words)
    print(f"Lexical diversity: {vocab_diversity:.3f}")
    
    # 3. 生成一致性
    consistency_scores = []
    
    print("\n计算生成一致性...")
    for i in range(min(10, len(test_images))):  
        img_path = os.path.join(test_dir, test_images[i])
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        captions = []
        for _ in range(3):  
            with torch.no_grad():
                prefix = clip_model.encode_image(image_input).to(device, dtype=torch.float32)
                prefix_embed = model.clip_project(prefix).reshape(1, 40, -1)
            
            caption = generate2(
                model, tokenizer,
                embed=prefix_embed,
                entry_length=30,
                temperature=1.0
            )
            captions.append(caption)
        
        # 计算BLEU分数作为一致性度量
        smoothing = SmoothingFunction().method1
        scores = []
        for i in range(len(captions)):
            for j in range(i+1, len(captions)):
                score = sentence_bleu([captions[i].split()], captions[j].split(), smoothing_function=smoothing)
                scores.append(score)
        
        if scores:
            consistency_scores.append(np.mean(scores))
    
    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    print(f"生成一致性: {avg_consistency:.3f}")
    
    # 4. 语法完整性
    proper_endings = sum(1 for item in generated_captions if item['caption'].strip().endswith('.'))
    grammar_score = proper_endings / len(generated_captions)
    print(f"语法完整性: {grammar_score:.3f}")
    
    # 5. 长度适当性
    appropriate_length = sum(1 for item in generated_captions 
                           if 5 <= len(item['caption'].split()) <= 20)
    length_score = appropriate_length / len(generated_captions)
    print(f"长度适当性: {length_score:.3f}")
    
    # 保存结果
    results = {
        'metrics': {
            '平均标题长度': avg_length,
            '词汇多样性': vocab_diversity,
            '生成一致性': avg_consistency,
            '语法完整性': grammar_score,
            '长度适当性': length_score
        },
        'generated_captions': generated_captions
    }
    
    # 计算综合得分
    overall_score = (vocab_diversity + avg_consistency + grammar_score + length_score) / 4
    results['metrics']['综合得分'] = overall_score
    
    # 保存到文件
    output_file = 'test_evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n综合得分: {overall_score:.3f}")
    print(f"评估结果已保存到: {output_file}")
    
    # 输出示例标题
    print("\n示例生成的标题:")
    print("-" * 50)
    for i in range(min(5, len(generated_captions))):
        print(f"图片: {generated_captions[i]['image']}")
        print(f"标题: {generated_captions[i]['caption']}")
        print("-" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估图像标题生成模型')
    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/clip_pro_prefix-best.pt',
                       help='模型权重路径')
    parser.add_argument('--test_dir', type=str, 
                       default='./Test_Set',
                       help='测试图片目录')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='使用的设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    results = evaluate_test_set(args.model_path, args.test_dir, args.device)


if __name__ == "__main__":
    main()