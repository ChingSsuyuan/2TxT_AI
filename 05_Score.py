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
from typing import Optional, List, Dict
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.tokenize import word_tokenize
from collections import Counter
import math

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


# CIDEr Score Implementation
def calculate_cider(candidate, references, n=4):
    """
    Calculate CIDEr score between a candidate sentence and reference sentences
    
    Args:
        candidate: a list of tokens of the candidate sentence
        references: a list of lists of tokens, each list is a reference sentence
        n: maximum n-gram used for CIDEr score calculation
    
    Returns:
        CIDEr score
    """
    # Process candidate and references
    candidate = [c.lower() for c in candidate]
    references = [[r.lower() for r in ref] for ref in references]
    
    def compute_tf(tokens, n):
        """Compute term frequency for a list of tokens"""
        n_grams = {}
        for k in range(1, n+1):
            for i in range(len(tokens)-k+1):
                g = tuple(tokens[i:i+k])
                n_grams[g] = n_grams.get(g, 0) + 1
        return n_grams
    
    def compute_doc_freq(refs, n):
        """Compute document frequency for references"""
        doc_freq = {}
        for ref in refs:
            # Compute n-grams for this reference
            n_grams = set()
            for k in range(1, n+1):
                for i in range(len(ref)-k+1):
                    n_grams.add(tuple(ref[i:i+k]))
            
            # Update document frequency
            for g in n_grams:
                doc_freq[g] = doc_freq.get(g, 0) + 1
        return doc_freq
    cand_tfs = compute_tf(candidate, n)
    doc_freq = compute_doc_freq(references, n)
    num_refs = len(references)
    ref_len_avg = sum(len(ref) for ref in references) / num_refs
    
    cider_scores = []
    for k in range(1, n+1):
        tfidf_cand = {}
        for g, tf in cand_tfs.items():
            if len(g) == k:
                df = doc_freq.get(g, 0)
                tfidf_cand[g] = tf * math.log(num_refs / (1.0 + df))
        
        tfidf_refs = []
        for ref in references:
            ref_tfs = compute_tf(ref, n)
            tfidf_ref = {}
            for g, tf in ref_tfs.items():
                if len(g) == k:
                    df = doc_freq.get(g, 0)
                    tfidf_ref[g] = tf * math.log(num_refs / (1.0 + df))
            tfidf_refs.append(tfidf_ref)
        cider_k = 0.0
        for ref_tfidf in tfidf_refs:
            numerator = 0.0
            for g, w in tfidf_cand.items():
                if len(g) == k and g in ref_tfidf:
                    numerator += w * ref_tfidf[g]
            
            norm_cand = math.sqrt(sum(w * w for g, w in tfidf_cand.items() if len(g) == k))
            norm_ref = math.sqrt(sum(w * w for g, w in ref_tfidf.items() if len(g) == k))
            if norm_cand > 0 and norm_ref > 0:
                cider_k += numerator / (norm_cand * norm_ref)
        
        if len(tfidf_refs) > 0:
            cider_k /= len(tfidf_refs)
        else:
            cider_k = 0.0
        
        cider_scores.append(cider_k)
    return sum(cider_scores) / len(cider_scores)


def evaluate_test_set(model_path, test_dir, ground_truth_path=None, device='cpu'):
    """Evaluating model performance on test sets using BLEU and CIDEr scores"""
    
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
    ground_truth = {}
    if ground_truth_path and os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        print(f"Loaded {len(ground_truth)} ground truth captions")
    
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend([f for f in os.listdir(test_dir) if f.lower().endswith(ext[1:])])
    
    print(f"Found {len(test_images)} images")
    generated_captions = []
    
    for img_file in tqdm(test_images, desc="Generating captions:"):
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
    
    print("\nEvaluation Metrics:")
    print("-" * 30)
    
    if ground_truth:
        bleu_scores = []
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []
        cider_scores = []
        
        smoothing = SmoothingFunction().method1
        
        for item in generated_captions:
            img_name = item['image']
            candidate = item['caption'].lower().strip()
            candidate_tokens = word_tokenize(candidate)
            
            if img_name in ground_truth:
                references = ground_truth[img_name]
                reference_tokens = [word_tokenize(ref.lower().strip()) for ref in references]
                
                bleu1 = sentence_bleu(reference_tokens, candidate_tokens, 
                                     weights=(1, 0, 0, 0), smoothing_function=smoothing)
                bleu2 = sentence_bleu(reference_tokens, candidate_tokens, 
                                     weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                bleu3 = sentence_bleu(reference_tokens, candidate_tokens, 
                                     weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
                bleu4 = sentence_bleu(reference_tokens, candidate_tokens, 
                                     weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                
                bleu1_scores.append(bleu1)
                bleu2_scores.append(bleu2)
                bleu3_scores.append(bleu3)
                bleu4_scores.append(bleu4)
                bleu_scores.append(bleu4)  
                cider = calculate_cider(candidate_tokens, reference_tokens)
                cider_scores.append(cider)
        
        if bleu_scores:
            avg_bleu1 = np.mean(bleu1_scores)
            avg_bleu2 = np.mean(bleu2_scores)
            avg_bleu3 = np.mean(bleu3_scores)
            avg_bleu4 = np.mean(bleu4_scores)
            avg_cider = np.mean(cider_scores)
            
            print(f"BLEU-1: {avg_bleu1:.4f}")
            print(f"BLEU-2: {avg_bleu2:.4f}")
            print(f"BLEU-3: {avg_bleu3:.4f}")
            print(f"BLEU-4: {avg_bleu4:.4f}")
            print(f"CIDEr: {avg_cider:.4f}")
            
            metrics = {
                'BLEU-1': avg_bleu1,
                'BLEU-2': avg_bleu2,
                'BLEU-3': avg_bleu3,
                'BLEU-4': avg_bleu4,
                'CIDEr': avg_cider
            }
        else:
            print("Warning: No matching images found in ground truth data")
            metrics = {}
    else:
        print("Warning: No ground truth captions provided. Cannot calculate BLEU and CIDEr scores.")
        metrics = {}
    avg_length = np.mean([len(item['caption'].split()) for item in generated_captions])
    print(f"Average caption length: {avg_length:.2f} words")
    
    if not metrics:
        metrics = {}
    
    metrics['Average caption length'] = avg_length
    
    results = {
        'metrics': metrics,
        'generated_captions': generated_captions
    }
    
    output_file = 'test_evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to: {output_file}")
    
    print("\nExample generated captions:")
    print("-" * 50)
    for i in range(min(5, len(generated_captions))):
        print(f"Image: {generated_captions[i]['image']}")
        print(f"Caption: {generated_captions[i]['caption']}")
        print("-" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluating image caption generation models')
    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/clip_pro_prefix-best.pt',
                       help='model path')
    parser.add_argument('--test_dir', type=str, 
                       default='./Test_Set',
                       help='Test image file path')
    parser.add_argument('--ground_truth', type=str, 
                       default=None,
                       help='Path to ground truth captions JSON file')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Use device (cpu/cuda)')
    
    args = parser.parse_args()
    
    results = evaluate_test_set(
        args.model_path, 
        args.test_dir,
        args.ground_truth,
        args.device
    )

if __name__ == "__main__":
    main()