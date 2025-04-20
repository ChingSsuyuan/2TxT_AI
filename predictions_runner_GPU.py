import sys
sys.path.append("/Users/michaeljing/Desktop/Projects/AI_model")
from transformers import GPT2Tokenizer
import os
import numpy as np
import torch
import torch.nn.functional as nnf
from modified_train_script import ClipCaptionModel, MappingType, noise_injection
from PIL import Image
import matplotlib.pyplot as plt
import json
import clip   # 需要安装 https://github.com/openai/CLIP
import argparse, pickle
import sqlite3
from torchvision import transforms
import os.path


class Timer:
    """
    measure inference time
    """
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.timings = []

    def __enter__(self):
        self.starter.record()
        return self

    def __exit__(self, *args):
        self.ender.record()
        torch.cuda.synchronize()
        interval = self.starter.elapsed_time(self.ender)
        self.timings.append(interval)
        self.sum += interval
        self.count += 1

    def __str__(self):
        mean_syn = self.sum / self.count
        std_syn = np.std(self.timings)
        return f"mean: {mean_syn:.2f} ms, std: {std_syn:.2f} ms"


def get_precalculated_centers():
    center_path = os.path.join(os.path.dirname(__file__), 'others/CLIP_embeddings_centers_info.pkl')
    if not os.path.exists(center_path):
        print(f"Warning: Could not find modality offset file at {center_path}")
        return None
    
    with open(center_path, 'rb') as f:
        return pickle.load(f)


def image_to_display(img):
    if type(img) is str:
        img = Image.open(str(img))
    return img


def imshow(img, title=None):
    img = image_to_display(img)
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close('all')


def generate_beam(model, tokenizer, embed, beam_size: int = 5, entry_length=67, temperature=1.0, stop_token: str = '.'):
    """用束搜索生成文本"""
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        embed = embed.expand(beam_size, -1, -1)
        generated = torch.zeros((beam_size, entry_length), dtype=torch.long, device=device)
        # 初始化序列
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=embed, labels=None, attention_mask=None)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1)
            
            if tokens is None:
                tokens = torch.topk(logits, beam_size)[1]
                scores = torch.topk(logits, beam_size)[0]
                generated[:, 0] = tokens
                continue
                
            logits[:, tokenizer.pad_token_id] = -float('Inf')
            
            # 计算候选序列的分数
            beam_scores = torch.log(scores.unsqueeze(1)) + torch.log(logits)
            beam_scores = beam_scores.reshape(-1)
            
            # 获取前beam_size个候选
            next_scores, next_tokens = torch.topk(beam_scores, beam_size)
            next_beam = next_tokens // logits.shape[1]
            next_words = next_tokens % logits.shape[1]
            
            # 更新生成序列
            generated = generated[next_beam]
            generated[:, i] = next_words
            
            # 更新分数
            scores = next_scores
            
            # 更新是否停止
            is_stopped = is_stopped[next_beam]
            is_stopped = is_stopped | (next_words == stop_token_index)
            if is_stopped.all():
                break
            
            # 更新模型输入
            embed = torch.cat([embed[next_beam], model.gpt.transformer.wte(next_words.unsqueeze(1))], dim=1)
    
    # 根据分数选择最佳序列
    scores = scores / seq_lengths
    max_score_idx = scores.argmax()
    output_text = tokenizer.decode(generated[max_score_idx])
    
    # 处理输出文本
    output_text = output_text.replace('<|endoftext|>', '')
    if '.' in output_text:
        output_text = output_text[:output_text.find('.')+1]
    
    return [output_text]


def generate2(model, tokenizer, embed, entry_length=67, temperature=1., stop_token: str = '.'):
    """用greedy搜索生成文本"""
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    device = next(model.parameters()).device
    
    with torch.no_grad():
        prefix_embed = embed
        generated = torch.zeros((1, entry_length), dtype=torch.long, device=device)
        
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=prefix_embed)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            
            if next_token.item() == stop_token_index:
                break
                
            generated[0, i] = next_token
            
            prefix_embed = torch.cat(
                (prefix_embed, model.gpt.transformer.wte(next_token)), dim=1
            )
    
    output_text = tokenizer.decode(generated[0])
    output_text = output_text.replace('<|endoftext|>', '')
    
    # 截断到第一个句号
    if '.' in output_text:
        output_text = output_text[:output_text.find('.')+1]
    
    return output_text


def load_data_from_db(db_path, table_name, limit=None):
    """从SQLite数据库加载测试数据"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # 首先获取要评估的图像ID和文件名
    if limit:
        c.execute(f"SELECT id, file_name FROM {table_name} LIMIT {limit}")
    else:
        c.execute(f"SELECT id, file_name FROM {table_name}")
    
    rows = c.fetchall()
    conn.close()
    
    # 构建数据集
    data = []
    for row in rows:
        img_id, file_name = row
        # 从文件名创建一个虚拟标题
        caption = os.path.splitext(os.path.basename(file_name))[0].replace('_', ' ')
        data.append({"image_id": img_id, "filename": file_name, "caption": caption})
    
    return data


def make_preds(data, model, out_path, tokenizer, db_path, table_name, args):
    """从数据库生成预测"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 加载CLIP模型
    if args.is_rn:
        print("Loading RN50x4 CLIP model...")
        clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    else:
        print("Loading ViT-B/32 CLIP model...")
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # 加载模态偏移（如果启用）
    if args.add_modality_offset:
        centers = get_precalculated_centers()
        if centers is not None and 'offset_to_add_in_inference' in centers:
            modality_offset = centers['offset_to_add_in_inference'].to(device)
        else:
            print("Warning: Modality offset requested but not found or incorrect format.")
            modality_offset = None
    else:
        modality_offset = None
    
    # 连接到数据库获取特征
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # 确保特征表存在
    c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    if not c.fetchone():
        print(f"Error: Table '{table_name}' does not exist in database.")
        conn.close()
        return 1
    
    results = []
    new_data = []
    skips = 0
    timer = Timer()
    
    for ii, d in enumerate(data):
        img_id = d["image_id"]
        
        # 获取图像文件路径
        image_path = d.get("filename")
        
        if args.use_db_features:
            # 从数据库获取特征
            c.execute(f"SELECT features FROM {table_name} WHERE id = ?", (img_id,))
            row = c.fetchone()
            
            if not row:
                skips += 1
                print(f'Skip {skips}: Cannot find features for image_id {img_id}')
                continue
                
            # 处理数据库中的特征
            with torch.no_grad():
                timer.__enter__()
                features_blob = row[0]
                features = np.frombuffer(features_blob, dtype=np.float32).reshape(1, -1)
                prefix = torch.from_numpy(features).to(device, dtype=torch.float32)
                
                # 归一化和噪声注入
                if not args.dont_normalize_prefix:
                    prefix = prefix / prefix.norm(2, -1)
                if args.add_modality_offset and modality_offset is not None:
                    prefix = prefix + modality_offset
                    
                # 投影到GPT空间
                prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)
        else:
            # 使用CLIP模型从图像计算特征
            if not os.path.isfile(image_path):
                skips += 1
                print(f'Skip {skips}: File not found {image_path}')
                continue
                
            image_raw = Image.open(image_path).convert("RGB")
            image = preprocess(image_raw).unsqueeze(0).to(device)
            
            with torch.no_grad():
                timer.__enter__()
                prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                
                # 归一化和噪声注入
                if not args.dont_normalize_prefix:
                    prefix = prefix / prefix.norm(2, -1)
                if args.add_modality_offset and modality_offset is not None:
                    prefix = prefix + modality_offset
                    
                # 投影到GPT空间
                prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)
        
        # 生成文本
        if args.beam:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
            
        timer.__exit__()
        
        # 收集结果
        results.append((img_id, d["caption"], generated_text_prefix.lower()))
        d["caption"] = generated_text_prefix.lower()
        new_data.append({"caption": generated_text_prefix.lower(), "image_id": d["image_id"]})
        
        # 定期打印状态和保存结果
        if ii % 20 == 0:
            print(f"Processed {ii}/{len(data)} images")
            print(f"Timing: {timer}")
            for r in results[-5:]:  # 只打印最后5个结果
                print(r)
                
            with open(out_path, 'w') as outfile:
                json.dump(new_data, outfile)
    
    # 保存最终结果
    with open(out_path, 'w') as outfile:
        json.dump(new_data, outfile)
        
    conn.close()
    return 0


def main():
    print('Starting prediction runner...')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print('Loaded tokenizer')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./checkpoints/coco_prefix-009.pt', help='Path to model checkpoint')
    parser.add_argument('--out', default='./coco_image_title_data/predictions.json', help='Output path for predictions')
    parser.add_argument('--db_path', default='coco_image_title_data/image_title_database.db', help='Path to SQLite database')
    parser.add_argument('--table_name', default='image_features_clip', help='Table name in database')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of samples to predict')
    parser.add_argument('--use_db_features', action='store_true', default=False, help='Use features from database instead of computing them')
    parser.add_argument('--beam', action='store_true', default=True, help='Use beam search for generation')
    parser.add_argument('--is_rn', action='store_true', default=True, help='Use ResNet CLIP model')
    parser.add_argument('--dont_normalize_prefix', action='store_true', default=False, help='Don\'t normalize prefix')
    parser.add_argument('--add_modality_offset', action='store_true', default=False, help='Add modality offset')
    parser.add_argument('--prefix_length', type=int, default=40, help='Prefix length')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--prefix_length_clip', type=int, default=40, help='Prefix length for clip')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='Mapping type (mlp/transformer)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # 保存命令行参数
    with open(f'{os.path.dirname(args.out)}/pred_commandline_args.txt', 'w') as f:
        args_dict = args.__dict__.copy()
        json.dump(args_dict, f, indent=2)
        print(f'Args saved to file {os.path.dirname(args.out)}/pred_commandline_args.txt')
    
    # 从数据库加载测试数据
    data = load_data_from_db(args.db_path, args.table_name, args.limit)
    print(f'Loaded {len(data)} samples for evaluation')
    
    # 创建模型
    prefix_dim = 640 if args.is_rn else 512
    mapping_type = MappingType.MLP if args.mapping_type.lower() == 'mlp' else MappingType.Transformer
    
    model = ClipCaptionModel(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                             num_layers=args.num_layers, mapping_type=mapping_type)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    print(f'Loaded model from {args.checkpoint}')
    
    # 生成预测
    make_preds(data, model, args.out, tokenizer, args.db_path, args.table_name, args)
    
    print(f'Predictions saved to {args.out}')


if __name__ == '__main__':
    main()