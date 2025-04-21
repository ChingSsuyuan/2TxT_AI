import sys
import os
import torch
import sqlite3
from PIL import Image
import json
import clip
from transformers import GPT2Tokenizer
import numpy as np

# 导入必要的模型类和函数
from gpt2_prefix import ClipCaptionModel, MappingType
from gpt2_prefix_eval import generate_beam, generate2

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据库处理函数
def load_data_from_db(db_path, limit=None):
    """从SQLite数据库加载数据"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if limit:
        cursor.execute("SELECT id, file_name, caption FROM Training_Set LIMIT ?", (limit,))
    else:
        cursor.execute("SELECT id, file_name, caption FROM Training_Set")
    
    data = cursor.fetchall()
    conn.close()
    
    return data

# 图像标题生成
def generate_captions(model_path, db_path, images_dir, output_path, 
                     prefix_length=40, mapping_type='transformer_decoder', 
                     is_rn=False, use_beam=True, batch_size=1, limit=None):
    """使用训练好的模型为图像生成标题"""
    # 1. 加载模型
    if mapping_type == 'mlp':
        mapping_type_enum = MappingType.MLP
    else:  # 默认为 'transformer'
        mapping_type_enum = MappingType.TransformerDecoder
    
    prefix_dim = 640 if is_rn else 512
    model = ClipCaptionModel(prefix_length, prefix_dim=prefix_dim, mapping_type=mapping_type_enum)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 2. 加载CLIP模型
    if is_rn:
        clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    else:
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # 3. 加载GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 4. 加载数据
    data = load_data_from_db(db_path, limit)
    print(f"加载了 {len(data)} 条数据")
    
    # 5. 生成标题
    results = []
    for i, (id, file_name, gt_caption) in enumerate(data):
        image_path = os.path.join(images_dir, file_name)
        
        if not os.path.exists(image_path):
            print(f"警告: 图片不存在 {image_path}")
            continue
        
        try:
            # 处理图像
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 提取图像特征
                prefix = clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
                # 归一化特征
                prefix = prefix / prefix.norm(2, -1)
                # 映射到模型空间
                prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
            
            # 生成文本
            if use_beam:
                generated_text = generate_beam(model, tokenizer, embed=prefix_embed, beam_size=5)[0]
            else:
                generated_text = generate2(model, tokenizer, embed=prefix_embed)
            
            # 保存结果
            results.append({
                "id": id,
                "file_name": file_name,
                "ground_truth": gt_caption,
                "generated": generated_text
            })
            
            # 打印进度和结果
            if i % 10 == 0:
                print(f"进度: {i}/{len(data)} ({i/len(data)*100:.2f}%)")
                print(f"示例 - 文件: {file_name}")
                print(f"原标题: {gt_caption}")
                print(f"生成标题: {generated_text}")
                print("-" * 50)
                
        except Exception as e:
            print(f"处理图片 {file_name} 时出错: {str(e)}")
    
    # 6. 保存结果
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"生成完成! 已处理 {len(results)}/{len(data)} 张图片。结果已保存到 {output_path}")
    return results

# 主函数
def main():
    # 配置参数
    model_path = "checkpoints/coco_prefix-009.pt"
    db_path = "coco_image_title_data/image_title_database.db"
    images_dir = "coco_image_title_data/images"
    output_path = "caption_generation_results.json"
    
    # 模型参数
    prefix_length = 40  # 根据训练时使用的值设置
    mapping_type = "transformer_decoder"  # 'mlp' 或 'transformer'
    is_rn = True  # 是否使用 RN50x4
    use_beam = True  # 是否使用束搜索
    
    # 可选参数
    limit = 4  # 设置为具体数字以限制处理的图片数量，用于测试
    
    # 生成标题
    generate_captions(
        model_path, db_path, images_dir, output_path,
        prefix_length, mapping_type, is_rn, use_beam, limit=limit
    )

if __name__ == "__main__":
    main()