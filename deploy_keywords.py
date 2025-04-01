#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
部署脚本: 加载最佳模型，提取图像特征，生成关键词
使用方法: python deploy_keywords.py --image path/to/image.jpg --model path/to/best_model.pth --db path/to/database.db
"""

import os
import argparse
import torch
import numpy as np
import sqlite3
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import sys
import datetime

# 导入必要的类
from transformer_decoder import (
    COCOImageCaptionDataset, 
    ImageCaptioningTransformer,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_DECODER_LAYERS,
    NUM_HEADS,
    DROPOUT,
    FEATURE_DIM
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='部署图像关键词生成模型')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--db', type=str, required=True, help='数据库路径 (用于加载词汇表)')
    parser.add_argument('--top_k', type=int, default=3, help='输出的最大关键词数量')
    parser.add_argument('--device', type=str, default=None, 
                        help='使用的设备 (cuda/cpu，如果不指定则自动检测)')
    parser.add_argument('--temp', type=float, default=0.7, 
                        help='生成温度 (较低值使输出更确定，较高值更多样化)')
    return parser.parse_args()

def load_model(model_path, dataset, device):
    """加载训练好的模型
    
    Args:
        model_path: 模型检查点路径
        dataset: 数据集对象，用于获取词汇表大小等信息
        device: 设备 (cuda/cpu)
        
    Returns:
        model: 加载好的模型
    """
    # 获取词汇表大小
    vocab_size = dataset.get_vocab_size()
    
    # 创建模型
    model = ImageCaptioningTransformer(
        vocab_size=vocab_size,
        feature_dim=FEATURE_DIM,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        pad_idx=dataset.pad_idx
    ).to(device)
    
    # 加载训练好的参数
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型已加载，训练epoch: {checkpoint['epoch']}, 验证损失: {checkpoint['val_loss']:.4f}")
    return model

def extract_image_features(image_path, device):
    """从图像中提取ResNet50特征
    
    Args:
        image_path: 图像路径
        device: 设备 (cuda/cpu)
        
    Returns:
        features: 提取的特征张量
    """
    # 加载预训练的ResNet50模型
    resnet = models.resnet50(pretrained=True).to(device)
    
    # 移除最后的全连接层，以获取特征
    modules = list(resnet.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules).to(device)
    feature_extractor.eval()
    
    # 图像预处理
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加载并预处理图像
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocessing(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"无法加载或处理图像: {str(e)}")
        return None
    
    # 提取特征
    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.squeeze()
    
    return features

def generate_keywords(model, image_features, dataset, device, max_keywords=3, temp=0.7):
    """生成图像关键词
    
    Args:
        model: 训练好的模型
        image_features: 图像特征
        dataset: 数据集对象
        device: 设备 (cuda/cpu)
        max_keywords: 最大关键词数量
        temp: 生成温度
        
    Returns:
        keywords: 生成的关键词列表
    """
    # 确保特征维度正确
    if image_features.dim() == 1:
        image_features = image_features.unsqueeze(0)
    
    # 生成标题
    generated_tokens = model.generate_caption(
        image_features,
        max_len=30,  # 限制最大生成长度
        start_idx=dataset.start_idx,
        end_idx=dataset.end_idx,
        temp=temp
    )
    
    # 转换tokens为文本
    idx_to_word = dataset.idx_to_word
    generated_words = []
    for idx in generated_tokens:
        # 跳过特殊token
        if idx in [dataset.pad_idx, dataset.start_idx, dataset.end_idx]:
            continue
        # 获取词
        if idx in idx_to_word:
            generated_words.append(idx_to_word[idx])
        else:
            generated_words.append(idx_to_word[dataset.unk_idx])
    
    generated_caption = ' '.join(generated_words)
    print(f"生成的完整标题: {generated_caption}")
    
    # 提取关键词
    keywords = dataset.extract_keywords(generated_caption, max_keywords=max_keywords)
    
    return keywords

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集以获取词汇表
    try:
        dataset = COCOImageCaptionDataset(db_path=args.db)
    except Exception as e:
        print(f"无法加载数据库: {str(e)}")
        return
    
    # 加载模型
    try:
        model = load_model(args.model, dataset, device)
    except Exception as e:
        print(f"无法加载模型: {str(e)}")
        return
    
    # 提取图像特征
    image_features = extract_image_features(args.image, device)
    if image_features is None:
        return
    
    # 生成关键词
    keywords = generate_keywords(
        model, 
        image_features, 
        dataset, 
        device, 
        max_keywords=args.top_k,
        temp=args.temp
    )
    
    # 打印结果
    print("\n生成的关键词:")
    for i, keyword in enumerate(keywords):
        print(f"{i+1}. {keyword}")
    
    # 保存结果到文件
    output_dir = os.path.dirname(args.model)
    if not output_dir:
        output_dir = '.'
    
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    output_file = os.path.join(output_dir, f"{image_name}_keywords.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"图像: {args.image}\n")
        f.write(f"生成的完整标题: {' '.join(generated_words)}\n\n")
        f.write("关键词:\n")
        for keyword in keywords:
            f.write(f"- {keyword}\n")
    
    print(f"\n结果已保存到: {output_file}")

# 批处理多张图片
def batch_process(image_dir, model_path, db_path, output_dir=None, device=None, top_k=3, temp=0.7):
    """批量处理目录中的所有图片
    
    Args:
        image_dir: 图片目录路径
        model_path: 模型路径
        db_path: 数据库路径
        output_dir: 输出目录路径 (默认为image_dir)
        device: 设备 (cuda/cpu)
        top_k: 最大关键词数量
        temp: 生成温度
    """
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"使用设备: {device}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(image_dir, 'keywords_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    try:
        dataset = COCOImageCaptionDataset(db_path=db_path)
    except Exception as e:
        print(f"无法加载数据库: {str(e)}")
        return
    
    # 加载模型
    try:
        model = load_model(model_path, dataset, device)
    except Exception as e:
        print(f"无法加载模型: {str(e)}")
        return
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, file))
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 汇总结果
    all_results = []
    
    # 处理每张图片
    for image_path in tqdm(image_files, desc="处理图片"):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 提取特征
        image_features = extract_image_features(image_path, device)
        if image_features is None:
            continue
        
        # 生成关键词
        try:
            keywords = generate_keywords(
                model, 
                image_features, 
                dataset, 
                device, 
                max_keywords=top_k,
                temp=temp
            )
            
            # 保存结果
            result = {
                'image': image_path,
                'keywords': keywords
            }
            all_results.append(result)
            
            # 写入单个图片结果
            output_file = os.path.join(output_dir, f"{image_name}_keywords.txt")
            with open(output_file, 'w') as f:
                f.write(f"图像: {image_path}\n")
                f.write("关键词:\n")
                for keyword in keywords:
                    f.write(f"- {keyword}\n")
                    
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
    
    # 写入汇总结果
    summary_file = os.path.join(output_dir, "all_keywords_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总共处理了 {len(all_results)} 张图片\n\n")
        
        for result in all_results:
            image_name = os.path.basename(result['image'])
            f.write(f"图像: {image_name}\n")
            f.write(f"关键词: {', '.join(result['keywords'])}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\n批处理完成，结果保存在: {output_dir}")
    print(f"汇总报告: {summary_file}")

if __name__ == "__main__":
    try:
        # 检查是否为批处理模式
        if len(sys.argv) > 1 and sys.argv[1] == 'batch':
            # 批处理模式
            if len(sys.argv) < 5:
                print("批处理用法: python deploy_keywords.py batch <image_dir> <model_path> <db_path> [output_dir]")
                sys.exit(1)
                
            image_dir = sys.argv[2]
            model_path = sys.argv[3]
            db_path = sys.argv[4]
            output_dir = sys.argv[5] if len(sys.argv) > 5 else None
            
            batch_process(image_dir, model_path, db_path, output_dir)
        else:
            # 单图片模式
            main()
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
