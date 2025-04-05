#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估model_output/best_model.pth模型的单词准确率
使用BLEU分数和单词匹配率进行评估
"""

import os
import torch
import torch.nn as nn
import numpy as np
import sqlite3
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from collections import Counter
from tqdm import tqdm
import re

# 导入原始代码中的类和函数
from modified_transformer_decoder import (
    COCOImageCaptionDataset, 
    ImageCaptioningTransformer,
    FEATURE_DIM,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_DECODER_LAYERS,
    NUM_HEADS,
    DROPOUT
)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def clean_caption(caption):
    """清理标题文本(去除标点和多余空格)"""
    # 转为小写
    caption = caption.lower()
    # 移除标点符号
    caption = re.sub(r'[^\w\s]', '', caption)
    # 移除多余空格
    caption = re.sub(r'\s+', ' ', caption).strip()
    return caption

def calculate_word_accuracy(reference, hypothesis):
    """计算单词级别的精确率、召回率和F1分数
    
    Args:
        reference: 参考文本的单词列表
        hypothesis: 假设文本的单词列表
        
    Returns:
        precision: 精确率 - 预测正确的单词 / 预测的单词总数
        recall: 召回率 - 预测正确的单词 / 参考文本中的单词总数
        f1: F1分数 - 精确率和召回率的调和平均
    """
    if not hypothesis:
        return 0.0, 0.0, 0.0
    
    # 计算正确预测的单词数量
    reference_counter = Counter(reference)
    hypothesis_counter = Counter(hypothesis)
    
    # 计算交集和单词数
    correct_words = sum((reference_counter & hypothesis_counter).values())
    reference_words = sum(reference_counter.values())
    hypothesis_words = sum(hypothesis_counter.values())
    
    # 计算精确率、召回率和F1
    precision = correct_words / hypothesis_words if hypothesis_words > 0 else 0
    recall = correct_words / reference_words if reference_words > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_model(model_path, db_path, num_samples=100):
    """评估模型的标题生成准确率
    
    Args:
        model_path: 模型权重路径
        db_path: 数据库路径
        num_samples: 评估样本数
        
    Returns:
        results: 评估结果字典
    """
    # 加载数据集
    dataset = COCOImageCaptionDataset(db_path=db_path)
    
    # 创建模型
    model = ImageCaptioningTransformer(
        vocab_size=dataset.get_vocab_size(),
        feature_dim=FEATURE_DIM,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        pad_idx=dataset.pad_idx
    ).to(device)
    
    # 加载模型权重
    print(f"加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 准备评估指标变量
    bleu1_scores = []
    bleu4_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    meteor_scores = []
    
    # 准备样本级别的结果
    sample_results = []
    
    # 随机选择num_samples张图片进行评估
    from torch.utils.data import Subset, DataLoader
    import random
    
    # 获取所有唯一图像ID
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM images")
        all_image_ids = [row[0] for row in cursor.fetchall()]
    
    # 随机选择num_samples个图像ID
    random.seed(42)  # 确保可重复性
    num_eval_samples = min(num_samples, len(all_image_ids))
    eval_image_ids = set(random.sample(all_image_ids, num_eval_samples))
    
    # 找到对应这些图像ID的样本索引
    eval_indices = []
    for i, (_, image_id, _) in enumerate(dataset.captions_data):
        if image_id in eval_image_ids and len(eval_indices) < num_samples:
            eval_indices.append(i)
    
    # 创建评估数据集和加载器
    eval_dataset = Subset(dataset, eval_indices)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"评估 {len(eval_dataset)} 个样本...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader, desc="评估中")):
            image_features = batch['image_features'].to(device)
            true_caption = batch['caption'][0]
            
            # 生成标题
            generated_tokens = model.generate_caption(
                image_features[0].unsqueeze(0),
                start_idx=dataset.start_idx,
                end_idx=dataset.end_idx
            )
            
            # 将tokens转换为单词（忽略特殊token）
            generated_words = []
            for idx in generated_tokens:
                if idx in [dataset.pad_idx, dataset.start_idx, dataset.end_idx]:
                    continue
                if idx in dataset.idx_to_word:
                    generated_words.append(dataset.idx_to_word[idx])
                else:
                    generated_words.append(dataset.idx_to_word[dataset.unk_idx])
            
            generated_caption = ' '.join(generated_words)
            
            # 清理标题文本
            clean_true_caption = clean_caption(true_caption)
            clean_gen_caption = clean_caption(generated_caption)
            
            # 转换为单词列表
            true_words = clean_true_caption.split()
            gen_words = clean_gen_caption.split()
            
            # 计算单词准确率指标
            precision, recall, f1 = calculate_word_accuracy(true_words, gen_words)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            # 计算BLEU分数
            bleu1 = sentence_bleu([true_words], gen_words, weights=(1, 0, 0, 0))
            bleu4 = sentence_bleu([true_words], gen_words, weights=(0.25, 0.25, 0.25, 0.25))
            bleu1_scores.append(bleu1)
            bleu4_scores.append(bleu4)
            
            # 计算METEOR分数（如果nltk支持）
            try:
                meteor = meteor_score([true_words], gen_words)
                meteor_scores.append(meteor)
            except:
                meteor = -1  # 如果不支持METEOR
            
            # 保存样本结果
            sample_results.append({
                'image_id': batch['image_id'][0].item(),
                'true_caption': true_caption,
                'generated_caption': generated_caption,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'bleu1': bleu1,
                'bleu4': bleu4,
                'meteor': meteor
            })
    
    # 计算平均分数
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_bleu1 = np.mean(bleu1_scores)
    avg_bleu4 = np.mean(bleu4_scores)
    avg_meteor = np.mean(meteor_scores) if meteor_scores else -1
    
    # 汇总结果
    results = {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'avg_bleu1': avg_bleu1,
        'avg_bleu4': avg_bleu4,
        'avg_meteor': avg_meteor,
        'sample_results': sample_results
    }
    
    return results

def print_results(results, num_examples=5):
    """打印评估结果
    
    Args:
        results: 评估结果字典
        num_examples: 打印样本数量
    """
    print("\n===== 模型评估结果 =====")
    print(f"平均精确率 (Precision): {results['avg_precision']:.4f}")
    print(f"平均召回率 (Recall): {results['avg_recall']:.4f}")
    print(f"平均F1分数: {results['avg_f1']:.4f}")
    print(f"平均BLEU-1分数: {results['avg_bleu1']:.4f}")
    print(f"平均BLEU-4分数: {results['avg_bleu4']:.4f}")
    
    if results['avg_meteor'] >= 0:
        print(f"平均METEOR分数: {results['avg_meteor']:.4f}")
    
    print("\n===== 样本评估结果 =====")
    for i, sample in enumerate(results['sample_results'][:num_examples]):
        print(f"\n样本 {i+1}:")
        print(f"真实标题: {sample['true_caption']}")
        print(f"生成标题: {sample['generated_caption']}")
        print(f"精确率: {sample['precision']:.4f}")
        print(f"召回率: {sample['recall']:.4f}")
        print(f"F1分数: {sample['f1']:.4f}")
        print(f"BLEU-1: {sample['bleu1']:.4f}")
        print(f"BLEU-4: {sample['bleu4']:.4f}")
        if sample['meteor'] >= 0:
            print(f"METEOR: {sample['meteor']:.4f}")

def save_results_to_file(results, output_path):
    """将评估结果保存到文件
    
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("===== 模型评估结果 =====\n")
        f.write(f"平均精确率 (Precision): {results['avg_precision']:.4f}\n")
        f.write(f"平均召回率 (Recall): {results['avg_recall']:.4f}\n")
        f.write(f"平均F1分数: {results['avg_f1']:.4f}\n")
        f.write(f"平均BLEU-1分数: {results['avg_bleu1']:.4f}\n")
        f.write(f"平均BLEU-4分数: {results['avg_bleu4']:.4f}\n")
        
        if results['avg_meteor'] >= 0:
            f.write(f"平均METEOR分数: {results['avg_meteor']:.4f}\n")
        
        f.write("\n===== 样本评估结果 =====\n")
        for i, sample in enumerate(results['sample_results']):
            f.write(f"\n样本 {i+1}:\n")
            f.write(f"真实标题: {sample['true_caption']}\n")
            f.write(f"生成标题: {sample['generated_caption']}\n")
            f.write(f"精确率: {sample['precision']:.4f}\n")
            f.write(f"召回率: {sample['recall']:.4f}\n")
            f.write(f"F1分数: {sample['f1']:.4f}\n")
            f.write(f"BLEU-1: {sample['bleu1']:.4f}\n")
            f.write(f"BLEU-4: {sample['bleu4']:.4f}\n")
            if sample['meteor'] >= 0:
                f.write(f"METEOR: {sample['meteor']:.4f}\n")

if __name__ == "__main__":
    # 设置路径
    model_path = "model_output/best_model.pth"
    db_path = "coco_image_title_data/image_title_database.db"
    output_path = "model_output/word_accuracy_evaluation.txt"
    
    # 评估模型
    results = evaluate_model(model_path, db_path, num_samples=100)
    
    # 打印结果
    print_results(results, num_examples=5)
    
    # 保存结果到文件
    save_results_to_file(results, output_path)
    
    print(f"\n评估结果已保存到: {output_path}")
