#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像关键词生成API服务器 - 修复版
增加了错误处理和调试信息，处理UNK标记问题
"""

import os
import io
import base64
import json
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import uuid
import torchvision.models as models
import torchvision.transforms as transforms
import datetime
import traceback
import sys

# 导入模型和功能
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

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 启用跨域请求

# 全局配置
MODEL_PATH = os.environ.get('MODEL_PATH', 'model_output/best_model.pth')
DB_PATH = os.environ.get('DB_PATH', 'coco_image_title_data/image_title_database.db')
MAX_KEYWORDS = int(os.environ.get('MAX_KEYWORDS', '3'))
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.3'))  # 降低默认温度以获得更确定的输出
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploaded_images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 默认关键词 - 当无法生成有效关键词时使用
DEFAULT_KEYWORDS = ["image", "photo", "picture"]

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 全局变量
global_model = None
global_dataset = None
global_resnet = None
global_feature_extractor = None
def extract_non_unk_keywords(generated_words, idx_to_word, unk_idx, max_keywords=3):
    """从生成的词中提取频率最高的非UNK词汇"""
    from collections import Counter
    
    # 过滤掉UNK词
    unk_word = idx_to_word[unk_idx]
    non_unk_words = [word for word in generated_words if word != unk_word]
    
    # 如果没有非UNK词，返回默认关键词
    if not non_unk_words:
        return ["image", "photo", "picture"][:max_keywords]
    
    # 统计词频
    word_freq = Counter(non_unk_words)
    
    # 获取频率最高的词
    keywords = [word for word, _ in word_freq.most_common(max_keywords)]
    
    # 如果关键词不足，添加默认词
    if len(keywords) < max_keywords:
        default_keywords = ["image", "photo", "picture"]
        for kw in default_keywords:
            if kw not in keywords and len(keywords) < max_keywords:
                keywords.append(kw)
    
    return keywords
# 定义函数
def extract_keywords(text, max_keywords=3, stopwords=None):
    """从文本中提取关键词
    
    Args:
        text: 输入文本
        max_keywords: 最大关键词数量
        stopwords: 停用词集合(可选)
        
    Returns:
        keywords: 关键词列表
    """
    import re
    from collections import Counter
    
    # 基本停用词
    if stopwords is None:
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 
                   'were', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 
                   'with', 'by', 'about', 'of', 'from'}
    
    # 预处理：小写，移除标点，分词
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    # 过滤掉停用词和特殊标记
    special_tokens = {'<PAD>', '<UNK>', '<START>', '<END>'}
    filtered_words = [word for word in words 
                   if word not in stopwords 
                   and word not in special_tokens
                   and len(word) > 2]  # 排除过短的词
    
    # 如果过滤后没有词，返回原始词列表的前N个
    if not filtered_words and words:
        return words[:max_keywords]
    
    # 词频统计
    word_freq = Counter(filtered_words)
    
    # 获取频率最高的N个词
    keywords = [word for word, _ in word_freq.most_common(max_keywords)]
    
    return keywords
def load_model(model_path, dataset, device):
    """加载训练好的模型
    
    Args:
        model_path: 模型检查点路径
        dataset: 数据集对象，用于获取词汇表大小等信息
        device: 设备 (cuda/cpu)
        
    Returns:
        model: 加载好的模型
    """
    try:
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
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        traceback.print_exc()
        return None

def initialize_resnet():
    """初始化并返回ResNet模型和特征提取器"""
    try:
        # 加载预训练的ResNet50模型
        resnet = models.resnet50(weights=None).to(device)
        # 或者使用更明确的方式:
        # resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        
        # 移除最后的全连接层，以获取特征
        modules = list(resnet.children())[:-1]
        feature_extractor = torch.nn.Sequential(*modules).to(device)
        feature_extractor.eval()
        
        return resnet, feature_extractor
    except Exception as e:
        print(f"初始化ResNet时出错: {str(e)}")
        traceback.print_exc()
        return None, None

def extract_image_features(image_path, device):
    """从图像中提取ResNet50特征
    
    Args:
        image_path: 图像路径
        device: 设备 (cuda/cpu)
        
    Returns:
        features: 提取的特征张量
    """
    global global_resnet, global_feature_extractor
    
    try:
        # 确保ResNet已初始化
        if global_feature_extractor is None:
            _, global_feature_extractor = initialize_resnet()
            if global_feature_extractor is None:
                raise Exception("无法初始化特征提取器")
        
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
        image = Image.open(image_path).convert('RGB')
        print(f"图像尺寸: {image.size}")
        
        image_tensor = preprocessing(image).unsqueeze(0).to(device)
        print(f"图像张量形状: {image_tensor.shape}")
        
        # 提取特征
        with torch.no_grad():
            features = global_feature_extractor(image_tensor)
            features = features.squeeze()
            print(f"特征张量形状: {features.shape}")
            
            # 验证特征不全为零或NaN
            if torch.isnan(features).any():
                print("警告: 特征包含NaN值")
            
            if torch.all(features == 0):
                print("警告: 特征全为零")
                
            print(f"特征统计: 最小值={features.min().item()}, 最大值={features.max().item()}, 平均值={features.mean().item()}")
        
        return features
    except Exception as e:
        print(f"提取图像特征时出错: {str(e)}")
        traceback.print_exc()
        return None

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
    try:
        # 确保特征维度正确
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
            print(f"调整后的特征形状: {image_features.shape}")
            
        # 尝试不同的温度值
        all_results = {}
        temps_to_try = [temp, 0.1, 0.3, 0.5, 0.7, 1.0] 
        
        # 确保不重复尝试相同的温度
        temps_to_try = sorted(list(set(temps_to_try)))
        
        best_keywords = None
        
        for t in temps_to_try:
            print(f"\n尝试温度 t={t}")
            
            # 生成标题
            generated_tokens = model.generate_caption(
                image_features,
                max_len=30,  # 限制最大生成长度
                start_idx=dataset.start_idx,
                end_idx=dataset.end_idx,
                temp=t
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
            print(f"温度 {t} 生成的完整标题: {generated_caption}")
            
            # 检查是否全是UNK
            unk_word = idx_to_word[dataset.unk_idx]
            unk_count = generated_words.count(unk_word)
            if unk_count / len(generated_words) > 0.5 if generated_words else True:
                print(f"警告: 温度 {t} 生成的标题有超过50%是未知词")
                # 改为尝试提取非UNK词汇
                keywords = extract_non_unk_keywords(generated_words, idx_to_word, dataset.unk_idx, max_keywords)
                print(f"从非UNK词汇中提取: {keywords}")
            else:
                # 提取关键词
                keywords = extract_keywords(generated_caption, max_keywords=max_keywords)
                
                # 如果没有提取出有效关键词，使用默认值
                if not keywords:
                    print(f"警告: 温度 {t} 未能提取出有效关键词")
                    keywords = DEFAULT_KEYWORDS[:max_keywords]
                else:
                    # 找到有效的关键词，保存结果
                    best_keywords = keywords
                    print(f"温度 {t} 生成的关键词: {keywords}")
                    break  # 找到有效关键词，停止尝试其他温度
            
            all_results[t] = keywords
        
        # 如果没有找到最佳关键词，使用第一个温度的结果
        if best_keywords is None:
            print("警告: 所有温度值都未生成有效关键词")
            
            # 查找任何非默认关键词
            for t, kw in all_results.items():
                if kw != DEFAULT_KEYWORDS[:max_keywords]:
                    best_keywords = kw
                    break
            
            # 如果还是没有，使用第一个温度的结果
            if best_keywords is None:
                best_keywords = all_results[temps_to_try[0]]
        
        print(f"最终选择的关键词: {best_keywords}")
        return best_keywords
        
    except Exception as e:
        print(f"生成关键词时出错: {str(e)}")
        traceback.print_exc()
        return DEFAULT_KEYWORDS[:max_keywords]

# 加载数据集和模型
try:
    global_dataset = COCOImageCaptionDataset(db_path=DB_PATH)
    global_model = load_model(MODEL_PATH, global_dataset, device)
    global_resnet, global_feature_extractor = initialize_resnet()
    print("模型和数据集加载成功!")
except Exception as e:
    print(f"初始化失败: {str(e)}")
    traceback.print_exc()
    global_model = None
    global_dataset = None
    global_resnet = None
    global_feature_extractor = None

# API路由

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': global_model is not None,
        'dataset_loaded': global_dataset is not None,
        'resnet_loaded': global_feature_extractor is not None,
        'device': str(device)
    })

@app.route('/api/keywords', methods=['POST'])
def get_keywords():
    """从上传的图片生成关键词"""
    try:
        # 检查模型是否加载
        if global_model is None or global_dataset is None:
            return jsonify({'error': 'Model or dataset not loaded', 'keywords': DEFAULT_KEYWORDS}), 500
            
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        # 检查文件名
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # 读取并处理图片
        try:
            # 保存上传的图片
            filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(filepath)
            
            print(f"处理图像: {filepath}")
            
            # 提取特征
            image_features = extract_image_features(filepath, device)
            if image_features is None:
                return jsonify({'error': 'Failed to extract image features', 'keywords': DEFAULT_KEYWORDS}), 500
            
            # 获取参数
            top_k = int(request.form.get('top_k', MAX_KEYWORDS))
            temp = float(request.form.get('temperature', TEMPERATURE))
            
            # 生成关键词
            keywords = generate_keywords(global_model, image_features, global_dataset, device, top_k, temp)
            
            # 构建响应
            response = {
                'image_id': filename,
                'keywords': keywords
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"处理图像时出错: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Image processing failed: {str(e)}', 'keywords': DEFAULT_KEYWORDS}), 500
            
    except Exception as e:
        print(f"请求处理失败: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Request processing failed: {str(e)}', 'keywords': DEFAULT_KEYWORDS}), 500

@app.route('/api/keywords/base64', methods=['POST'])
def get_keywords_base64():
    """从Base64编码的图片生成关键词"""
    try:
        # 检查模型是否加载
        if global_model is None or global_dataset is None:
            return jsonify({'error': 'Model or dataset not loaded', 'keywords': DEFAULT_KEYWORDS}), 500
            
        # 获取JSON数据
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # 解码Base64图片
        try:
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            image_bytes = base64.b64decode(image_data)
            
            # 创建临时文件保存图像
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_bytes)
                temp_filepath = temp_file.name
            
            print(f"处理Base64图像: {temp_filepath}")
            
            # 提取特征
            image_features = extract_image_features(temp_filepath, device)
            if image_features is None:
                os.unlink(temp_filepath)
                return jsonify({'error': 'Failed to extract image features', 'keywords': DEFAULT_KEYWORDS}), 500
            
            # 获取参数
            top_k = data.get('top_k', MAX_KEYWORDS)
            temp = data.get('temperature', TEMPERATURE)
            
            # 生成关键词
            keywords = generate_keywords(global_model, image_features, global_dataset, device, top_k, temp)
            
            # 删除临时文件
            os.unlink(temp_filepath)
            
            # 构建响应
            response = {
                'keywords': keywords
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"处理Base64图像时出错: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Image processing failed: {str(e)}', 'keywords': DEFAULT_KEYWORDS}), 500
            
    except Exception as e:
        print(f"请求处理失败: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Request processing failed: {str(e)}', 'keywords': DEFAULT_KEYWORDS}), 500

@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    """批量分析多张图片"""
    try:
        # 检查模型是否加载
        if global_model is None or global_dataset is None:
            return jsonify({'error': 'Model or dataset not loaded', 'keywords': DEFAULT_KEYWORDS}), 500
            
        # 检查是否有文件
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        # 获取参数
        top_k = int(request.form.get('top_k', MAX_KEYWORDS))
        temp = float(request.form.get('temperature', TEMPERATURE))
        
        # 获取所有图片文件
        image_files = request.files.getlist('images')
        
        if not image_files:
            return jsonify({'error': 'No images selected'}), 400
        
        # 处理每张图片
        results = []
        for image_file in image_files:
            try:
                # 保存上传的图片
                filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                image_file.save(filepath)
                
                print(f"批量处理图像: {filepath}")
                
                # 提取特征
                image_features = extract_image_features(filepath, device)
                if image_features is None:
                    results.append({
                        'filename': image_file.filename,
                        'error': 'Failed to extract image features',
                        'keywords': DEFAULT_KEYWORDS[:top_k]
                    })
                    continue
                
                # 生成关键词
                keywords = generate_keywords(global_model, image_features, global_dataset, device, top_k, temp)
                
                # 添加到结果
                results.append({
                    'filename': image_file.filename,
                    'image_id': filename,
                    'keywords': keywords
                })
                
            except Exception as e:
                # 记录错误但继续处理其他图片
                print(f"处理图像 {image_file.filename} 时出错: {str(e)}")
                traceback.print_exc()
                results.append({
                    'filename': image_file.filename,
                    'error': str(e),
                    'keywords': DEFAULT_KEYWORDS[:top_k]
                })
        
        # 构建响应
        response = {
            'total': len(image_files),
            'processed': len([r for r in results if 'error' not in r]),
            'failed': len([r for r in results if 'error' in r]),
            'results': results
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"批量处理失败: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

# 返回前端页面
@app.route('/', methods=['GET'])
def index():
    """返回前端页面"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading frontend: {str(e)}", 500

# 调试用API，返回模型信息
@app.route('/api/debug', methods=['GET'])
def debug_info():
    """返回调试信息"""
    info = {
        'model_loaded': global_model is not None,
        'dataset_loaded': global_dataset is not None,
        'feature_extractor_loaded': global_feature_extractor is not None,
        'device': str(device),
        'default_temperature': TEMPERATURE,
        'max_keywords': MAX_KEYWORDS,
        'upload_folder': UPLOAD_FOLDER,
        'python_version': sys.version,
        'torch_version': torch.__version__
    }
    
    # 添加数据集信息
    if global_dataset:
        try:
            info['dataset_info'] = {
                'vocab_size': global_dataset.get_vocab_size(),
                'num_captions': len(global_dataset) if hasattr(global_dataset, '__len__') else 'unknown',
                'special_tokens': list(global_dataset.special_tokens.keys()) if hasattr(global_dataset, 'special_tokens') else []
            }
        except Exception as e:
            info['dataset_info_error'] = str(e)
    
    return jsonify(info)

# 主函数
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # 使用5001端口以避免与AirPlay冲突
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"启动服务器在端口 {port}, 调试模式: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)