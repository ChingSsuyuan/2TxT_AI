#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer解码器用于图像标题生成
数据源: coco_image_title_data/image_title_database.db
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sqlite3
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义超参数
BATCH_SIZE = 32
EMBEDDING_DIM = 512  # 词嵌入维度
HIDDEN_DIM = 2048    # 前馈网络隐藏层维度
NUM_DECODER_LAYERS = 6  # Transformer解码器层数
NUM_HEADS = 8        # 多头注意力的头数
DROPOUT = 0.1
MAX_CAPTION_LENGTH = 50  # 最大标题长度
FEATURE_DIM = 2048   # ResNet50特征维度

# 自定义数据集类
class COCOImageCaptionDataset(Dataset):
    def __init__(self, db_path, max_len=MAX_CAPTION_LENGTH):
        """初始化数据集对象，从SQLite数据库加载数据
        
        Args:
            db_path: SQLite数据库路径
            max_len: 最大标题长度
        """
        self.db_path = db_path
        self.max_len = max_len
        
        # 使用临时连接加载所有必要数据
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 加载词汇表
            self.word_to_idx = {}  # 词到索引的映射
            self.idx_to_word = {}  # 索引到词的映射
            self.special_tokens = {}  # 特殊标记
            
            # 从数据库加载词汇表和索引映射
            cursor.execute("SELECT id, word, special_token FROM vocabulary")
            vocabulary_data = cursor.fetchall()
            
            cursor.execute("SELECT word_id, index_id, special_token FROM word_indices")
            word_indices_data = cursor.fetchall()
            
            # 创建词到索引的映射
            for word_id, index_id, special_token in word_indices_data:
                # 在词汇表中查找词
                for vocab_id, word, _ in vocabulary_data:
                    if vocab_id == word_id:
                        self.idx_to_word[index_id] = word
                        self.word_to_idx[word] = index_id
                        
                        # 标记特殊token
                        if special_token == 1:
                            self.special_tokens[word] = index_id
                        break
            
            # 确保特殊标记存在
            assert '<PAD>' in self.special_tokens, "找不到PAD标记"
            assert '<UNK>' in self.special_tokens, "找不到UNK标记"
            assert '<START>' in self.special_tokens, "找不到START标记"
            assert '<END>' in self.special_tokens, "找不到END标记"
            
            # 设置特殊标记的索引
            self.pad_idx = self.special_tokens['<PAD>']
            self.unk_idx = self.special_tokens['<UNK>']
            self.start_idx = self.special_tokens['<START>']
            self.end_idx = self.special_tokens['<END>']
            
            # 获取标题总数
            cursor.execute("SELECT COUNT(*) FROM captions")
            self.num_captions = cursor.fetchone()[0]
            
            # 预加载所有标题和图像ID以避免保持连接打开
            cursor.execute("SELECT id, image_id, caption FROM captions")
            self.captions_data = cursor.fetchall()
            
            # 预加载所有图像特征
            cursor.execute("SELECT image_id, features FROM image_features_resnet50")
            self.features_data = {}
            for img_id, feature_blob in cursor.fetchall():
                self.features_data[img_id] = self._parse_features(feature_blob)
        
        self.vocab_size = len(self.idx_to_word)
        print(f"数据集加载完成: {self.num_captions}个标题, 词汇量{self.vocab_size}")
    
    def __len__(self):
        """返回数据集中标题的数量"""
        return self.num_captions
    
    def __getitem__(self, idx):
        """获取指定索引的样本
        
        返回包含以下键的字典:
        - image_id: 图像ID
        - image_features: 图像特征张量
        - caption: 原始标题文本
        - tokenized_caption: 标记化的标题 (tensor)
        - caption_length: 标题长度
        """
        caption_id, image_id, caption = self.captions_data[idx]
        
        # 从预加载数据中获取图像特征
        if image_id not in self.features_data:
            raise ValueError(f"找不到image_id为{image_id}的特征")
        
        features = self.features_data[image_id]
        image_features = torch.tensor(features, dtype=torch.float32)
        
        # 标记化标题
        tokenized_caption = self._tokenize_caption(caption)
        
        return {
            'image_id': image_id,
            'image_features': image_features,
            'caption': caption,
            'tokenized_caption': tokenized_caption,
            'caption_length': len(tokenized_caption)
        }
    
    def _parse_features(self, feature_blob):
        """解析二进制特征数据为numpy数组
        
        特征是通过 np.ndarray.tobytes() 方法存储的，
        需要使用 np.frombuffer 将其转换回 numpy 数组
        """
        try:
            # 特征是作为二进制字节流存储的
            if isinstance(feature_blob, bytes):
                # 将二进制字节流重新解析为float32数组
                # ResNet50特征维度为2048
                features = np.frombuffer(feature_blob, dtype=np.float32)
                
                # 检查特征长度是否符合预期
                if len(features) == FEATURE_DIM:
                    return features
                elif len(features) > FEATURE_DIM:
                    # 如果特征长度大于预期，可能是批量存储的，只取前面的FEATURE_DIM个
                    print(f"警告: 特征长度 ({len(features)}) 大于预期 ({FEATURE_DIM})，将只使用前 {FEATURE_DIM} 个元素")
                    return features[:FEATURE_DIM]
                else:
                    print(f"警告: 特征长度 ({len(features)}) 小于预期 ({FEATURE_DIM})，将返回随机数组")
            
            # 如果无法解析，返回随机数组
            return np.random.randn(FEATURE_DIM)
        except Exception as e:
            print(f"警告: 无法解析特征数据，错误: {str(e)}")
            return np.random.randn(FEATURE_DIM)
    
    def _tokenize_caption(self, caption):
        """将标题文本转换为token索引序列
        
        添加<START>和<END>标记，并填充到max_len长度
        """
        # 简单的tokenization: 分割空格并转小写
        tokens = caption.lower().split()
        
        # 转换tokens为索引
        token_indices = [self.start_idx]
        for token in tokens:
            if token in self.word_to_idx:
                token_indices.append(self.word_to_idx[token])
            else:
                token_indices.append(self.unk_idx)
        
        token_indices.append(self.end_idx)
        
        # 填充或截断到max_len
        if len(token_indices) < self.max_len:
            token_indices.extend([self.pad_idx] * (self.max_len - len(token_indices)))
        else:
            token_indices = token_indices[:self.max_len-1] + [self.end_idx]
        
        # 确保所有索引都在有效范围内
        vocab_size = len(self.word_to_idx)
        token_indices = [min(idx, vocab_size-1) for idx in token_indices]
        
        return torch.tensor(token_indices, dtype=torch.long)
    
    def get_vocab_size(self):
        """返回词汇表大小"""
        return self.vocab_size

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """初始化位置编码层
        
        Args:
            d_model: 嵌入维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为buffer (不是参数但应该和模型一起保存和恢复)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """将位置编码添加到输入嵌入中"""
        return x + self.pe[:, :x.size(1), :]

# 图像标题生成的Transformer解码器模型
class ImageCaptioningTransformer(nn.Module):
    def __init__(self, vocab_size, feature_dim, embedding_dim, hidden_dim, 
                 num_layers, num_heads, dropout, pad_idx):
        """初始化图像标题生成Transformer模型
        
        Args:
            vocab_size: 词汇表大小
            feature_dim: 图像特征维度
            embedding_dim: 词嵌入维度
            hidden_dim: 前馈网络隐藏层维度
            num_layers: 解码器层数
            num_heads: 多头注意力头数
            dropout: dropout率
            pad_idx: 填充token的索引
        """
        super(ImageCaptioningTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        
        # 图像特征投影层
        self.feature_projection = nn.Linear(feature_dim, embedding_dim)
        
        # 词嵌入层
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 位置编码层
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            batch_first=False  # 序列维度在第二维
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # 初始化参数
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        # 初始化投影层
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.constant_(self.feature_projection.bias, 0)
        
        # 初始化输出层
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
        
        # 初始化嵌入层
        nn.init.normal_(self.word_embedding.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.word_embedding.weight[self.pad_idx], 0)
    
    def create_mask(self, target):
        """创建填充掩码和前瞻掩码
        
        Args:
            target: 目标序列 [batch_size, seq_len]
            
        Returns:
            target_padding_mask: 填充掩码
            target_mask: 前瞻掩码
        """
        # 创建目标序列的填充掩码
        target_padding_mask = (target == self.pad_idx)
        
        # 创建前瞻掩码，防止在训练期间关注未来的token
        seq_len = target.size(1)
        try:
            target_mask = torch.triu(torch.ones((seq_len, seq_len), device=target.device), diagonal=1).bool()
        except RuntimeError as e:
            print(f"创建掩码时出错: {e}")
            print(f"目标形状: {target.shape}, 序列长度: {seq_len}")
            # 作为备用方案，创建一个全零掩码
            target_mask = torch.zeros((seq_len, seq_len), device=target.device).bool()
        
        return target_padding_mask, target_mask
    
    def forward(self, image_features, target_captions, target_padding_mask=None, target_mask=None):
        """前向传播
        
        Args:
            image_features: 图像特征 [batch_size, feature_dim]
            target_captions: 目标标题 [batch_size, seq_len]
            target_padding_mask: 目标填充掩码 (可选)
            target_mask: 目标前瞻掩码 (可选)
            
        Returns:
            logits: 输出logits [batch_size, seq_len, vocab_size]
        """
        # 投影图像特征到嵌入维度
        # image_features形状: [batch_size, feature_dim]
        # projected_features形状: [batch_size, 1, embedding_dim]
        projected_features = self.feature_projection(image_features).unsqueeze(1)
        
        # 创建解码器的memory (就是图像特征)
        # memory形状: [1, batch_size, embedding_dim]
        memory = projected_features.permute(1, 0, 2)
        
        # 嵌入目标标题
        # target_captions形状: [batch_size, seq_len]
        # embedded_captions形状: [batch_size, seq_len, embedding_dim]
        embedded_captions = self.word_embedding(target_captions)
        
        # 添加位置编码
        embedded_captions = self.positional_encoding(embedded_captions)
        
        # 准备Transformer解码器输入 (更改维度)
        # tgt形状: [seq_len, batch_size, embedding_dim]
        tgt = embedded_captions.permute(1, 0, 2)
        
        # 创建掩码（如果未提供）
        if target_padding_mask is None or target_mask is None:
            target_padding_mask, target_mask = self.create_mask(target_captions)
        
        # 应用Transformer解码器
        # output形状: [seq_len, batch_size, embedding_dim]
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask
        )
        
        # 重塑输出并投影到词汇表大小
        # output形状经过permute后: [batch_size, seq_len, embedding_dim]
        # logits形状: [batch_size, seq_len, vocab_size]
        output = output.permute(1, 0, 2)
        logits = self.output_layer(output)
        
        return logits
    
    def generate_caption(self, image_features, max_len=20, start_idx=2, end_idx=3, temp=1.0):
        """为给定的图像特征生成标题
        
        Args:
            image_features: 图像特征张量 [1, feature_dim]
            max_len: 生成标题的最大长度
            start_idx: <START>标记的索引
            end_idx: <END>标记的索引
            temp: 温度参数，控制采样随机性
            
        Returns:
            generated_tokens: 生成的标题token列表
        """
        self.eval()  # 设置为评估模式
        
        with torch.no_grad():
            # 投影并准备图像特征
            projected_features = self.feature_projection(image_features).unsqueeze(1)
            memory = projected_features.permute(1, 0, 2)
            
            # 从<START>标记开始
            current_token = torch.tensor([[start_idx]], dtype=torch.long, device=image_features.device)
            generated_tokens = [start_idx]
            
            # 逐个生成token
            for _ in range(max_len):
                # 嵌入当前序列
                embedded = self.word_embedding(current_token)
                embedded = self.positional_encoding(embedded)
                tgt = embedded.permute(1, 0, 2)
                
                # 创建掩码（对于单个token不需要填充掩码）
                seq_len = current_token.size(1)
                target_mask = torch.triu(
                    torch.ones((seq_len, seq_len), device=image_features.device), 
                    diagonal=1
                ).bool()
                
                # 解码
                output = self.transformer_decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=target_mask
                )
                
                # 获取最终token预测
                output = output.permute(1, 0, 2)
                logits = self.output_layer(output[:, -1, :])
                
                # 应用温度并采样
                if temp == 0:
                    # 贪婪解码
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    # 带温度的采样
                    probs = F.softmax(logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                next_token_item = next_token.item()
                
                # 确保索引在有效范围内
                if next_token_item >= self.vocab_size:
                    next_token_item = self.unk_idx
                
                generated_tokens.append(next_token_item)
                
                # 如果生成了<END>标记则停止
                if next_token_item == end_idx:
                    break
                
                # 与之前的token连接，用于下一次迭代
                current_token = torch.cat([current_token, next_token], dim=1)
        
        return generated_tokens

# 带标签平滑的损失函数
class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps=0.1, ignore_index=0, reduction='mean'):
        """初始化带标签平滑的损失函数
        
        Args:
            eps: 平滑系数
            ignore_index: 忽略的索引（通常是PAD）
            reduction: 损失计算方式 ('mean', 'sum')
        """
        super(LabelSmoothingLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, output, target):
        """计算损失
        
        Args:
            output: 模型输出 [batch_size * seq_len, vocab_size]
            target: 目标索引 [batch_size * seq_len]
            
        Returns:
            loss: 损失值
        """
        c = output.size(-1)
        log_preds = F.log_softmax(output, dim=-1)
        
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        
        # 应用标签平滑
        mask = (target != self.ignore_index).float()
        target_one_hot = torch.zeros_like(log_preds).scatter_(
            -1, target.unsqueeze(-1), 1)
        
        smooth_target = (1 - self.eps) * target_one_hot + self.eps / (c - 1) * (1 - target_one_hot)
        smooth_loss = -(smooth_target * log_preds).sum(dim=-1)
        
        # 应用掩码，处理填充token
        smooth_loss = (smooth_loss * mask).sum() / mask.sum()
        
        return smooth_loss

# 训练函数
def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        
    Returns:
        epoch_loss: 平均epoch损失
    """
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="训练中"):
        image_features = batch['image_features'].to(device)
        tokenized_caption = batch['tokenized_caption'].to(device)
        
        # 准备用于损失计算的目标（向右移动一个位置）
        # 输入: [<START>, w1, w2, ..., <END>, <PAD>, ...]
        # 目标: [w1, w2, ..., <END>, <PAD>, <PAD>, ...]
        input_caption = tokenized_caption[:, :-1]
        target_caption = tokenized_caption[:, 1:]
        
        # 创建掩码
        target_padding_mask, target_mask = model.create_mask(input_caption)
        
        # 前向传播
        output = model(image_features, input_caption, target_padding_mask, target_mask)
        
        # 重塑以进行损失计算
        output = output.reshape(-1, output.shape[-1])
        target_caption = target_caption.reshape(-1)
        
        # 计算损失
        loss = criterion(output, target_caption)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

# 验证函数
def validate(model, dataloader, criterion, device):
    """在验证集上评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        val_loss: 平均验证损失
    """
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            image_features = batch['image_features'].to(device)
            tokenized_caption = batch['tokenized_caption'].to(device)
            
            # 准备目标
            input_caption = tokenized_caption[:, :-1]
            target_caption = tokenized_caption[:, 1:]
            
            # 创建掩码
            target_padding_mask, target_mask = model.create_mask(input_caption)
            
            # 前向传播
            output = model(image_features, input_caption, target_padding_mask, target_mask)
            
            # 重塑以进行损失计算
            output = output.reshape(-1, output.shape[-1])
            target_caption = target_caption.reshape(-1)
            
            # 计算损失
            loss = criterion(output, target_caption)
            
            val_loss += loss.item()
    
    return val_loss / len(dataloader)

# 主训练函数
def train_model(db_path, output_dir, num_epochs=10, learning_rate=0.0001, num_val_samples=2):
    """训练图像标题生成模型
    
    Args:
        db_path: 数据库路径
        output_dir: 输出目录
        num_epochs: 训练的epoch数
        learning_rate: 学习率
        num_val_samples: 验证集样本数
        
    Returns:
        model: 训练好的模型
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据集
    dataset = COCOImageCaptionDataset(db_path=db_path)
    
    # 确保使用所有图像 - 获取所有唯一图像ID
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # 直接从数据库查询所有图像ID
        cursor.execute("SELECT id FROM images")
        all_image_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"数据库中共有 {len(all_image_ids)} 张图片")
    
    # 随机选择验证图像ID
    import random
    random.seed(42)  # 设置随机种子以保证可重复性
    num_val_samples = min(num_val_samples, len(all_image_ids))  # 确保验证样本数不超过总图像数
    val_image_ids = set(random.sample(all_image_ids, num_val_samples))
    
    # 创建训练集和验证集索引
    train_indices = []
    val_indices = []
    
    # 遍历所有标题，根据图像ID分配到训练集或验证集
    for i, (_, image_id, _) in enumerate(dataset.captions_data):
        if image_id in val_image_ids:
            val_indices.append(i)
        else:
            train_indices.append(i)
    
    # 创建自定义子集
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"训练集大小: {len(train_dataset)} 条标题 (对应 {len(all_image_ids) - len(val_image_ids)} 张图片)")
    print(f"验证集大小: {len(val_indices)} 条标题 (对应 {len(val_image_ids)} 张图片)")
    
    # 创建数据加载器 (num_workers=0避免SQLite多进程问题)
    actual_batch_size = min(BATCH_SIZE, len(train_dataset))
    if actual_batch_size < BATCH_SIZE:
        print(f"调整batch_size为 {actual_batch_size} (原计划: {BATCH_SIZE})")
    
    train_dataloader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=actual_batch_size, num_workers=0)
    
    # 创建模型
    vocab_size = dataset.get_vocab_size()
    print(f"词汇表大小: {vocab_size}")
    
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
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = LabelSmoothingLoss(eps=0.1, ignore_index=dataset.pad_idx)
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss = validate(model, val_dataloader, criterion, device)
        print(f"验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
            print("已保存最佳模型检查点")
        
        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(output_dir, 'latest_model.pth'))
    
    print("训练完成!")
    return model

# 推理函数
def generate_captions(model, dataset, dataloader, device, num_samples=5):
    """生成标题并与真实标题进行比较
    
    Args:
        model: 模型
        dataset: 数据集
        dataloader: 数据加载器
        device: 设备
        num_samples: 生成样本数
        
    Returns:
        generated_captions: 生成的标题列表
    """
    model.eval()
    idx_to_word = dataset.idx_to_word
    vocab_size = len(idx_to_word)
    
    generated_captions = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            image_features = batch['image_features'].to(device)
            true_caption = batch['caption'][0]
            
            # 生成标题
            generated_tokens = model.generate_caption(
                image_features[0].unsqueeze(0),
                start_idx=dataset.start_idx,
                end_idx=dataset.end_idx
            )
            
            # 将tokens转换为单词（确保只使用有效索引）
            generated_words = []
            for idx in generated_tokens:
                # 跳过特殊token
                if idx in [dataset.pad_idx, dataset.start_idx, dataset.end_idx]:
                    continue
                # 确保索引在有效范围内
                if idx in idx_to_word:
                    generated_words.append(idx_to_word[idx])
                else:
                    # 如果索引无效，使用未知词
                    generated_words.append(idx_to_word[dataset.unk_idx])
            
            generated_caption = ' '.join(generated_words)
            
            generated_captions.append({
                'image_id': batch['image_id'][0].item(),
                'true_caption': true_caption,
                'generated_caption': generated_caption
            })
    
    return generated_captions

# 主函数
if __name__ == "__main__":
    db_path = "coco_image_title_data/image_title_database.db"
    output_dir = "model_output"
    
    # 训练模型
    model = train_model(db_path, output_dir, num_epochs=10)
    
    # 加载数据集用于推理
    dataset = COCOImageCaptionDataset(db_path=db_path)
    
    # 创建测试数据加载器
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # 生成一些标题
    captions = generate_captions(model, dataset, test_dataloader, device, num_samples=5)
    
    # 打印结果
    for item in captions:
        print(f"图像ID: {item['image_id']}")
        print(f"真实标题: {item['true_caption']}")
        print(f"生成标题: {item['generated_caption']}")
        print("-" * 50)