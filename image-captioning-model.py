import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import json
import sqlite3
from torch.utils.data import Dataset, DataLoader

# Transformer模型的位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=100):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加批次维度 [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，不作为模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

# 图像特征投影层
class ImageFeatureEmbedding(nn.Module):
    def __init__(self, image_feature_size=2048, embed_dim=512):
        super(ImageFeatureEmbedding, self).__init__()
        self.embedding = nn.Linear(image_feature_size, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, image_features):
        # image_features: [batch_size, image_feature_size]
        embeddings = self.embedding(image_features)
        embeddings = self.dropout(embeddings)
        embeddings = self.norm(embeddings)
        # 输出: [batch_size, embed_dim]
        return embeddings

# 完整的图像描述模型
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=2048, num_head=8, 
                 num_layers=6, dropout=0.1, max_seq_len=100, image_feature_size=2048):
        super(ImageCaptioningModel, self).__init__()
        
        # 词嵌入层
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 图像特征投影
        self.image_embedding = ImageFeatureEmbedding(image_feature_size, embed_dim)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_head,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # 构建完整的Transformer解码器
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出层 - 预测下一个词
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # 存储模型参数
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
    def create_masks(self, tgt):
        # 创建注意力掩码（确保模型不会提前看到未来的词）
        # tgt: [batch_size, seq_len]
        seq_len = tgt.size(1)
        # 创建下三角掩码
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        # 将掩码放到正确的设备上
        tgt_mask = tgt_mask.to(tgt.device)
        
        # 创建填充掩码（确保模型不会关注PAD标记）
        tgt_padding_mask = (tgt == 0)  # 假设PAD的索引为0
        
        return tgt_mask, tgt_padding_mask
        
    def forward(self, image_features, captions):
        # image_features: [batch_size, image_feature_size]
        # captions: [batch_size, seq_len]
        
        # 处理图像特征
        image_embeddings = self.image_embedding(image_features)  # [batch_size, embed_dim]
        # 扩展为序列长度为1的序列，以便作为memory输入给解码器
        image_embeddings = image_embeddings.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 处理文本输入
        caption_embeddings = self.word_embedding(captions)  # [batch_size, seq_len, embed_dim]
        caption_embeddings = self.positional_encoding(caption_embeddings)
        
        # 创建掩码
        tgt_mask, tgt_padding_mask = self.create_masks(captions)
        
        # 应用Transformer解码器
        # 图像特征作为memory，文本作为tgt
        output = self.transformer_decoder(
            tgt=caption_embeddings,
            memory=image_embeddings,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # 输出层 - 预测下一个词
        output = self.output_layer(output)  # [batch_size, seq_len, vocab_size]
        
        return output
    
    def greedy_decode(self, image_features, start_token_id, end_token_id, max_len):
        """贪婪解码算法生成描述"""
        # image_features: [batch_size, image_feature_size]
        batch_size = image_features.shape[0]
        device = next(self.parameters()).device
        
        # 处理图像特征
        image_embeddings = self.image_embedding(image_features)  # [batch_size, embed_dim]
        # 扩展为序列长度为1的序列
        image_embeddings = image_embeddings.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 初始化输出序列，以start_token开始
        output_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long).to(device)
        
        # 逐步解码
        for i in range(max_len-1):
            # 获取当前序列的输出预测
            caption_embeddings = self.word_embedding(output_ids)  # [batch_size, curr_len, embed_dim]
            caption_embeddings = self.positional_encoding(caption_embeddings)
            
            tgt_mask = torch.triu(torch.ones(i+1, i+1) * float('-inf'), diagonal=1).to(device)
            
            # 应用Transformer解码器
            output = self.transformer_decoder(
                tgt=caption_embeddings,
                memory=image_embeddings,
                tgt_mask=tgt_mask
            )
            
            # 获取下一个词的预测
            logits = self.output_layer(output[:, -1, :])  # [batch_size, vocab_size]
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # 将预测的词添加到输出序列中
            output_ids = torch.cat([output_ids, next_tokens], dim=1)
            
            # 检查是否所有序列都已经生成了结束标记
            if (next_tokens == end_token_id).all():
                break
                
        return output_ids
    
    def beam_search_decode(self, image_features, start_token_id, end_token_id, max_len, beam_size=3):
        """使用束搜索解码算法生成描述"""
        # image_features: [1, image_feature_size]
        device = next(self.parameters()).device
        
        # 确保batch_size为1
        assert image_features.shape[0] == 1, "Beam search only supports batch_size=1"
        
        # 处理图像特征
        image_embeddings = self.image_embedding(image_features)  # [1, embed_dim]
        image_embeddings = image_embeddings.unsqueeze(1)  # [1, 1, embed_dim]
        
        # 开始标记
        start_token = torch.full((1, 1), start_token_id, dtype=torch.long).to(device)
        
        # 初始状态
        caption_embeddings = self.word_embedding(start_token)  # [1, 1, embed_dim]
        caption_embeddings = self.positional_encoding(caption_embeddings)
        
        # 第一步输出
        output = self.transformer_decoder(
            tgt=caption_embeddings,
            memory=image_embeddings
        )
        logits = self.output_layer(output[:, -1, :])  # [1, vocab_size]
        
        # 计算前beam_size个候选的概率和序列
        log_probs, indices = torch.topk(F.log_softmax(logits, dim=-1), beam_size)  # [1, beam_size]
        beams = torch.cat([start_token.repeat(beam_size, 1), indices.view(beam_size, 1)], dim=1)  # [beam_size, 2]
        log_probs = log_probs.view(beam_size)  # [beam_size]
        
        # 束搜索循环
        for i in range(max_len-2):
            # 存储候选
            candidates = []
            
            # 对每个当前束进行评估
            for beam_idx in range(min(beam_size, beams.size(0))):
                curr_beam = beams[beam_idx]  # [curr_len]
                curr_log_prob = log_probs[beam_idx]
                
                # 如果当前序列已经结束，则将其添加到候选中并继续
                if curr_beam[-1].item() == end_token_id:
                    candidates.append((curr_log_prob, curr_beam))
                    continue
                    
                # 计算当前序列的下一步预测
                caption_embeddings = self.word_embedding(curr_beam.unsqueeze(0))  # [1, curr_len, embed_dim]
                caption_embeddings = self.positional_encoding(caption_embeddings)
                
                tgt_mask = torch.triu(torch.ones(curr_beam.size(0), curr_beam.size(0)) * float('-inf'), diagonal=1).to(device)
                
                output = self.transformer_decoder(
                    tgt=caption_embeddings,
                    memory=image_embeddings,
                    tgt_mask=tgt_mask
                )
                
                logits = self.output_layer(output[:, -1, :])  # [1, vocab_size]
                next_log_probs, next_indices = torch.topk(F.log_softmax(logits, dim=-1), beam_size)  # [1, beam_size]
                
                # 将所有可能的下一步添加到候选中
                for k in range(beam_size):
                    next_token = next_indices[0, k].item()
                    next_log_prob = next_log_probs[0, k].item()
                    new_beam = torch.cat([curr_beam, torch.tensor([next_token], device=device)])
                    new_log_prob = curr_log_prob + next_log_prob
                    candidates.append((new_log_prob, new_beam))
            
            # 如果没有候选，则跳出循环
            if not candidates:
                break
                
            # 按照概率排序候选
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # 选择前beam_size个候选
            top_candidates = candidates[:beam_size]
            
            # 更新beams和log_probs
            beams = torch.stack([c[1] for c in top_candidates])
            log_probs = torch.tensor([c[0] for c in top_candidates], device=device)
            
            # 如果所有序列都已经结束，则跳出循环
            if all(b[-1].item() == end_token_id for b in beams):
                break
                
        # 返回概率最高的序列
        return beams[0].unsqueeze(0)  # [1, seq_len]

# 数据集类
class ImageCaptioningDataset(Dataset):
    def __init__(self, db_path, vocab_path, image_feature_table='image_features_resnet50',
                 max_seq_len=100, transform=None):
        super(ImageCaptioningDataset, self).__init__()
        self.db_path = db_path
        self.max_seq_len = max_seq_len
        self.transform = transform
        
        # 加载词汇表
        with open(vocab_path, 'r') as f:
            self.word_to_idx = json.load(f)['word_to_idx']
            
        # 特殊标记的索引
        self.start_token_id = self.word_to_idx['<START>']
        self.end_token_id = self.word_to_idx['<END>']
        self.pad_token_id = self.word_to_idx['<PAD>']
        self.unk_token_id = self.word_to_idx['<UNK>']
        
        # 获取数据集大小
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取所有有特征和标题的图像ID
        query = f"""
        SELECT i.id, COUNT(c.id) as caption_count
        FROM images i
        JOIN {image_feature_table} f ON i.id = f.image_id
        JOIN captions c ON i.id = c.image_id
        GROUP BY i.id
        """
        cursor.execute(query)
        self.image_caption_counts = cursor.fetchall()
        
        conn.close()
    
    def __len__(self):
        return sum([count for _, count in self.image_caption_counts])
    
    def __getitem__(self, idx):
        # 找到对应的图像ID和标题索引
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 计算对应的图像ID和标题索引
        image_idx = 0
        caption_idx = idx
        
        for img_id, count in self.image_caption_counts:
            if caption_idx < count:
                image_idx = img_id
                break
            caption_idx -= count
        
        # 获取图像特征
        cursor.execute(f"SELECT features FROM image_features_resnet50 WHERE image_id = ?", (image_idx,))
        feature_blob = cursor.fetchone()[0]
        
        # 解析特征BLOB为numpy数组
        image_feature = np.frombuffer(feature_blob, dtype=np.float32)
        
        # 获取对应的标题
        cursor.execute("""
        SELECT caption FROM captions 
        WHERE image_id = ? 
        ORDER BY id
        LIMIT 1 OFFSET ?
        """, (image_idx, caption_idx))
        
        caption = cursor.fetchone()[0]
        conn.close()
        
        # 处理标题 - 分词并转换为索引
        caption_tokens = self._tokenize_caption(caption)
        
        # 转换为tensor
        image_feature = torch.tensor(image_feature, dtype=torch.float)
        caption_indices = torch.tensor(caption_tokens, dtype=torch.long)
        
        return image_feature, caption_indices
    
    def _tokenize_caption(self, caption):
        # 简单的空格分词
        words = caption.lower().split()
        
        # 添加开始和结束标记
        tokens = [self.start_token_id]
        
        # 将词转换为索引
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.unk_token_id)
                
        # 添加结束标记
        tokens.append(self.end_token_id)
        
        # 如果序列太长，则截断
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len-1] + [self.end_token_id]
            
        return tokens
    
    def collate_fn(self, batch):
        # 将批次中的样本整理到一起
        image_features, captions = zip(*batch)
        
        # 对标题进行填充
        caption_lengths = [len(cap) for cap in captions]
        max_caption_len = max(caption_lengths)
        
        # 创建填充后的标题张量
        padded_captions = torch.full((len(captions), max_caption_len), 
                                     self.pad_token_id, dtype=torch.long)
        
        # 填充标题
        for i, (cap, cap_len) in enumerate(zip(captions, caption_lengths)):
            padded_captions[i, :cap_len] = cap
        
        # 堆叠图像特征
        image_features = torch.stack(image_features)
        
        return image_features, padded_captions
