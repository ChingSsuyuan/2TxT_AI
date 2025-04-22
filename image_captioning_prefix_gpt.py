#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
import math
import sqlite3
import numpy as np
from typing import Tuple, Optional, Union

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class DatabaseClipDataset(Dataset):
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        return tokens, mask, prefix

    def __init__(self, db_path: str, table_name: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        
        # 从数据库中加载数据
        self.load_from_database(db_path, table_name)
        
        # 处理标题文本
        self.process_captions()
    def load_from_database(self, db_path, table_name):
        """从SQLite数据库加载CLIP嵌入"""
        print(f"Loading data from database: {db_path}, table: {table_name}")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # 获取图像特征和对应的标题
        query = f"""
        SELECT f.id, f.file_name, f.features, c.caption 
        FROM {table_name} f
        LEFT JOIN images i ON f.file_name = i.file_name
        LEFT JOIN captions c ON i.id = c.image_id
        """
        
        try:
            c.execute(query)
            data = c.fetchall()
            conn.close()
            
            # 检查是否找到了数据
            if not data:
                raise ValueError(f"未在 {table_name} 表中找到数据或标题")
            
            # 处理数据
            self.image_ids = []
            self.file_names = []
            self.prefixes = []
            self.captions = []
            
            for id, file_name, features_blob, caption in data:
                # 如果没有标题，则使用文件名生成一个
                if caption is None:
                    caption = os.path.splitext(os.path.basename(file_name))[0].replace('_', ' ')
                    caption = f"An image of {caption}"
                
                self.image_ids.append(id)
                self.file_names.append(file_name)
                
                # 将二进制BLOB转换为numpy数组，然后转换为torch.Tensor
                features = np.frombuffer(features_blob, dtype=np.float32).reshape(1, -1).copy()
                features_tensor = torch.from_numpy(features).to(device)
                self.prefixes.append(features_tensor)
                self.captions.append(caption)
                
            print(f"Loaded {len(self.image_ids)} samples from database")
        
        except sqlite3.OperationalError as e:
            print(f"SQL Error: {e}")
            conn.close()
            raise
    def process_captions(self):
        """处理标题文本,转换为token"""
        print("Processing captions...")
        self.captions_tokens = []
        self.caption2embedding = []
        max_seq_len = 0
        
        for i, caption in enumerate(self.captions):
            self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
            self.caption2embedding.append(i)  # 映射到prefixes的索引
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
        
        # 计算合适的序列长度
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        print(f"Max sequence length: {self.max_seq_len}")


class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
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
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
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
    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):
    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                 clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def load_model(pretrained_path, prefix_length=40, prefix_size=640, mapping_type=MappingType.Transformer, num_layers=8):
    """加载预训练的ClipCaptionModel"""
    model = ClipCaptionModel(
        prefix_length=prefix_length,
        clip_length=prefix_length,  # Set clip_length equal to prefix_length
        prefix_size=prefix_size,
        num_layers=num_layers,
        mapping_type=mapping_type
    )
    
    print(f"Loading pretrained model from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model


def train_gpt2_only(pretrained_path, db_path, train_table, val_table, output_dir,
                   prefix_length=40, prefix_size=640, mapping_type=MappingType.Transformer, num_layers=8,
                   epochs=5, batch_size=5, lr=1e-5, warmup_steps=1000, save_every=1):
    """只训练GPT-2部分，同时冻结prefix映射网络"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预训练模型
    model = load_model(
        pretrained_path, 
        prefix_length=prefix_length,
        prefix_size=prefix_size,
        mapping_type=mapping_type,
        num_layers=num_layers
    )
    model = model.to(device)
    
    # 冻结prefix映射网络的参数
    for param in model.clip_project.parameters():
        param.requires_grad = False
    
    # 确保GPT-2部分的参数可训练
    for param in model.gpt.parameters():
        param.requires_grad = True
    
    # 统计可训练的参数和冻结的参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print(f"冻结参数: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params:.2%})")
    
    # 加载训练数据
    train_dataset = DatabaseClipDataset(
        db_path=db_path, 
        table_name=train_table, 
        prefix_length=prefix_length,
        normalize_prefix=True
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    # 加载验证数据（如果有）
    if val_table:
        val_dataset = DatabaseClipDataset(
            db_path=db_path, 
            table_name=val_table, 
            prefix_length=prefix_length,
            normalize_prefix=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=True
        )
    else:
        val_dataloader = None
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=epochs * len(train_dataloader)
    )
    
    # 记录训练损失
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 开始训练
    model.train()
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch+1}/{epochs}")
        
        # 训练一个epoch
        train_loss = 0
        progress = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens = tokens.to(device)
            mask = mask.to(device)
            prefix = prefix.to(device, dtype=torch.float32)
            
            # 前向传播
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
            
            # 计算损失
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), 
                tokens.flatten(), 
                ignore_index=0
            )
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 更新进度条
            loss_value = loss.item()
            train_loss += loss_value
            progress.set_postfix({"loss": loss_value})
            progress.update()
            
            # 定期保存模型
            if (idx + 1) % 500 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "GPT2_model_latest.pt")
                )
        
        progress.close()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} - Avg train loss: {avg_train_loss:.4f}")
        
        # 验证
        if val_dataloader:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for tokens, mask, prefix in val_dataloader:
                    tokens = tokens.to(device)
                    mask = mask.to(device)
                    prefix = prefix.to(device, dtype=torch.float32)
                    
                    # 前向传播
                    outputs = model(tokens, prefix, mask)
                    logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
                    
                    # 计算损失
                    loss = nnf.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), 
                        tokens.flatten(), 
                        ignore_index=0
                    )
                    
                    val_loss += loss.item()
            
            # 计算平均验证损失
            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1} - Validation loss: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "GPT2_model_best.pt")
                )
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
            model.train()
        
        # 定期保存模型
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"GPT2_model-{epoch+1:03d}.pt")
            )
        
        # 保存损失历史
        loss_history = {
            'train': train_losses,
            'val': val_losses
        }
        
        with open(os.path.join(output_dir, "loss_history.json"), 'w') as f:
            json.dump(loss_history, f, indent=4)
    
    # 保存最终模型
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "GPT2_model_final.pt")
    )
    
    print("GPT-2训练完成！")
    return model


def main():
    parser = argparse.ArgumentParser(description='只训练GPT-2部分的CLIP+GPT2模型')
    parser.add_argument('--pretrained_model', type=str, default='checkpoints/CLIP_GPT-014.pt',
                        help='预训练模型路径')
    parser.add_argument('--db_path', type=str, default='coco_image_title_data/image_title_database.db',
                        help='数据库路径')
    parser.add_argument('--train_table', type=str, default='image_features_clip_train',
                        help='训练数据表名')
    parser.add_argument('--val_table', type=str, default='image_features_clip_val',
                        help='验证数据表名')
    parser.add_argument('--output_dir', type=str, default='./gpt2_checkpoints',
                        help='输出目录')
    parser.add_argument('--prefix_length', type=int, default=40,
                        help='前缀长度')
    parser.add_argument('--prefix_size', type=int, default=640,
                        help='CLIP特征维度，RN50x4为640')
    parser.add_argument('--mapping_type', type=str, default='transformer',
                        choices=['mlp', 'transformer'],
                        help='映射类型')
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Transformer层数')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='预热步数')
    parser.add_argument('--save_every', type=int, default=1,
                        help='每隔几个epoch保存一次模型')
    parser.add_argument('--combine_train_val', action='store_true',
                        help='合并训练集和验证集进行训练')
    parser.add_argument('--train_final', action='store_true',
                        help='使用所有数据（训练集+验证集+测试集）训练最终模型')
    
    args = parser.parse_args()
    
    # 转换mapping_type为枚举类型
    mapping_type = MappingType.MLP if args.mapping_type == 'mlp' else MappingType.Transformer
    
    # 如果选择合并训练集和验证集
    if args.combine_train_val:
        print("合并训练集和验证集进行训练...")
        
        # 创建合并表
        conn = sqlite3.connect(args.db_path)
        c = conn.cursor()
        
        # 检查表是否已存在，如果存在则删除
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_features_clip_train_val'")
        if c.fetchone():
            c.execute("DROP TABLE image_features_clip_train_val")
        
        # 创建合并表
        c.execute("""
        CREATE TABLE image_features_clip_train_val AS
        SELECT * FROM image_features_clip_train
        UNION ALL
        SELECT * FROM image_features_clip_val
        """)
        
        conn.commit()
        conn.close()
        
        args.train_table = 'image_features_clip_train_val'
        args.val_table = ''  # 不使用验证集
    
    # 如果选择训练最终模型（使用所有数据）
    if args.train_final:
        print("使用所有数据（训练集+验证集+测试集）训练最终模型...")
        
        # 创建合并表
        conn = sqlite3.connect(args.db_path)
        c = conn.cursor()
        
        # 检查表是否已存在，如果存在则删除
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_features_clip_all'")
        if c.fetchone():
            c.execute("DROP TABLE image_features_clip_all")
        
        # 创建合并表
        c.execute("""
        CREATE TABLE image_features_clip_all AS
        SELECT * FROM image_features_clip_train
        UNION ALL
        SELECT * FROM image_features_clip_val
        UNION ALL
        SELECT * FROM image_features_clip_test
        """)
        
        conn.commit()
        conn.close()
        
        args.train_table = 'image_features_clip_all'
        args.val_table = ''  # 不使用验证集
        args.output_dir = './final_model'  # 更改输出目录
    
    # 训练GPT-2
    train_gpt2_only(
        pretrained_path=args.pretrained_model,
        db_path=args.db_path,
        train_table=args.train_table,
        val_table=args.val_table,
        output_dir=args.output_dir,
        prefix_length=args.prefix_length,
        prefix_size=args.prefix_size,
        mapping_type=mapping_type,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every
    )


if __name__ == "__main__":
    main()