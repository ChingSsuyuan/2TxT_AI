#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from collections import Counter
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    nltk.download('stopwords', quiet=True)
except:
    print("无法下载NLTK资源，将使用基本的关键词提取")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

BATCH_SIZE = 32
EMBEDDING_DIM = 512
HIDDEN_DIM = 2048
NUM_DECODER_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1
MAX_CAPTION_LENGTH = 20
FEATURE_DIM = 2048
GRAD_CLIP = 5.0
ALPHA_C = 1.0

class COCOImageCaptionDataset(Dataset):
    def __init__(self, db_path, max_len=MAX_CAPTION_LENGTH, split='train'):
        self.db_path = db_path
        self.max_len = max_len
        self.split = split
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            self.word_to_idx = {}
            self.idx_to_word = {}
            self.special_tokens = {}
            
            cursor.execute("SELECT id, word, special_token FROM vocabulary")
            vocabulary_data = cursor.fetchall()
            
            cursor.execute("SELECT word_id, index_id, special_token FROM word_indices")
            word_indices_data = cursor.fetchall()
            
            for word_id, index_id, special_token in word_indices_data:
                for vocab_id, word, _ in vocabulary_data:
                    if vocab_id == word_id:
                        self.idx_to_word[index_id] = word
                        self.word_to_idx[word] = index_id
                        
                        if special_token == 1:
                            self.special_tokens[word] = index_id
                        break
            
            assert '<PAD>' in self.special_tokens, "找不到PAD标记"
            assert '<UNK>' in self.special_tokens, "找不到UNK标记"
            assert '<START>' in self.special_tokens, "找不到START标记"
            assert '<END>' in self.special_tokens, "找不到END标记"
            
            self.pad_idx = self.special_tokens['<PAD>']
            self.unk_idx = self.special_tokens['<UNK>']
            self.start_idx = self.special_tokens['<START>']
            self.end_idx = self.special_tokens['<END>']
            
            cursor.execute("SELECT COUNT(*) FROM captions")
            self.num_captions = cursor.fetchone()[0]
            
            cursor.execute("SELECT id, image_id, caption FROM captions")
            self.captions_data = cursor.fetchall()
            
            cursor.execute("SELECT image_id, features FROM image_features_resnet50")
            self.features_data = {}
            for img_id, feature_blob in cursor.fetchall():
                self.features_data[img_id] = self._parse_features(feature_blob)
        
        self.vocab_size = len(self.idx_to_word)
        print(f"数据集加载完成: {self.num_captions}个标题, 词汇量{self.vocab_size}")
        
        all_image_ids = list(set(img_id for _, img_id, _ in self.captions_data))
        
        import random
        random.seed(42)
        random.shuffle(all_image_ids)
        
        num_train = int(len(all_image_ids) * 0.8)
        num_val = int(len(all_image_ids) * 0.1)
        
        if split == 'train':
            self.image_ids = all_image_ids[:num_train]
        elif split == 'val':
            self.image_ids = all_image_ids[num_train:num_train+num_val]
        else:
            self.image_ids = all_image_ids[num_train+num_val:]
            
        self.captions_data = [item for item in self.captions_data if item[1] in self.image_ids]
        print(f"{split}集: {len(self.captions_data)}条标题, {len(self.image_ids)}张图片")
        
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 
                             'were', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 
                             'with', 'by', 'about', 'of', 'from'}
        
        self.stemmer = PorterStemmer()
    
    def __len__(self):
        return len(self.captions_data)
    
    def __getitem__(self, idx):
        caption_id, image_id, caption = self.captions_data[idx]
        
        if image_id not in self.features_data:
            raise ValueError(f"找不到image_id为{image_id}的特征")
        
        features = self.features_data[image_id]
        image_features = torch.tensor(features, dtype=torch.float32)
        
        tokenized_caption, caption_length = self._tokenize_caption(caption)
        
        if self.split != 'train':
            all_captions = []
            for _, img_id, cap in self.captions_data:
                if img_id == image_id:
                    all_captions.append(self._tokenize_caption(cap)[0])
            all_captions = torch.stack(all_captions)
        else:
            all_captions = None
        
        return {
            'image_id': image_id,
            'image_features': image_features,
            'caption': caption,
            'tokenized_caption': tokenized_caption,
            'caption_length': caption_length,
            'all_captions': all_captions
        }
    
    def _parse_features(self, feature_blob):
        try:
            if isinstance(feature_blob, bytes):
                features = np.frombuffer(feature_blob, dtype=np.float32)
                
                if len(features) == FEATURE_DIM:
                    return features
                elif len(features) > FEATURE_DIM:
                    return features[:FEATURE_DIM]
                else:
                    print(f"警告: 特征长度 ({len(features)}) 小于预期 ({FEATURE_DIM})，将返回随机数组")
            
            return np.random.randn(FEATURE_DIM)
        except Exception as e:
            print(f"警告: 无法解析特征数据，错误: {str(e)}")
            return np.random.randn(FEATURE_DIM)
    
    def _tokenize_caption(self, caption):
        tokens = caption.lower().split()
        
        token_indices = [self.start_idx]
        for token in tokens:
            if token in self.word_to_idx:
                token_indices.append(self.word_to_idx[token])
            else:
                token_indices.append(self.unk_idx)
        
        token_indices.append(self.end_idx)
        caption_length = len(token_indices)
        
        if caption_length < self.max_len:
            token_indices.extend([self.pad_idx] * (self.max_len - caption_length))
        else:
            token_indices = token_indices[:self.max_len-1] + [self.end_idx]
            caption_length = self.max_len
        
        vocab_size = len(self.word_to_idx)
        token_indices = [min(idx, vocab_size-1) for idx in token_indices]
        
        return torch.tensor(token_indices, dtype=torch.long), caption_length
    
    def extract_keywords(self, text, max_keywords=3):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        
        filtered_words = [word for word in words 
                         if word not in self.stopwords 
                         and word not in self.special_tokens
                         and len(word) > 2]
        
        if not filtered_words and words:
            return words[:max_keywords]
        
        word_freq = Counter(filtered_words)
        
        keywords = [word for word, _ in word_freq.most_common(max_keywords)]
        
        return keywords
    
    def calculate_keyword_match(self, generated_keywords, true_keywords):
        stemmed_generated = [self.stemmer.stem(word) for word in generated_keywords]
        stemmed_true = [self.stemmer.stem(word) for word in true_keywords]
        
        matches = [word for word in stemmed_generated if word in stemmed_true]
        
        if len(stemmed_true) == 0:
            return 0.0, []
        
        match_score = len(matches) / min(len(stemmed_generated), len(stemmed_true))
        
        matched_keywords = []
        for i, stem in enumerate(stemmed_generated):
            if stem in stemmed_true:
                matched_keywords.append(generated_keywords[i])
        
        return match_score, matched_keywords
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def collate_fn(self, batch):
        image_ids = [item['image_id'] for item in batch]
        image_features = [item['image_features'] for item in batch]
        captions = [item['caption'] for item in batch]
        tokenized_captions = [item['tokenized_caption'] for item in batch]
        caption_lengths = [item['caption_length'] for item in batch]
        
        image_features = torch.stack(image_features, dim=0)
        tokenized_captions = torch.stack(tokenized_captions, dim=0)
        caption_lengths = torch.tensor(caption_lengths, dtype=torch.long)
        
        caption_lengths, sort_ind = caption_lengths.sort(descending=True)
        image_features = image_features[sort_ind]
        tokenized_captions = tokenized_captions[sort_ind]
        
        if batch[0]['all_captions'] is not None:
            all_captions = [item['all_captions'] for item in batch]
            all_captions = [caps[sort_ind] for caps in all_captions]
        else:
            all_captions = None
        
        return {
            'image_ids': [image_ids[i] for i in sort_ind],
            'image_features': image_features,
            'captions': [captions[i] for i in sort_ind],
            'tokenized_captions': tokenized_captions,
            'caption_lengths': caption_lengths,
            'all_captions': all_captions
        }

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.embed_dim)
        
        output = self.output_proj(context)
        
        return output, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EnhancedTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(EnhancedTransformerDecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        residual = tgt
        tgt = self.norm1(tgt)
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = residual + self.dropout(tgt2)
        
        residual = tgt
        tgt = self.norm2(tgt)
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, mask=memory_mask)
        tgt = residual + self.dropout(tgt2)
        
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = residual + self.dropout(self.feed_forward(tgt))
        
        return tgt, self_attn_weights, cross_attn_weights

class EnhancedTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(EnhancedTransformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = tgt
        self_attn_weights = []
        cross_attn_weights = []
        
        for layer in self.layers:
            output, self_attn, cross_attn = layer(output, memory, tgt_mask, memory_mask)
            self_attn_weights.append(self_attn)
            cross_attn_weights.append(cross_attn)
        
        return output, (self_attn_weights, cross_attn_weights)

class EnhancedImageCaptioningTransformer(nn.Module):
    def __init__(self, vocab_size, feature_dim, embedding_dim, hidden_dim, 
                 num_layers, num_heads, dropout, pad_idx):
        super(EnhancedImageCaptioningTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        
        self.feature_projection = nn.Linear(feature_dim, embedding_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        decoder_layer = EnhancedTransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        
        self.transformer_decoder = EnhancedTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.constant_(self.feature_projection.bias, 0)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
        
        nn.init.normal_(self.word_embedding.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.word_embedding.weight[self.pad_idx], 0)
    
    def create_masks(self, target_seq, target_pad_mask=None):
        batch_size, seq_len = target_seq.size()
        
        subsequent_mask = torch.tril(torch.ones(seq_len, seq_len, device=target_seq.device))
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        if target_pad_mask is None:
            target_pad_mask = (target_seq == self.pad_idx)
        
        expanded_pad_mask = target_pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
        
        target_mask = (expanded_pad_mask | (1 - subsequent_mask).bool())
        
        return target_mask, target_pad_mask
    
    def forward(self, image_features, target_captions, target_lengths=None):
        batch_size, seq_len = target_captions.size()
        
        gate = self.feature_gate(image_features)
        projected_features = self.feature_projection(image_features) * gate
        memory = projected_features.unsqueeze(1)
        
        embedded_captions = self.word_embedding(target_captions)
        
        embedded_captions = self.positional_encoding(embedded_captions)
        
        target_pad_mask = (target_captions == self.pad_idx)
        target_mask, _ = self.create_masks(target_captions, target_pad_mask)
        
        output, (self_attn_weights, cross_attn_weights) = self.transformer_decoder(
            tgt=embedded_captions,
            memory=memory,
            tgt_mask=target_mask
        )
        
        logits = self.output_layer(output)
        
        alphas = cross_attn_weights[-1]
        
        return logits, alphas
    
    def generate_caption(self, image_features, max_len=20, temperature=1.0, beam_size=3):
        self.eval()
        
        with torch.no_grad():
            gate = self.feature_gate(image_features)
            projected_features = self.feature_projection(image_features) * gate
            memory = projected_features.unsqueeze(1)
            
            k = beam_size
            complete_seqs = []
            complete_scores = []
            complete_alphas = []
            
            seqs = torch.full((k, 1), self.start_idx, dtype=torch.long, device=image_features.device)
            scores = torch.zeros(k, device=image_features.device)
            all_alphas = [[] for _ in range(k)]
            
            for step in range(max_len - 1):
                current_beam_size = len(seqs)
                
                embedded = self.word_embedding(seqs)
                embedded = self.positional_encoding(embedded)
                
                target_mask, _ = self.create_masks(seqs)
                
                expanded_memory = memory.repeat(current_beam_size, 1, 1)
                
                output, (_, cross_attn) = self.transformer_decoder(
                    tgt=embedded,
                    memory=expanded_memory,
                    tgt_mask=target_mask
                )
                
                output = output[:, -1, :]
                logits = self.output_layer(output)
                
                if temperature > 0:
                    logits = logits / temperature
                
                log_probs = F.log_softmax(logits, dim=-1)
                
                scores = scores.unsqueeze(1) + log_probs
                
                if step == 0:
                    scores = scores[0].unsqueeze(0)
                
                scores, indices = scores.view(-1).topk(k, largest=True, sorted=True)
                
                prev_seq_indices = indices // self.vocab_size
                next_word_indices = indices % self.vocab_size
                
                next_seqs = []
                next_all_alphas = []
                
                for i, (prev_idx, next_word_idx) in enumerate(zip(prev_seq_indices, next_word_indices)):
                    prev_seq = seqs[prev_idx]
                    
                    next_seq = torch.cat([prev_seq, next_word_idx.unsqueeze(0)], dim=0)
                    next_seqs.append(next_seq)
                    
                    prev_alphas = all_alphas[prev_idx].copy()
                    curr_alpha = cross_attn[-1][prev_idx]
                    prev_alphas.append(curr_alpha)
                    next_all_alphas.append(prev_alphas)
                    
                    if next_word_idx.item() == self.end_idx:
                        complete_seqs.append(next_seq)
                        complete_scores.append(scores[i])
                        complete_alphas.append(prev_alphas)
                    
                if len(complete_seqs) == k:
                    break
                
                incomplete_indices = [i for i, next_word_idx in enumerate(next_word_indices) 
                                      if next_word_idx.item() != self.end_idx]
                
                if not incomplete_indices:
                    break
                
                seqs = [next_seqs[i] for i in incomplete_indices]
                scores = scores[incomplete_indices]
                all_alphas = [next_all_alphas[i] for i in incomplete_indices]
                
                if len(seqs) < k:
                    num_fill = k - len(seqs)
                    seqs.extend([seqs[0] for _ in range(num_fill)])
                    scores = torch.cat([scores, scores[0].repeat(num_fill)])
                    all_alphas.extend([all_alphas[0] for _ in range(num_fill)])
                
                seqs = torch.stack(seqs)
            
            if len(complete_seqs) == 0:
                best_idx = scores.argmax()
                best_seq = seqs[best_idx]
                best_alphas = all_alphas[best_idx]
            else:
                best_idx = complete_scores.index(max(complete_scores))
                best_seq = complete_seqs[best_idx]
                best_alphas = complete_alphas[best_idx]
            
            best_caption = best_seq.tolist()
            
            return best_caption, best_alphas

class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps=0.1, ignore_index=0, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, output, target):
        vocab_size = output.size(-1)
        
        loss = self.criterion(output, target)
        
        mask = (target != self.ignore_index).float()
        
        if self.eps == 0:
            if self.reduction == 'mean':
                return (loss * mask).sum() / mask.sum().clamp(min=1e-12)
            else:
                return (loss * mask).sum()
        
        smooth_loss = -torch.log_softmax(output, dim=-1)
        
        smooth_target = torch.ones_like(output) * self.eps / (vocab_size - 1)
        
        smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        
        smooth_loss = torch.sum(smooth_target * smooth_loss, dim=-1)
        
        smooth_loss = smooth_loss * mask
        
        if self.reduction == 'mean':
            return smooth_loss.sum() / mask.sum().clamp(min=1e-12)
        else:
            return smooth_loss.sum()

def accuracy(scores, targets, k=5):
    batch_size = targets.size(0)
    
    _, pred = scores.topk(k, 1, True, True)
    correct = pred.eq(targets.unsqueeze(1).expand_as(pred))
    
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() / batch_size

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def train_epoch(model, dataloader, optimizer, criterion, device, alpha_c=1.0, grad_clip=None):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="训练中")):
        image_features = batch['image_features'].to(device)
        tokenized_captions = batch['tokenized_captions'].to(device)
        caption_lengths = batch['caption_lengths'].to(device)
        
        input_caption = tokenized_captions[:, :-1]
        target_caption = tokenized_captions[:, 1:]
        
        output, alphas = model(image_features, input_caption)
        
        decode_lengths = [min(length - 1, input_caption.size(1)) for length in caption_lengths]
        max_decode_length = max(decode_lengths)
        
        output = output[:, :max_decode_length, :]
        target_caption = target_caption[:, :max_decode_length]
        
        packed_output = pack_padded_sequence(output, decode_lengths, batch_first=True)
        packed_target = pack_padded_sequence(target_caption, decode_lengths, batch_first=True)
        
        packed_output_data, _ = pad_packed_sequence(packed_output, batch_first=True)
        packed_target_data, _ = pad_packed_sequence(packed_target, batch_first=True)
        
        reshaped_output = packed_output_data.contiguous().view(-1, model.vocab_size)
        reshaped_target = packed_target_data.view(-1)
        
        loss = criterion(reshaped_output, reshaped_target)
        
        if alpha_c > 0:
            attention_sum = alphas.sum(dim=2)
            alpha_reg = alpha_c * ((1.0 - attention_sum) ** 2).mean()
            loss += alpha_reg
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        
        optimizer.step()
        
        top5_acc = accuracy(reshaped_output, reshaped_target, k=5)
        
        epoch_loss += loss.item() * sum(decode_lengths)
        epoch_acc += top5_acc * sum(decode_lengths)
        total_samples += sum(decode_lengths)
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Top-5 Acc: {top5_acc:.4f}")
    
    epoch_loss /= total_samples
    epoch_acc /= total_samples
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, alpha_c=1.0):
    model.eval()
    val_loss = 0
    val_acc = 0
    total_samples = 0
    
    all_refs = []
    all_hyps = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="验证中")):
            image_features = batch['image_features'].to(device)
            tokenized_captions = batch['tokenized_captions'].to(device)
            caption_lengths = batch['caption_lengths'].to(device)
            
            input_caption = tokenized_captions[:, :-1]
            target_caption = tokenized_captions[:, 1:]
            
            output, alphas = model(image_features, input_caption)
            
            decode_lengths = [min(length - 1, input_caption.size(1)) for length in caption_lengths]
            max_decode_length = max(decode_lengths)
            
            output = output[:, :max_decode_length, :]
            target_caption = target_caption[:, :max_decode_length]
            
            packed_output = pack_padded_sequence(output, decode_lengths, batch_first=True)
            packed_target = pack_padded_sequence(target_caption, decode_lengths, batch_first=True)
            
            packed_output_data, _ = pad_packed_sequence(packed_output, batch_first=True)
            packed_target_data, _ = pad_packed_sequence(packed_target, batch_first=True)
            
            reshaped_output = packed_output_data.contiguous().view(-1, model.vocab_size)
            reshaped_target = packed_target_data.view(-1)
            
            loss = criterion(reshaped_output, reshaped_target)
            
            if alpha_c > 0:
                attention_sum = alphas.sum(dim=2)
                alpha_reg = alpha_c * ((1.0 - attention_sum) ** 2).mean()
                loss += alpha_reg
            
            top5_acc = accuracy(reshaped_output, reshaped_target, k=5)
            
            val_loss += loss.item() * sum(decode_lengths)
            val_acc += top5_acc * sum(decode_lengths)
            total_samples += sum(decode_lengths)
            
            _, preds = torch.max(output, dim=2)
            preds = preds.cpu().tolist()
            
            targets = target_caption.cpu().tolist()
            
            all_caption_sets = batch['all_captions']
            if all_caption_sets is not None:
                for i, (pred, target, length) in enumerate(zip(preds, targets, decode_lengths)):
                    pred_words = []
                    for idx in pred[:length]:
                        if idx not in [model.pad_idx, model.start_idx, model.end_idx]:
                            pred_words.append(model.idx_to_word[idx])
                    
                    references = []
                    for caption_tensor in all_caption_sets:
                        caption = caption_tensor[i].tolist()
                        ref_words = []
                        for idx in caption:
                            if idx not in [model.pad_idx, model.start_idx, model.end_idx]:
                                ref_words.append(model.idx_to_word[idx])
                        references.append(ref_words)
                    
                    all_hyps.append(pred_words)
                    all_refs.append(references)
    
    val_loss /= total_samples
    val_acc /= total_samples
    
    try:
        from nltk.translate.bleu_score import corpus_bleu
        bleu4 = corpus_bleu(all_refs, all_hyps)
    except:
        print("警告: 无法计算BLEU分数")
        bleu4 = 0.0
    
    return val_loss, val_acc, bleu4

def evaluate_keyword_matching(model, dataset, output_dir, num_samples=50, beam_size=3):
    model.eval()
    
    test_dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0
    )
    
    results = []
    total_score = 0
    total_bleu = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="评估中")):
            if i >= num_samples:
                break
            
            image_features = batch['image_features'].to(device)
            true_caption = batch['caption'][0]
            tokenized_caption = batch['tokenized_caption'].to(device)
            
            generated_tokens, alphas = model.generate_caption(
                image_features, 
                max_len=MAX_CAPTION_LENGTH,
                beam_size=beam_size
            )
            
            generated_words = []
            for idx in generated_tokens:
                if idx not in [model.pad_idx, model.start_idx, model.end_idx]:
                    if idx in model.idx_to_word:
                        generated_words.append(model.idx_to_word[idx])
                    else:
                        generated_words.append(model.idx_to_word[model.unk_idx])
            
            generated_caption = ' '.join(generated_words)
            
            true_keywords = dataset.extract_keywords(true_caption, max_keywords=5)
            generated_keywords = dataset.extract_keywords(generated_caption, max_keywords=5)
            
            match_score, matched_keywords = dataset.calculate_keyword_match(
                generated_keywords, true_keywords
            )
            
            try:
                from nltk.translate.bleu_score import sentence_bleu
                ref_words = true_caption.lower().split()
                hyp_words = generated_caption.lower().split()
                bleu = sentence_bleu([ref_words], hyp_words)
            except:
                bleu = 0.0
            
            results.append({
                'image_id': batch['image_id'].item(),
                'true_caption': true_caption,
                'true_keywords': true_keywords,
                'generated_caption': generated_caption,
                'generated_keywords': generated_keywords,
                'matched_keywords': matched_keywords,
                'match_score': match_score,
                'bleu_score': bleu
            })
            
            total_score += match_score
            total_bleu += bleu
    
    avg_score = total_score / len(results) if results else 0
    avg_bleu = total_bleu / len(results) if results else 0
    
    print("\nKeyword Matching Results:")
    print("=" * 50)
    print(f"Average Keyword Match Score: {avg_score:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    
    result_file = os.path.join(output_dir, "keyword_matching_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Keyword Matching Results:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Average Keyword Match Score: {avg_score:.4f}\n")
        f.write(f"Average BLEU Score: {avg_bleu:.4f}\n\n")
        
        for i, item in enumerate(results):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Image ID: {item['image_id']}\n")
            f.write(f"原始标题: {item['true_caption']}\n")
            f.write(f"关键词: {', '.join(item['true_keywords'])}\n")
            f.write(f"生成标题: {item['generated_caption']}\n")
            f.write(f"生成关键词: {', '.join(item['generated_keywords'])}\n")
            f.write(f"匹配关键词: {', '.join(item['matched_keywords'])}\n")
            f.write(f"匹配分数: {item['match_score']:.4f}\n")
            f.write(f"BLEU分数: {item['bleu_score']:.4f}\n")
            f.write("-" * 50 + "\n")
    
    print(f"结果已保存到: {result_file}")
    return results, avg_score, avg_bleu

def train_model(db_path, output_dir, num_epochs=10, learning_rate=0.0001, alpha_c=1.0, grad_clip=5.0):
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset = COCOImageCaptionDataset(db_path=db_path, split='train')
    val_dataset = COCOImageCaptionDataset(db_path=db_path, split='val')
    
    print(f"训练集大小: {len(train_dataset)} 条标题")
    print(f"验证集大小: {len(val_dataset)} 条标题")
    
    actual_batch_size = min(BATCH_SIZE, len(train_dataset))
    if actual_batch_size < BATCH_SIZE:
        print(f"调整batch_size为 {actual_batch_size} (原计划: {BATCH_SIZE})")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=actual_batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=train_dataset.collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=actual_batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=val_dataset.collate_fn
    )
    
    vocab_size = train_dataset.get_vocab_size()
    print(f"词汇表大小: {vocab_size}")
    
    model = EnhancedImageCaptioningTransformer(
        vocab_size=vocab_size,
        feature_dim=FEATURE_DIM,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        pad_idx=train_dataset.pad_idx
    ).to(device)
    
    model.idx_to_word = train_dataset.idx_to_word
    model.pad_idx = train_dataset.pad_idx
    model.start_idx = train_dataset.start_idx
    model.end_idx = train_dataset.end_idx
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = LabelSmoothingLoss(eps=0.1, ignore_index=train_dataset.pad_idx)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    best_val_loss = float('inf')
    best_bleu = 0.0
    epochs_since_improvement = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            criterion, 
            device,
            alpha_c=alpha_c,
            grad_clip=grad_clip
        )
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        val_loss, val_acc, bleu = validate(
            model, 
            val_dataloader, 
            criterion, 
            device,
            alpha_c=alpha_c
        )
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, BLEU-4: {bleu:.4f}")
        
        scheduler.step(val_loss)
        
        is_best_loss = val_loss < best_val_loss
        is_best_bleu = bleu > best_bleu
        
        if is_best_loss:
            best_val_loss = val_loss
        
        if is_best_bleu:
            best_bleu = bleu
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        print(f"上次BLEU改进后的轮次数: {epochs_since_improvement}")
        
        if is_best_loss or is_best_bleu:
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'bleu': bleu,
                'vocab_size': vocab_size,
                'idx_to_word': train_dataset.idx_to_word,
                'word_to_idx': train_dataset.word_to_idx,
                'special_tokens': train_dataset.special_tokens,
            }, checkpoint_path)
            
            print(f"已保存最佳模型, BLEU: {best_bleu:.4f}, 验证损失: {best_val_loss:.4f}")
        
        checkpoint_path = os.path.join(output_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'bleu': bleu,
            'vocab_size': vocab_size,
            'idx_to_word': train_dataset.idx_to_word,
            'word_to_idx': train_dataset.word_to_idx,
            'special_tokens': train_dataset.special_tokens,
        }, checkpoint_path)
        
        if epochs_since_improvement >= 10:
            print("10轮没有改进，停止训练")
            break
    
    print("训练完成!")
    return model

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab_size = checkpoint['vocab_size']
    idx_to_word = checkpoint['idx_to_word']
    word_to_idx = checkpoint.get('word_to_idx', {})
    special_tokens = checkpoint.get('special_tokens', {})
    
    model = EnhancedImageCaptioningTransformer(
        vocab_size=vocab_size,
        feature_dim=FEATURE_DIM,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        pad_idx=special_tokens.get('<PAD>', 0)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.idx_to_word = idx_to_word
    model.pad_idx = special_tokens.get('<PAD>', 0)
    model.start_idx = special_tokens.get('<START>', 1)
    model.end_idx = special_tokens.get('<END>', 2)
    
    return model

if __name__ == "__main__":
    db_path = "coco_image_title_data/image_title_database.db"
    output_dir = "model_output"
    
    checkpoint_path = os.path.join(output_dir, 'best_model.pth')
    train_new_model = True
    
    if train_new_model:
        print("训练新模型...")
        model = train_model(
            db_path=db_path,
            output_dir=output_dir,
            num_epochs=10,
            learning_rate=0.0001,
            alpha_c=ALPHA_C,
            grad_clip=GRAD_CLIP
        )
    else:
        if os.path.exists(checkpoint_path):
            print(f"加载模型从 {checkpoint_path}...")
            model = load_model(checkpoint_path, device)
        else:
            print(f"模型文件 {checkpoint_path} 不存在, 将训练新模型")
            model = train_model(
                db_path=db_path,
                output_dir=output_dir,
                num_epochs=10,
                learning_rate=0.0001,
                alpha_c=ALPHA_C,
                grad_clip=GRAD_CLIP
            )
    
    test_dataset = COCOImageCaptionDataset(db_path=db_path, split='test')
    print(f"测试集: {len(test_dataset)} 条标题")
    
    print("评估关键词匹配性能...")
    results, avg_match_score, avg_bleu = evaluate_keyword_matching(
        model=model,
        dataset=test_dataset,
        output_dir=output_dir,
        num_samples=50,
        beam_size=3
    )
    
    print(f"评估完成! 平均关键词匹配分数: {avg_match_score:.4f}, 平均BLEU分数: {avg_bleu:.4f}")
    
    print("\n示例预测:")
    for i, result in enumerate(results[:5]):
        print(f"示例 {i+1}:")
        print(f"原始标题: {result['true_caption']}")
        print(f"关键词: {', '.join(result['true_keywords'])}")
        print(f"生成标题: {result['generated_caption']}")
        print(f"生成关键词: {', '.join(result['generated_keywords'])}")
        print(f"匹配分数: {result['match_score']:.4f}")
        print(f"BLEU分数: {result['bleu_score']:.4f}")
        print("-" * 50)