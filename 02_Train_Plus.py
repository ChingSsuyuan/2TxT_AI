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
import random
import numpy as np
from typing import Tuple, Optional, Union


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipProDataset(Dataset):

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
        prefix = self.prefixes[self.caption2embedding[item]]  # Use mapping from caption to embedding
        
        # 如果启用了特征噪声，添加噪声
        if self.feature_noise_scale > 0:
            noise = torch.randn_like(prefix) * self.feature_noise_scale
            prefix = prefix + noise
            
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False, feature_noise_scale=0.0):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.feature_noise_scale = feature_noise_scale
        
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        
        # Handle the structure of CLIP_Pro_train_merged.pkl
        print("Loading CLIP_Pro data...")
        self.prefixes = all_data["clip_embedding"]
        
        # Get captions from the structure based on 01_Database_to_Clip.py
        captions_raw = all_data["captions"]
        self.captions = [caption['caption'] for caption in captions_raw]
        
        print(f"Data size is {len(self.prefixes)}")
        print(f"Found {len(self.captions)} captions")
        sys.stdout.flush()
        
        # Process captions
        tokens_cache_path = f"{data_path[:-4]}_tokens.pkl"
        
        if os.path.isfile(tokens_cache_path):
            print(f"Loading cached tokens from {tokens_cache_path}")
            with open(tokens_cache_path, 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            print("Tokenizing captions...")
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            
            # Save tokens to cache
            with open(tokens_cache_path, 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
            print(f"Tokens cached to {tokens_cache_path}")
            
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        print(f"Maximum sequence length: {self.max_seq_len}")


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


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        # 特殊处理MappingType类型，将其转换为字符串
        if key == 'mapping_type' and isinstance(item, MappingType):
            config[key] = item.value  # 使用枚举的值（字符串）而不是枚举对象
        else:
            config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def set_seed(seed):
    """设置随机种子确保可重复性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(dataset: ClipProDataset, val_dataset: Optional[ClipProDataset], model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    # 检查CUDA可用性，但不强制使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    print(f"使用设备: {device}")
    
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    
    # 使用命令行指定的学习率
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 创建验证集的DataLoader（如果有验证集）
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"验证集大小: {len(val_dataset)} 样本")
    
    # 计算总训练步数，用于学习率调度
    total_steps = len(train_dataloader) * epochs
    warmup_steps = min(warmup_steps, int(total_steps * 0.1))  # 默认warmup为总步数的10%
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # 记录最佳验证损失，用于保存最佳模型
    best_val_loss = float('inf')
    save_config(args)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [] if val_dataset else None,
        'best_epoch': 0
    }
    
    for epoch in range(epochs):
        print(f">>> 训练周期 {epoch+1}/{epochs}")
        sys.stdout.flush()
        model.train()  # 设置为训练模式
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        train_loss_sum = 0.0
        train_samples = 0
        
        # 训练循环
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            try:
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                loss.backward()
                
                # 梯度裁剪防止梯度爆炸
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 累积训练损失
                train_loss_sum += loss.item() * tokens.size(0)
                train_samples += tokens.size(0)
                
                progress.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"\n错误: CUDA相关错误 - {str(e)}")
                    print("尝试减小batch size或使用CPU: --bs 16 --use_cpu")
                    sys.exit(1)
                else:
                    raise e
            
            progress.update()
            if (idx + 1) % args.save_steps == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        
        progress.close()
        
        # 计算平均训练损失
        epoch_train_loss = train_loss_sum / train_samples if train_samples > 0 else float('inf')
        history['train_loss'].append(epoch_train_loss)
        print(f"Epoch {epoch+1}/{epochs} 平均训练损失: {epoch_train_loss:.4f}")
        
        # 验证循环（如果有验证集）
        if val_dataloader is not None:
            model.eval()  # 设置为评估模式
            val_loss_sum = 0.0
            val_samples = 0
            print("在验证集上评估模型...")
            
            with torch.no_grad():  # 不计算梯度
                for val_tokens, val_mask, val_prefix in tqdm(val_dataloader, desc="验证"):
                    try:
                        val_tokens = val_tokens.to(device)
                        val_mask = val_mask.to(device)
                        val_prefix = val_prefix.to(device, dtype=torch.float32)
                        
                        val_outputs = model(val_tokens, val_prefix, val_mask)
                        val_logits = val_outputs.logits[:, val_dataset.prefix_length - 1: -1]
                        val_loss = nnf.cross_entropy(val_logits.reshape(-1, val_logits.shape[-1]), 
                                                  val_tokens.flatten(), ignore_index=0)
                        
                        # 累积验证损失
                        val_loss_sum += val_loss.item() * val_tokens.size(0)
                        val_samples += val_tokens.size(0)
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            print(f"\n错误: 验证中的CUDA错误 - {str(e)}")
                            continue
                        else:
                            raise e
            
            # 计算平均验证损失
            epoch_val_loss = val_loss_sum / val_samples if val_samples > 0 else float('inf')
            history['val_loss'].append(epoch_val_loss)
            print(f"Epoch {epoch+1}/{epochs} 平均验证损失: {epoch_val_loss:.4f}")
            
            # 保存最佳模型
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                history['best_epoch'] = epoch
                print(f"发现新的最佳模型! 验证损失: {best_val_loss:.4f}")
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_best.pt"),
                )
        
        # 定期保存模型
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    
    # 保存训练历史
    with open(os.path.join(output_dir, f"{output_prefix}_history.json"), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"训练完成。最佳验证损失: {best_val_loss:.4f}" if val_dataloader is not None else "训练完成。")
    return model


def main():
    parser = argparse.ArgumentParser()
    # 数据和输出相关参数
    parser.add_argument('--data', default='./CLIP_Pro_train_merged.pkl',
                       help='训练数据文件路径')
    parser.add_argument('--val_data', default='./CLIP_Pro_val_merged.pkl', 
                       help='验证集数据文件路径，例如 ./CLIP_Pro_val_merged.pkl')
    parser.add_argument('--out_dir', default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--prefix', default='clip_pro_prefix', 
                       help='保存文件名前缀')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=8,
                       help='训练轮数')
    parser.add_argument('--save_every', type=int, default=1,
                       help='每多少个epoch保存一次模型')
    parser.add_argument('--save_steps', type=int, default=10000,
                       help='每多少步保存一次最新模型')
    parser.add_argument('--bs', type=int, default=40,
                       help='批次大小')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子，用于结果可复现性')
    
    # 学习率和优化器相关参数
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='预热步数比例')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减系数')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                       help='梯度裁剪阈值，设为0则禁用梯度裁剪')
    
    # 模型相关参数
    parser.add_argument('--prefix_length', type=int, default=10,
                       help='前缀长度')
    parser.add_argument('--prefix_length_clip', type=int, default=10,
                       help='CLIP编码长度')
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true',
                       help='仅训练前缀映射，保持GPT-2冻结')
    parser.add_argument('--mapping_type', type=str, default='transformer', choices=['mlp', 'transformer'],
                       help='使用的映射类型: mlp 或 transformer')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Transformer映射层数')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true',
                       help='是否归一化前缀')
    
    # 数据增强参数
    parser.add_argument('--feature_noise_scale', type=float, default=0.01,
                       help='CLIP特征噪声比例，0表示不添加噪声')
    
    # 硬件相关参数
    parser.add_argument('--use_cpu', dest='use_cpu', action='store_true',
                       help='强制使用CPU进行训练')
    parser.add_argument('--use_mixed_precision', action='store_true',
                       help='使用混合精度训练（需要CUDA）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 自动检测CUDA，但如果用户明确指定了使用CPU，则尊重用户选择
    if args.use_cpu:
        print("根据用户设置，强制使用CPU进行训练")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif torch.cuda.is_available():
        print(f"检测到CUDA设备，将使用GPU进行训练")
    else:
        print("未检测到CUDA设备，将使用CPU进行训练")
    
    # 输出当前设备信息
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and not args.use_cpu:
        print(f"可用的CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    
    # 加载训练数据集
    print(f"加载训练数据: {args.data}")
    dataset = ClipProDataset(args.data, 
                            args.prefix_length, 
                            normalize_prefix=args.normalize_prefix,
                            feature_noise_scale=args.feature_noise_scale)
    
    # 加载验证数据集（如果提供）
    val_dataset = None
    if args.val_data is not None:
        print(f"加载验证数据: {args.val_data}")
        val_dataset = ClipProDataset(args.val_data, 
                                    args.prefix_length, 
                                    normalize_prefix=args.normalize_prefix)
    
    # 使用RN50x4的特征维度 640
    prefix_dim = 640
    
    # 设置映射类型
    mapping_type = MappingType.MLP if args.mapping_type == 'mlp' else MappingType.Transformer
    
    if args.only_prefix:
        model = ClipCaptionPrefix(
            prefix_length=args.prefix_length, 
            clip_length=args.prefix_length_clip, 
            prefix_size=prefix_dim,
            num_layers=args.num_layers, 
            mapping_type=mapping_type
        )
        print(f"训练模式: 仅训练前缀映射，GPT-2保持冻结")
        if mapping_type == MappingType.Transformer:
            print(f"使用 Transformer 映射，层数: {args.num_layers}")
        else:
            print(f"使用 MLP 映射")
    else:
        model = ClipCaptionModel(
            prefix_length=args.prefix_length, 
            clip_length=args.prefix_length_clip, 
            prefix_size=prefix_dim,
            num_layers=args.num_layers, 
            mapping_type=mapping_type
        )
        print(f"训练模式: 训练前缀映射和GPT-2")
        if mapping_type == MappingType.Transformer:
            print(f"使用 Transformer 映射，层数: {args.num_layers}")
        else:
            print(f"使用 MLP 映射")
        sys.stdout.flush()
    
    # 打印模型总结
    print("-" * 50)
    print("模型配置摘要:")
    print(f"- 前缀长度: {args.prefix_length}")
    print(f"- 批次大小: {args.bs}")
    print(f"- 学习率: {args.learning_rate}")
    print(f"- 特征噪声比例: {args.feature_noise_scale}")
    print(f"- 映射类型: {args.mapping_type}")
    print(f"- 使用设备: {'CPU' if args.use_cpu or not torch.cuda.is_available() else 'GPU'}")
    print("-" * 50)
    
    train(dataset, val_dataset, model, args, 
         lr=args.learning_rate, 
         output_dir=args.out_dir, 
         output_prefix=args.prefix)


if __name__ == '__main__':
    main()