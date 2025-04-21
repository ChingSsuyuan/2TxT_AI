import os
import json
import argparse
import shutil
import subprocess
import sys
import heapq
from pathlib import Path

def run_training(train_args):
    """运行原始训练脚本"""
    print("开始模型训练...")
    
    # 构建命令行参数
    cmd = [sys.executable, 'modified_train_script.py']
    
    for key, value in train_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))
    
    # 运行训练脚本
    subprocess.run(cmd, check=True)
    print("训练完成！")

def filter_best_models(out_dir, prefix, top_n=2, use_val=True):
    """保留损失值最低的top_n个模型，删除其余模型"""
    print(f"过滤保留损失值最低的{top_n}个模型...")
    
    # 读取训练过程中记录的损失值
    loss_file = os.path.join(out_dir, "loss_per_epoch.json")
    
    if not os.path.exists(loss_file):
        print(f"错误: 损失记录文件 {loss_file} 不存在")
        return
    
    with open(loss_file, 'r') as f:
        loss_data = json.load(f)
    
    # 根据验证集或训练集损失值选择最佳模型
    loss_key = 'val' if use_val and 'val' in loss_data and loss_data['val'] else 'train'
    print(f"使用 {loss_key} 损失值评估模型...")
    
    losses = loss_data[loss_key]
    
    # 找出损失值最低的top_n个epoch
    # 使用负值进行最大堆模拟最小堆
    best_epochs = heapq.nsmallest(top_n, range(len(losses)), key=lambda i: losses[i])
    
    print(f"最佳epoch: {best_epochs}, 对应损失值: {[losses[i] for i in best_epochs]}")
    
    # 获取所有checkpoint文件
    model_files = list(Path(out_dir).glob(f"{prefix}-*.pt"))
    latest_file = list(Path(out_dir).glob(f"{prefix}_latest.pt"))
    
    # 保留最佳模型，删除其余模型
    kept_models = []
    for i, epoch in enumerate(best_epochs):
        model_path = os.path.join(out_dir, f"{prefix}-{epoch:03d}.pt")
        
        if os.path.exists(model_path):
            # 创建新的文件名
            new_name = os.path.join(out_dir, f"{prefix}_best_{i+1}.pt")
            shutil.copy(model_path, new_name)
            kept_models.append((epoch, new_name))
            print(f"保留第{epoch}个epoch的模型，重命名为: {new_name}")
        else:
            print(f"警告: 未找到第{epoch}个epoch的模型文件")
    
    # 删除不需要的模型文件
    for file_path in model_files:
        epoch_str = str(file_path).split('-')[-1].split('.')[0]
        try:
            epoch = int(epoch_str)
            if epoch not in best_epochs:
                os.remove(file_path)
                print(f"删除模型: {file_path}")
        except ValueError:
            print(f"无法解析epoch: {file_path}")
    
    # 保留latest模型作为参考
    print(f"最佳模型已保存: {kept_models}")
    
    return kept_models

def main():
    parser = argparse.ArgumentParser(description='运行训练并只保留最佳模型')
    
    # 训练参数
    parser.add_argument('--db_path', default='coco_image_title_data/image_title_database.db',
                        help='包含CLIP嵌入的SQLite数据库路径')
    parser.add_argument('--table_name', default='image_features_clip',
                        help='包含CLIP嵌入的数据库表名')
    parser.add_argument('--val_db_path', default='',
                        help='验证数据库路径（可选）')
    parser.add_argument('--val_table_name', default='',
                        help='验证表名（可选）')
    parser.add_argument('--out_dir', default='./checkpoints',
                        help='输出目录路径')
    parser.add_argument('--prefix', default='coco_prefix',
                        help='保存文件名的前缀')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练epoch数')
    parser.add_argument('--bs', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--only_prefix', action='store_true',
                        help='仅训练CLIP和GPT之间的映射器，同时冻结GPT')
    parser.add_argument('--mapping_type', type=str, default='transformer',
                        help='CLIP和GPT之间的架构类型（mlp/transformer）')
    parser.add_argument('--noise_variance', type=float, default=0.0,
                        help='噪声方差')
    
    # 模型筛选参数
    parser.add_argument('--top_n', type=int, default=2,
                        help='保留的最佳模型数量')
    parser.add_argument('--use_val', action='store_true',
                        help='使用验证集损失值评估模型（如果有）')
    
    args = parser.parse_args()
    
    # 提取训练参数
    train_args = {k: v for k, v in vars(args).items() if k not in ['top_n', 'use_val']}
    
    # 运行训练
    run_training(train_args)
    
    # 保留最佳模型
    filter_best_models(args.out_dir, args.prefix, args.top_n, args.use_val)

if __name__ == "__main__":
    main()
