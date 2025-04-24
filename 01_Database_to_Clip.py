import os
import json
import torch
import clip
from PIL import Image
import pickle
import requests
import argparse
from tqdm import tqdm
import random


def download_coco_images(annotation_file, output_dir, dataset_type, max_images=None):
    """从COCO数据集下载图像到指定目录"""
    print(f"加载标注文件: {annotation_file}")
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # 获取所有图像ID
    images_info = data['images']
    print(f"标注文件中的图像总数: {len(images_info)}")
    
    # 如果指定了最大图像数，随机抽样
    if max_images and max_images < len(images_info):
        images_info = random.sample(images_info, max_images)
        print(f"随机选择了 {max_images} 张图像进行下载")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载图像
    success_count = 0
    failed_count = 0
    
    for img_info in tqdm(images_info, desc=f"下载{dataset_type}图像"):
        img_id = img_info['id']
        file_name = f"{int(img_id):012d}.jpg"
        output_path = os.path.join(output_dir, file_name)
        
        # 如果文件已存在且大小正常，跳过下载
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"图像已存在: {output_path}")
            success_count += 1
            continue
        
        # 尝试多个下载URL
        urls = [
            f"http://images.cocodataset.org/{dataset_type}2017/{file_name}",
            f"http://images.cocodataset.org/train2017/{file_name}",
            f"http://images.cocodataset.org/val2017/{file_name}"
        ]
        
        download_success = False
        for url in urls:
            try:
                print(f"尝试从 {url} 下载图像")
                response = requests.get(url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                    
                    # 验证文件
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"成功下载图像到: {output_path}")
                        download_success = True
                        break
                    else:
                        print(f"下载的文件无效: {output_path}")
            except Exception as e:
                print(f"从 {url} 下载失败: {str(e)}")
        
        if download_success:
            success_count += 1
        else:
            failed_count += 1
            print(f"所有URL尝试失败，无法下载图像 ID: {img_id}")
    
    print(f"下载完成。成功: {success_count}, 失败: {failed_count}")
    return success_count


def encode_images(image_dir, clip_model, preprocess, batch_size, output_prefix, dataset_type):
    """使用CLIP模型对图像进行编码"""
    print(f"开始编码 {dataset_type} 图像...")
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 如果没有图像可编码
    if not image_files:
        print("没有找到需要编码的图像文件")
        return 0
    
    # 获取每个图像的ID
    image_ids = [int(f.split('.')[0]) for f in image_files]
    
    # 按批次处理
    all_embeddings = []
    all_captions = []
    
    # 加载标注信息
    annotation_file = f'./coco_data/annotations/captions_{dataset_type}2017.json'
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # 创建图像ID到标题的映射
    id_to_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in id_to_captions:
            id_to_captions[img_id] = []
        id_to_captions[img_id].append(ann['caption'])
    
    # 按批次处理图像
    total_processed = 0
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_ids = image_ids[i:i+batch_size]
        
        batch_embeddings = []
        batch_captions = []
        
        for j, (file_name, img_id) in enumerate(zip(batch_files, batch_ids)):
            file_path = os.path.join(image_dir, file_name)
            
            try:
                # 读取和预处理图像
                image = Image.open(file_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                # 使用CLIP模型编码图像
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input).cpu()
                
                # 获取图像的标题
                captions = id_to_captions.get(img_id, ["No caption available"])
                
                # 为每个标题创建一个项目
                for caption in captions:
                    caption_item = {
                        "image_id": img_id,
                        "id": total_processed + j,
                        "caption": caption,
                        "clip_embedding": total_processed + j,
                        "filename": file_name
                    }
                    
                    batch_embeddings.append(image_features)
                    batch_captions.append(caption_item)
                
                print(f"成功编码图像: {file_name}")
                
            except Exception as e:
                print(f"处理图像 {file_path} 时出错: {str(e)}")
        
        # 保存批次结果
        if batch_embeddings:
            batch_tensor = torch.cat(batch_embeddings, dim=0)
            batch_file = f"{output_prefix}_{dataset_type}_batch_{i//batch_size + 1}.pkl"
            
            with open(batch_file, 'wb') as f:
                pickle.dump({
                    "clip_embedding": batch_tensor,
                    "captions": batch_captions
                }, f)
            
            print(f"保存批次 {i//batch_size + 1} 到 {batch_file}, 包含 {len(batch_captions)} 个编码")
            total_processed += len(batch_captions)
    
    print(f"编码完成，共处理了 {total_processed} 个标题")
    return total_processed


def merge_pkl_files(output_prefix, dataset_type):
    """合并所有批次的PKL文件"""
    print(f"合并 {dataset_type} 的批次文件...")
    
    # 查找所有批次文件
    import glob
    batch_files = glob.glob(f"{output_prefix}_{dataset_type}_batch_*.pkl")
    batch_files.sort(key=lambda x: int(x.split('_batch_')[1].split('.')[0]))
    
    if not batch_files:
        print(f"未找到 {dataset_type} 的批次文件")
        return False
    
    print(f"找到 {len(batch_files)} 个批次文件")
    
    all_embeddings = []
    all_captions = []
    
    # 加载和合并所有批次
    for batch_file in batch_files:
        try:
            print(f"加载 {batch_file}...")
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            if 'clip_embedding' in batch_data and 'captions' in batch_data:
                all_embeddings.append(batch_data['clip_embedding'])
                all_captions.extend(batch_data['captions'])
            else:
                print(f"警告: {batch_file} 的数据结构无效")
        except Exception as e:
            print(f"加载 {batch_file} 时出错: {str(e)}")
    
    # 如果有有效数据，合并并保存
    if all_embeddings and all_captions:
        try:
            combined_embedding = torch.cat(all_embeddings, dim=0)
            merged_file = f"{output_prefix}_{dataset_type}_merged.pkl"
            
            print(f"合并 {len(all_embeddings)} 个张量，形状: {combined_embedding.shape}")
            print(f"合并 {len(all_captions)} 个标题")
            
            with open(merged_file, 'wb') as f:
                pickle.dump({
                    "clip_embedding": combined_embedding,
                    "captions": all_captions
                }, f)
            
            print(f"成功将所有批次合并到: {merged_file}")
            return True
        
        except Exception as e:
            print(f"合并文件时出错: {str(e)}")
            return False
    else:
        print("没有找到有效的批次数据")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载COCO图像并用CLIP编码')
    parser.add_argument('--dataset', type=str, choices=['train', 'val'], default='train',
                       help='要处理的数据集类型 (默认: train)')
    parser.add_argument('--batch-size', type=int, default=2000,
                       help='编码的批次大小 (默认: 200)')
    parser.add_argument('--output-prefix', type=str, default='./CLIP_Pro',
                       help='输出文件的前缀 (默认: ./CLIP_Pro)')
    parser.add_argument('--download-only', action='store_true',
                       help='只下载图像，不进行编码')
    parser.add_argument('--encode-only', action='store_true',
                       help='只编码已有图像，不下载')
    parser.add_argument('--merge-only', action='store_true',
                       help='只合并已有的批次文件')
    parser.add_argument('--max-images', type=int, default=200,
                       help='最多下载的图像数量 (默认: 全部)')
    args = parser.parse_args()
    
    # 设置路径
    dataset_type = args.dataset
    image_dir = f"./coco_data/images/{dataset_type}2017"
    annotation_file = f'./coco_data/annotations/captions_{dataset_type}2017.json'
    
    print(f"处理 {dataset_type} 数据集")
    print(f"图像目录: {image_dir}")
    print(f"标注文件: {annotation_file}")
    print(f"批次大小: {args.batch_size}")
    print(f"输出前缀: {args.output_prefix}")
    
    # 下载图像
    if not args.encode_only and not args.merge_only:
        print("\n===== 步骤 1: 下载图像 =====")
        download_coco_images(annotation_file, image_dir, dataset_type, args.max_images)
    
    # 如果只下载不编码，则退出
    if args.download_only:
        print("只下载模式，完成任务")
        exit(0)
    
    # 编码图像
    if not args.merge_only:
        print("\n===== 步骤 2: 编码图像 =====")
        # 加载CLIP模型
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        clip_model_type = "RN50x4"
        print(f"加载CLIP模型: {clip_model_type}")
        clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        
        encode_images(image_dir, clip_model, preprocess, args.batch_size, 
                     args.output_prefix, dataset_type)
    
    # 合并批次文件
    print("\n===== 步骤 3: 合并批次文件 =====")
    merge_success = merge_pkl_files(args.output_prefix, dataset_type)
    
    if merge_success:
        print(f"\n所有处理完成! 最终文件: {args.output_prefix}_{dataset_type}_merged.pkl")
    else:
        print("\n合并失败,请检查之前的错误信息")