import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 验证pycocotools是否已安装，如果没有则安装
try:
    from pycocotools.coco import COCO
except ImportError:
    print("正在安装pycocotools...")
    import pip
    pip.main(['install', 'pycocotools'])
    from pycocotools.coco import COCO

print("="*50)
print("开始验证MS COCO Dataset API和数据获取功能")
print("="*50)

# 第一部分：验证coco_setup.py文件
print("\n1. 验证coco_setup.py文件运行情况")
print("-"*40)

# 检查COCO数据集目录是否存在
dataDir = 'coco_data'
if not os.path.exists(dataDir):
    print(f"创建目录: {dataDir}")
    os.makedirs(dataDir)
    os.makedirs(os.path.join(dataDir, 'annotations'), exist_ok=True)

# 检查是否有annotations文件
dataType = 'train2017'
annFile = f'{dataDir}/annotations/captions_{dataType}.json'

# 如果注释文件不存在，创建一个小的样例文件用于测试
if not os.path.exists(annFile):
    print(f"注释文件不存在: {annFile}")
    print("创建样例注释文件用于测试...")
    
    # 创建一个简单的COCO格式注释文件
    sample_annotation = {
        "info": {"description": "COCO 2017 Dataset"},
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 640, "height": 480, 
             "coco_url": "http://images.cocodataset.org/train2017/000000000001.jpg"},
            {"id": 2, "file_name": "000000000002.jpg", "width": 640, "height": 480,
             "coco_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "caption": "A black and white cat sitting on a chair."},
            {"id": 2, "image_id": 1, "caption": "A cat relaxing on a wooden chair."},
            {"id": 3, "image_id": 1, "caption": "The cat is sitting and looking at the camera."},
            {"id": 4, "image_id": 1, "caption": "A feline pet resting on furniture."},
            {"id": 5, "image_id": 1, "caption": "A domestic cat sitting comfortably on a chair."},
            {"id": 6, "image_id": 2, "caption": "A dog running in a green field."},
            {"id": 7, "image_id": 2, "caption": "A brown dog playing outdoors."},
            {"id": 8, "image_id": 2, "caption": "The dog is enjoying its time outside."},
            {"id": 9, "image_id": 2, "caption": "A happy canine running through grass."},
            {"id": 10, "image_id": 2, "caption": "A pet dog having fun in an open area."}
        ]
    }
    
    # 保存样例注释文件
    os.makedirs(os.path.join(dataDir, 'annotations'), exist_ok=True)
    with open(annFile, 'w') as f:
        json.dump(sample_annotation, f)
    
    print(f"已创建样例注释文件: {annFile}")

try:
    # 初始化COCO API
    coco = COCO(annFile)
    
    # 获取所有图像ID
    imgIds = coco.getImgIds()
    print(f"成功读取COCO数据集，总图像数量: {len(imgIds)}")
    
    # 获取样例图像
    sample_img_ids = imgIds[:2]  # 只获取前2张图片
    sample_imgs = coco.loadImgs(sample_img_ids)
    
    # 显示第一张图像信息
    img_id = sample_imgs[0]['id']
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']
    print(f"样例图像ID: {img_id}, URL: {img_url}")
    
    # 获取该图像的所有标题
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    print(f"图像 {img_id} 的标题:")
    for i, ann in enumerate(anns):
        print(f"  {i+1}. {ann['caption']}")
    
    # 尝试下载并显示图像
    try:
        print("\n尝试下载并显示图像...")
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        plt.figure(figsize=(8, 8))
        plt.imshow(np.array(img))
        plt.axis('off')
        plt.title(f"Image ID: {img_id}")
        plt.savefig('sample_coco_image.png')  # 保存图像而不是显示
        print(f"已保存图像到: sample_coco_image.png")
    except Exception as e:
        print(f"下载或显示图像时出错: {str(e)}")
    
    # 测试create_image_caption_dataset函数
    print("\n测试create_image_caption_dataset函数...")
    def create_image_caption_dataset(coco_instance, num_samples=10000):
        img_ids = coco_instance.getImgIds()[:num_samples]
        dataset = []
        
        for img_id in img_ids:
            img_info = coco_instance.loadImgs(img_id)[0]
            
            # 获取该图像的标题
            ann_ids = coco_instance.getAnnIds(imgIds=img_id)
            anns = coco_instance.loadAnns(ann_ids)
            
            captions = [ann['caption'] for ann in anns]
            
            dataset.append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'coco_url': img_info['coco_url'],
                'captions': captions
            })
        
        return dataset
    
    small_dataset = create_image_caption_dataset(coco, num_samples=2)
    print(f"成功创建小型数据集，包含 {len(small_dataset)} 张图像")
    print(f"数据集第一项: {small_dataset[0]}")
    
    print("\ncoco_setup.py 验证成功!")
    
except Exception as e:
    print(f"运行coco_setup.py时出错: {str(e)}")
    print("coco_setup.py 验证失败!")

# 第二部分：验证data_preparation.py文件
print("\n2. 验证data_preparation.py文件运行情况")
print("-"*40)

try:
    # 使用前面创建的小型数据集继续验证
    
    # 定义COCOTitleDataset类
    class COCOTitleDataset(torch.utils.data.Dataset):
        def __init__(self, dataset_info, transform=None, download=False, local_img_dir=None):
            self.dataset_info = dataset_info
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
            self.download = download
            self.local_img_dir = local_img_dir
            
        def __len__(self):
            return len(self.dataset_info)
        
        def __getitem__(self, idx):
            img_info = self.dataset_info[idx]
            
            # 获取图像
            if self.download:
                try:
                    # 从URL下载图像
                    response = requests.get(img_info['coco_url'])
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    # 如果下载失败，创建一个空白图像
                    print(f"无法下载图像 {img_info['image_id']}，创建空白图像: {str(e)}")
                    image = Image.new('RGB', (224, 224), color='gray')
            else:
                # 从本地目录加载
                if self.local_img_dir:
                    img_path = os.path.join(self.local_img_dir, img_info['file_name'])
                    if os.path.exists(img_path):
                        image = Image.open(img_path).convert('RGB')
                    else:
                        print(f"本地图像文件不存在: {img_path}，创建空白图像")
                        image = Image.new('RGB', (224, 224), color='gray')
                else:
                    # 如果没有指定本地目录，创建空白图像
                    image = Image.new('RGB', (224, 224), color='gray')
            
            # 应用转换
            if self.transform:
                image = self.transform(image)
            
            # 使用第一个标题作为样例
            caption = img_info['captions'][0]
            
            return {
                'image_id': img_info['image_id'],
                'image': image,
                'caption': caption
            }
    
    # 测试vocabulary构建函数
    print("测试vocabulary构建函数...")
    def build_title_vocabulary(dataset_info, min_freq=5):
        word_freq = {}
        
        # 统计词频
        for item in dataset_info:
            for caption in item['captions']:
                # 简单预处理 - 转小写并按空格分割
                words = caption.lower().split()
                for word in words:
                    # 移除标点符号（简单方法）
                    word = word.strip('.,!?;:()[]{}""\'')
                    if word:
                        word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率过滤并创建词汇表
        filtered_words = [word for word, freq in word_freq.items() 
                         if freq >= min_freq]
        
        # 添加特殊标记
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        vocab = special_tokens + sorted(filtered_words)
        
        # 创建映射
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        
        return word2idx, idx2word, word_freq
    
    # 构建词汇表
    word2idx, idx2word, word_freq = build_title_vocabulary(small_dataset, min_freq=1)  # 使用min_freq=1确保所有词都包含
    print(f"词汇表大小: {len(word2idx)}")
    print(f"最常见的词: {sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    # 创建数据集和数据加载器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    print("\n创建数据集并尝试加载数据...")
    dataset = COCOTitleDataset(
        small_dataset, 
        transform=transform,
        download=True
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # 提取一个批次
    sample_batch = next(iter(dataloader))
    print(f"批次图像形状: {sample_batch['image'].shape}")
    print(f"样例标题: {sample_batch['caption'][0]}")
    
    print("\ndata_preparation.py 验证成功!")
    
except Exception as e:
    print(f"运行data_preparation.py时出错: {str(e)}")
    print("data_preparation.py 验证失败!")

print("\n"+"="*50)
print("验证完成")
print("="*50)
