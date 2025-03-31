# 图像编码模块实现
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sqlite3
import os
from transformers import ViTModel, ViTFeatureExtractor

class ImageEncoder:
    """图像编码器基类，定义接口"""
    def __init__(self, output_feature_dim):
        self.output_feature_dim = output_feature_dim
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet标准均值
                std=[0.229, 0.224, 0.225]    # ImageNet标准方差
            )
        ])
    
    def encode(self, image_tensor):
        """将图像张量编码为特征"""
        raise NotImplementedError("子类必须实现此方法")
    
    def preprocess_image(self, image_path):
        """预处理图像文件为张量"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        # 添加批次维度
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor


class ResNetEncoder(ImageEncoder):
    """使用预训练ResNet编码图像"""
    def __init__(self, output_feature_dim=2048, model_name='resnet50', return_feature_map=False):
        super().__init__(output_feature_dim)
        self.return_feature_map = return_feature_map
        
        # 加载预训练模型
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            self.output_feature_dim = 512
        elif model_name == 'resnet34':
            base_model = models.resnet34(pretrained=True)
            self.output_feature_dim = 512
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            self.output_feature_dim = 2048
        elif model_name == 'resnet101':
            base_model = models.resnet101(pretrained=True)
            self.output_feature_dim = 2048
        elif model_name == 'resnet152':
            base_model = models.resnet152(pretrained=True)
            self.output_feature_dim = 2048
        else:
            raise ValueError(f"不支持的ResNet类型: {model_name}")
        
        # 根据需要返回特征图或全局特征
        if self.return_feature_map:
            # 移除自适应池化和全连接层，保留特征图
            self.model = nn.Sequential(*list(base_model.children())[:-2])
        else:
            # 移除最后的全连接层，保留全局特征
            self.model = nn.Sequential(*list(base_model.children())[:-1])
        
        # 冻结参数（可选）
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode(self, image_tensor):
        """
        编码图像张量为特征
        
        Args:
            image_tensor: 形状为[批量大小, 3, H, W]的图像张量
            
        Returns:
            如果return_feature_map=True:
                特征图, 形状为[批量大小, 特征维度, h, w]
            否则:
                全局特征, 形状为[批量大小, 特征维度]
        """
        self.model.eval()  # 设置为评估模式
        
        with torch.no_grad():
            features = self.model(image_tensor)
            
            if not self.return_feature_map:
                # 调整形状为[批量大小, 特征维度]
                features = features.reshape(features.size(0), -1)
                
        return features


class ViTEncoder(ImageEncoder):
    """使用预训练Vision Transformer编码图像"""
    def __init__(self, model_name='google/vit-base-patch16-224', output_feature_dim=768):
        super().__init__(output_feature_dim)
        
        # 加载预训练ViT模型和特征提取器
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
        # 自定义ViT的预处理
        self.transform = None  # 使用ViT自己的预处理
        
        # 冻结参数（可选）
        for param in self.model.parameters():
            param.requires_grad = False
    
    def preprocess_image(self, image_path):
        """使用ViT的特征提取器预处理图像"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs
    
    def encode(self, image_input):
        """
        编码图像为特征
        
        Args:
            image_input: 由feature_extractor处理的输入字典
            
        Returns:
            图像特征, 形状为[批量大小, 特征维度]
        """
        self.model.eval()  # 设置为评估模式
        
        with torch.no_grad():
            outputs = self.model(**image_input)
            # 使用[CLS]令牌表示整个图像
            features = outputs.last_hidden_state[:, 0, :]
                
        return features


class DBImageDataLoader:
    """从数据库加载图像数据"""
    def __init__(self, db_path, image_dir):
        self.db_path = db_path
        self.image_dir = image_dir
        self.conn = None
    
    def connect(self):
        """连接到SQLite数据库"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
        return self.conn
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
    
    def get_image_path(self, image_id):
        """获取指定ID的图像路径"""
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_name FROM images WHERE id = ?", (image_id,))
        result = cursor.fetchone()
        
        if result:
            return os.path.join(self.image_dir, result['file_name'])
        return None
    
    def get_all_image_paths(self):
        """获取所有图像路径"""
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, file_name FROM images")
        results = cursor.fetchall()
        
        return [(row['id'], os.path.join(self.image_dir, row['file_name'])) for row in results]
    
    def get_image_captions(self, image_id):
        """获取指定图像的所有描述"""
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT caption FROM captions WHERE image_id = ?", (image_id,))
        results = cursor.fetchall()
        
        return [row['caption'] for row in results]


# 使用示例
def encode_image_example():
    # 配置路径
    db_path = "coco_image_title_data/image_title_database.db"
    image_dir = "coco_image_title_data/images"
    
    # 初始化数据加载器
    data_loader = DBImageDataLoader(db_path, image_dir)
    
    # 初始化编码器
    resnet_encoder = ResNetEncoder(model_name='resnet50', return_feature_map=False)
    vit_encoder = ViTEncoder()
    
    # 获取特定图像路径 (示例中的000000005802.jpg)
    image_id = 9  # 根据之前的分析，这是000000005802.jpg的ID
    image_path = data_loader.get_image_path(image_id)
    
    if image_path:
        print(f"处理图像: {image_path}")
        
        # ResNet编码
        image_tensor = resnet_encoder.preprocess_image(image_path)
        resnet_features = resnet_encoder.encode(image_tensor)
        print(f"ResNet特征形状: {resnet_features.shape}")  # 应为[1, 2048]
        
        # ViT编码
        vit_inputs = vit_encoder.preprocess_image(image_path)
        vit_features = vit_encoder.encode(vit_inputs)
        print(f"ViT特征形状: {vit_features.shape}")  # 应为[1, 768]
        
        # 获取图像描述
        captions = data_loader.get_image_captions(image_id)
        print(f"图像描述:")
        for caption in captions:
            print(f"- {caption}")
    else:
        print(f"图像ID {image_id} 未找到")
    
    # 关闭数据库连接
    data_loader.close()


# 如何在实际应用中获取特征图
def get_feature_maps_example():
    # 初始化返回特征图的编码器
    feature_map_encoder = ResNetEncoder(model_name='resnet50', return_feature_map=True)
    
    # 处理示例图像
    image_path = "coco_image_title_data/images/000000005802.jpg"
    image_tensor = feature_map_encoder.preprocess_image(image_path)
    feature_maps = feature_map_encoder.encode(image_tensor)
    
    print(f"特征图形状: {feature_maps.shape}")  # 应为[1, 2048, 7, 7]
    
    # 特征图可用于:
    # 1. 空间注意力
    # 2. 目标检测
    # 3. 图像分割
    # 4. 精细粒度的视觉-语言对齐


# 批量处理多张图像
def batch_process_example():
    # 配置
    db_path = "coco_image_title_data/image_title_database.db"
    image_dir = "coco_image_title_data/images"
    
    # 初始化
    data_loader = DBImageDataLoader(db_path, image_dir)
    encoder = ResNetEncoder()
    
    # 获取所有图像路径
    image_data = data_loader.get_all_image_paths()
    
    # 批量处理前5张图像
    batch_size = 5
    batch_tensors = []
    
    for i, (image_id, image_path) in enumerate(image_data[:batch_size]):
        # 预处理图像
        tensor = encoder.preprocess_image(image_path)
        batch_tensors.append(tensor)
    
    # 堆叠张量形成批次
    batch = torch.cat(batch_tensors, dim=0)
    print(f"批次形状: {batch.shape}")  # 应为[5, 3, 224, 224]
    
    # 批量编码
    features = encoder.encode(batch)
    print(f"批量特征形状: {features.shape}")  # 应为[5, 2048]
    
    # 关闭数据库连接
    data_loader.close()


if __name__ == "__main__":
    # 运行示例
    encode_image_example()
    print("\n" + "-"*50 + "\n")
    get_feature_maps_example()
    print("\n" + "-"*50 + "\n")
    batch_process_example()
