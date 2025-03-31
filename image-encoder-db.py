import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sqlite3
import os
import numpy as np
import time
from tqdm import tqdm  # 用于显示进度条

class ImageEncoder:
    """图像编码器基类"""
    def __init__(self, output_feature_dim):
        self.output_feature_dim = output_feature_dim
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def encode(self, image_tensor):
        """将图像张量编码为特征"""
        raise NotImplementedError("子类必须实现此方法")
    
    def preprocess_image(self, image_path):
        """预处理图像文件为张量"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            # 添加批次维度
            image_tensor = image_tensor.unsqueeze(0)
            return image_tensor
        except Exception as e:
            print(f"处理图像时出错 {image_path}: {e}")
            return None


class ResNetEncoder(ImageEncoder):
    """使用预训练ResNet编码图像"""
    def __init__(self, output_feature_dim=2048, model_name='resnet50', return_feature_map=False):
        super().__init__(output_feature_dim)
        self.return_feature_map = return_feature_map
        self.model_name = model_name
        
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
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode(self, image_tensor):
        """编码图像张量为特征"""
        if image_tensor is None:
            return None
            
        self.model.eval()
        
        with torch.no_grad():
            features = self.model(image_tensor)
            
            if not self.return_feature_map:
                # 调整形状为[批量大小, 特征维度]
                features = features.reshape(features.size(0), -1)
                
        return features


class DatabaseManager:
    """管理数据库连接和操作"""
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """连接到SQLite数据库"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def execute(self, query, params=()):
        """执行SQL查询"""
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor
    
    def fetchall(self, query, params=()):
        """执行查询并获取所有结果"""
        cursor = self.execute(query, params)
        return cursor.fetchall()
    
    def create_features_table(self, model_name, feature_dim):
        """创建存储图像特征的表"""
        table_name = f"image_features_{model_name}"
        
        # 检查表是否已存在
        cursor = self.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone():
            print(f"表 {table_name} 已存在，将被删除并重新创建")
            self.execute(f"DROP TABLE {table_name}")
        
        # 创建表
        self.execute(f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL,
            features BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
        """)
        
        # 创建索引
        self.execute(f"CREATE INDEX idx_{table_name}_image_id ON {table_name}(image_id)")
        
        print(f"已创建表 {table_name}")
        return table_name
    
    def insert_features(self, table_name, image_id, features):
        """将特征向量插入数据库"""
        if features is None:
            return False
            
        # 将PyTorch张量转换为NumPy数组
        features_np = features.cpu().numpy()
        
        # 将NumPy数组序列化为二进制数据
        features_blob = features_np.tobytes()
        
        try:
            self.execute(
                f"INSERT INTO {table_name} (image_id, features) VALUES (?, ?)",
                (image_id, features_blob)
            )
            return True
        except Exception as e:
            print(f"插入特征时出错 (image_id={image_id}): {e}")
            return False
    
    def get_image_paths(self):
        """获取所有图像的ID和路径"""
        return self.fetchall("SELECT id, file_name FROM images")


def encode_all_images(db_path, images_dir, model_name='resnet50', batch_size=16):
    """
    编码所有图像并将特征存储到数据库
    
    Args:
        db_path: 数据库文件路径
        images_dir: 图像目录路径
        model_name: 使用的ResNet模型类型
        batch_size: 批处理大小
    """
    # 初始化数据库管理器
    db_manager = DatabaseManager(db_path)
    
    # 初始化编码器
    encoder = ResNetEncoder(model_name=model_name)
    feature_dim = encoder.output_feature_dim
    
    # 创建特征表
    table_name = db_manager.create_features_table(model_name, feature_dim)
    
    # 获取所有图像路径
    images = db_manager.get_image_paths()
    total_images = len(images)
    print(f"找到 {total_images} 张图像")
    
    # 批处理编码和存储
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 将模型移到设备
    encoder.model = encoder.model.to(device)
    
    start_time = time.time()
    success_count = 0
    
    # 使用tqdm显示进度
    for i in tqdm(range(0, total_images, batch_size), desc=f"编码图像 ({model_name})"):
        batch_images = images[i:i+batch_size]
        batch_tensors = []
        batch_ids = []
        
        # 准备批次
        for image_row in batch_images:
            image_id = image_row['id']
            image_path = os.path.join(images_dir, image_row['file_name'])
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在 {image_path}")
                continue
                
            # 预处理图像
            tensor = encoder.preprocess_image(image_path)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_ids.append(image_id)
        
        if not batch_tensors:
            continue
            
        # 堆叠张量形成批次
        batch = torch.cat(batch_tensors, dim=0).to(device)
        
        # 批量编码
        features = encoder.encode(batch)
        
        # 存储每个图像的特征
        for j, image_id in enumerate(batch_ids):
            image_features = features[j:j+1]  # 保持批次维度
            if db_manager.insert_features(table_name, image_id, image_features):
                success_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"编码完成: {success_count}/{total_images} 图像成功处理")
    print(f"总用时: {elapsed_time:.2f} 秒, 平均每张图像 {elapsed_time/total_images:.4f} 秒")
    
    # 关闭数据库连接
    db_manager.close()


def retrieve_features(db_path, model_name, image_id):
    """
    从数据库检索图像特征
    
    Args:
        db_path: 数据库文件路径
        model_name: 模型名称
        image_id: 图像ID
    
    Returns:
        特征向量 (NumPy数组)
    """
    db_manager = DatabaseManager(db_path)
    table_name = f"image_features_{model_name}"
    
    try:
        results = db_manager.fetchall(
            f"SELECT features FROM {table_name} WHERE image_id = ?",
            (image_id,)
        )
        
        if results:
            # 将二进制数据转换回NumPy数组
            features_blob = results[0]['features']
            
            # 确定ResNet型号以获取正确的特征维度
            encoder = ResNetEncoder(model_name=model_name)
            feature_dim = encoder.output_feature_dim
            
            # 重建NumPy数组
            features = np.frombuffer(features_blob, dtype=np.float32).reshape(1, feature_dim)
            return features
        else:
            print(f"未找到图像ID {image_id} 的特征")
            return None
    finally:
        db_manager.close()


def main():
    # 配置路径
    db_path = "coco_image_title_data/image_title_database.db"
    images_dir = "coco_image_title_data/images"
    
    # 检查路径是否存在
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在 {db_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在 {images_dir}")
        return
    
    # 编码所有图像
    model_name = 'resnet50'  # 也可以选择 'resnet18', 'resnet34', 'resnet101', 'resnet152'
    encode_all_images(db_path, images_dir, model_name)
    
    # 测试检索特定图像的特征
    test_image_id = 9  # 这是之前提到的 000000005802.jpg 的ID
    features = retrieve_features(db_path, model_name, test_image_id)
    
    if features is not None:
        print(f"\n检索图像ID {test_image_id} 的特征:")
        print(f"特征形状: {features.shape}")
        print(f"特征样本 (前5个值): {features[0, :5]}")


if __name__ == "__main__":
    main()
