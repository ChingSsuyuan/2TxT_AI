import torch
import sqlite3
import os
import numpy as np
import time
from PIL import Image
import clip
from tqdm import tqdm
import random

# 固定路径配置
DB_PATH = "coco_image_title_data/image_title_database.db"
IMAGES_DIR = "coco_image_title_data/images"
CLIP_MODEL_TYPE = "RN50x4"
BATCH_SIZE = 16

class CLIPEncoder:
    """CLIP"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"CLIP Model {CLIP_MODEL_TYPE}...")
        self.model, self.preprocess = clip.load(CLIP_MODEL_TYPE, device=self.device, jit=False)
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"CLIP complete: {self.device}")
    
    def preprocess_image(self, image_path):
        """预处理图像文件为张量"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            print(f"处理图像时出错 {image_path}: {str(e)}")
            return None
    
    def encode_image(self, image_tensor):
        """编码图像张量为特征"""
        if image_tensor is None:
            return None
            
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
        return features


class DatabaseManager:
    """管理数据库连接和操作"""
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        """连接到SQLite数据库"""
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def execute(self, query, params=()):
        """执行SQL查询"""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor
    
    def fetchall(self, query, params=()):
        """执行查询并获取所有结果"""
        cursor = self.execute(query, params)
        return cursor.fetchall()
    
    def create_features_table(self, table_name="image_features_clip"):
        """创建存储图像特征的表"""
        # 检查表是否已存在
        cursor = self.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone():
            self.execute(f"DROP TABLE {table_name}")
        
        # 创建表
        self.execute(f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY,
            file_name TEXT NOT NULL,
            features BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 创建索引
        self.execute(f"CREATE INDEX idx_{table_name}_file_name ON {table_name}(file_name)")
        
        print(f"已创建表 {table_name}")
        return table_name
    
    def insert_features(self, table_name, file_name, features):
        """将特征向量插入数据库"""
        if features is None:
            return False
            
        # 将PyTorch张量转换为NumPy数组
        features_np = features.cpu().numpy()
        
        # 将NumPy数组序列化为二进制数据
        features_blob = features_np.tobytes()
        
        try:
            self.execute(
                f"INSERT INTO {table_name} (file_name, features) VALUES (?, ?)",
                (file_name, features_blob)
            )
            return True
        except Exception as e:
            print(f"插入特征时出错 (file_name={file_name}): {str(e)}")
            return False


def scan_images_directory():
    """扫描图像目录中的所有图像文件"""
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    
    # 检查目录是否存在
    if not os.path.exists(IMAGES_DIR):
        print(f"错误: 图像目录不存在 {IMAGES_DIR}")
        return []
    
    # 遍历目录中的所有文件
    for file in os.listdir(IMAGES_DIR):
        file_path = os.path.join(IMAGES_DIR, file)
        if os.path.isfile(file_path):
            # 检查文件扩展名
            _, ext = os.path.splitext(file.lower())
            if ext in supported_extensions:
                image_files.append(file)
    
    print(f"在目录中找到 {len(image_files)} 个图像文件")
    return image_files


def encode_all_images():
    """扫描目录并使用CLIP编码所有图像"""
    # 初始化数据库管理器和编码器
    db_manager = DatabaseManager()
    encoder = CLIPEncoder()
    
    # 创建特征表
    main_table = db_manager.create_features_table("image_features_clip")
    
    # 扫描图像目录
    image_files = scan_images_directory()
    if not image_files:
        print("没有找到图像文件，退出")
        return []
    
    # 显示前几个图像文件的路径
    sample_count = min(5, len(image_files))
    for i in range(sample_count):
        image_path = os.path.join(IMAGES_DIR, image_files[i])
        print(f"样本图像 {i+1}: {image_path} - {'存在' if os.path.exists(image_path) else '不存在'}")
    
    start_time = time.time()
    success_count = 0
    error_count = 0
    processed_files = []
    
    # 使用tqdm显示进度
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="编码图像"):
        batch_files = image_files[i:i+BATCH_SIZE]
        batch_tensors = []
        batch_file_names = []
        
        # 准备批次
        for file_name in batch_files:
            image_path = os.path.join(IMAGES_DIR, file_name)
            
            # 预处理图像
            tensor = encoder.preprocess_image(image_path)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_file_names.append(file_name)
        
        if not batch_tensors:
            continue
            
        # 堆叠张量形成批次
        batch = torch.cat(batch_tensors, dim=0)
        
        # 批量编码
        features = encoder.encode_image(batch)
        
        # 存储每个图像的特征
        for j, file_name in enumerate(batch_file_names):
            image_features = features[j:j+1]  # 保持批次维度
            if db_manager.insert_features(main_table, file_name, image_features):
                success_count += 1
                processed_files.append(file_name)
            else:
                error_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"编码完成: {success_count}/{len(image_files)} 图像成功处理")
    print(f"编码失败: {error_count} 图像")
    print(f"总用时: {elapsed_time:.2f} 秒, 平均每张图像 {elapsed_time/len(image_files):.4f} 秒")
    
    # 关闭数据库连接
    db_manager.close()
    
    return processed_files


def retrieve_features(file_name, table_name="image_features_clip"):
    """从数据库检索图像特征"""
    db_manager = DatabaseManager()
    
    try:
        results = db_manager.fetchall(
            f"SELECT features FROM {table_name} WHERE file_name = ?",
            (file_name,)
        )
        
        if results:
            # 将二进制数据转换回NumPy数组
            features_blob = results[0]['features']
            
            # 重建NumPy数组 (根据RN50x4的特征维度)
            feature_dim = 640  # RN50x4的特征维度
            features = np.frombuffer(features_blob, dtype=np.float32).reshape(1, feature_dim)
            return features
        else:
            print(f"未找到图像 {file_name} 的特征")
            return None
    finally:
        db_manager.close()


def split_dataset(processed_files):
    """将数据集分为训练集(85%)、验证集(10%)和测试集(5%)"""
    from dataset_split import split_encoded_images
    return split_encoded_images(processed_files, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05)


if __name__ == "__main__":
    print(f"开始处理图像编码...")
    print(f"数据库路径: {DB_PATH}")
    print(f"图像目录: {IMAGES_DIR}")
    print(f"CLIP模型类型: {CLIP_MODEL_TYPE}")
    
    # 检查图像目录是否存在
    if not os.path.exists(IMAGES_DIR):
        print(f"错误: 图像目录不存在 {IMAGES_DIR}")
    else:
        # 步骤1: 编码所有图像
        processed_files = encode_all_images()
        
        # 步骤2: 划分数据集为训练集(85%)、验证集(10%)和测试集(5%)
        if processed_files:
            from dataset_split import split_encoded_images, get_dataset_split_counts, validate_feature_structures
            
            # 执行数据集划分
            split_stats = split_encoded_images(processed_files, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05)
            
            if split_stats:
                print("\n数据集划分结果:")
                print(f"训练集: {split_stats['train']} 图像 ({split_stats['train_ratio']*100:.1f}%)")
                print(f"验证集: {split_stats['val']} 图像 ({split_stats['val_ratio']*100:.1f}%)")
                print(f"测试集: {split_stats['test']} 图像 ({split_stats['test_ratio']*100:.1f}%)")
            
            # 验证特征结构
            if validate_feature_structures():
                print("✓ 训练集、验证集和测试集的特征结构一致")
            else:
                print("✗ 特征结构不一致，请检查数据")
    
    # 测试特征检索
    image_files = scan_images_directory()
    if image_files:
        sample_file = image_files[0]
        print(f"\n测试特征检索，使用文件: {sample_file}")
        
        features = retrieve_features(sample_file)
        if features is not None:
            print(f"特征形状: {features.shape}")
            print(f"特征样本 (前5个值): {features[0, :5]}")
    
    print("处理完成")