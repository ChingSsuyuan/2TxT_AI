import torch
import sqlite3
import os
import numpy as np
import time
from PIL import Image
import clip
from tqdm import tqdm

# 固定路径配置
DB_PATH = "coco_image_title_data/image_title_database.db"
IMAGES_DIR = "coco_image_title_data/images"
CLIP_MODEL_TYPE = "RN50x4"
BATCH_SIZE = 16

class CLIPEncoder:
    """CLIP图像编码器"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"正在加载CLIP模型 {CLIP_MODEL_TYPE}...")
        self.model, self.preprocess = clip.load(CLIP_MODEL_TYPE, device=self.device, jit=False)
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"CLIP模型加载完成，使用设备: {self.device}")
    
    def preprocess_image(self, image_path):
        """预处理图像文件为张量"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            return image_tensor
        except:
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
    
    def create_features_table(self):
        """创建存储图像特征的表"""
        table_name = f"image_features_clip"
        
        # 检查表是否已存在
        cursor = self.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone():
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
        except:
            return False
    
    def get_image_paths(self):
        """获取所有图像的ID和路径"""
        return self.fetchall("SELECT id, file_name FROM images")


def encode_all_images():
    """使用CLIP编码所有图像并将特征存储到数据库"""
    # 初始化数据库管理器和编码器
    db_manager = DatabaseManager()
    encoder = CLIPEncoder()
    
    # 创建特征表
    table_name = db_manager.create_features_table()
    
    # 获取所有图像路径
    images = db_manager.get_image_paths()
    total_images = len(images)
    print(f"找到 {total_images} 张图像")
    
    start_time = time.time()
    success_count = 0
    
    # 使用tqdm显示进度
    for i in tqdm(range(0, total_images, BATCH_SIZE), desc="编码图像"):
        batch_images = images[i:i+BATCH_SIZE]
        batch_tensors = []
        batch_ids = []
        
        # 准备批次
        for image_row in batch_images:
            image_id = image_row['id']
            image_path = os.path.join(IMAGES_DIR, image_row['file_name'])
            
            # 预处理图像
            tensor = encoder.preprocess_image(image_path)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_ids.append(image_id)
        
        if not batch_tensors:
            continue
            
        # 堆叠张量形成批次
        batch = torch.cat(batch_tensors, dim=0)
        
        # 批量编码
        features = encoder.encode_image(batch)
        
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


if __name__ == "__main__":
    print(f"开始处理图像编码...")
    print(f"数据库路径: {DB_PATH}")
    print(f"图像目录: {IMAGES_DIR}")
    print(f"CLIP模型类型: {CLIP_MODEL_TYPE}")
    encode_all_images()
    print("处理完成")