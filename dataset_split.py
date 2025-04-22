import sqlite3
import os
import numpy as np
import random
from tqdm import tqdm

# 固定路径配置
DB_PATH = "coco_image_title_data/image_title_database.db"

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
    
    def create_table(self, table_name):
        """创建表"""
        # 检查表是否已存在
        cursor = self.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone():
            self.execute(f"DROP TABLE {table_name}")
            
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


def split_encoded_images(processed_files=None, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05):
    """将编码后的图像随机划分为训练集、验证集和测试集"""
    db_manager = DatabaseManager()
    
    # 步骤1: 创建训练集、验证集和测试集的表
    db_manager.create_table("image_features_clip_train")
    db_manager.create_table("image_features_clip_val")
    db_manager.create_table("image_features_clip_test")
    
    try:
        # 如果没有提供处理过的文件列表，则从主表中获取所有文件名
        if not processed_files:
            results = db_manager.fetchall("SELECT file_name FROM image_features_clip")
            processed_files = [row['file_name'] for row in results]
        
        if not processed_files:
            print("没有找到已处理的图像文件")
            return None
            
        # 随机打乱文件顺序
        random.shuffle(processed_files)
        
        # 计算分割点
        total_files = len(processed_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        # 划分数据集
        train_files = processed_files[:train_end]
        val_files = processed_files[train_end:val_end]
        test_files = processed_files[val_end:]
        
        print(f"将 {total_files} 个图像划分为:")
        print(f"- 训练集: {len(train_files)} 个图像 ({len(train_files)/total_files*100:.1f}%)")
        print(f"- 验证集: {len(val_files)} 个图像 ({len(val_files)/total_files*100:.1f}%)")
        print(f"- 测试集: {len(test_files)} 个图像 ({len(test_files)/total_files*100:.1f}%)")
        
        # 处理训练集
        process_split(db_manager, "image_features_clip", "image_features_clip_train", train_files, "训练集")
        
        # 处理验证集
        process_split(db_manager, "image_features_clip", "image_features_clip_val", val_files, "验证集")
        
        # 处理测试集
        process_split(db_manager, "image_features_clip", "image_features_clip_test", test_files, "测试集")
        
        # 返回划分结果统计
        return {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files),
            'total': total_files,
            'train_ratio': len(train_files)/total_files,
            'val_ratio': len(val_files)/total_files,
            'test_ratio': len(test_files)/total_files
        }
        
    except Exception as e:
        print(f"划分数据集时出错: {str(e)}")
        return None
    finally:
        db_manager.close()


def process_split(db_manager, source_table, target_table, file_names, split_name):
    """处理数据集划分，将特征从源表复制到目标表"""
    print(f"正在处理{split_name}...")
    
    # 批处理大小
    batch_size = 100
    success_count = 0
    
    for i in tqdm(range(0, len(file_names), batch_size), desc=f"处理{split_name}"):
        batch_files = file_names[i:i+batch_size]
        
        for file_name in batch_files:
            # 从源表中获取特征
            results = db_manager.fetchall(
                f"SELECT features FROM {source_table} WHERE file_name = ?",
                (file_name,)
            )
            
            if results:
                # 将特征插入目标表
                db_manager.execute(
                    f"INSERT INTO {target_table} (file_name, features) VALUES (?, ?)",
                    (file_name, results[0]['features'])
                )
                success_count += 1
    
    print(f"{split_name}处理完成: {success_count}/{len(file_names)} 个文件成功处理")


def get_dataset_split_counts():
    """获取数据集划分的统计信息"""
    db_manager = DatabaseManager()
    
    try:
        # 检查表是否存在
        tables = ["image_features_clip_train", "image_features_clip_val", "image_features_clip_test"]
        for table in tables:
            cursor = db_manager.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if not cursor.fetchone():
                print(f"错误: 表 {table} 不存在")
                return None
        
        # 获取各个表的记录数
        train_count = db_manager.fetchall("SELECT COUNT(*) as count FROM image_features_clip_train")[0]['count']
        val_count = db_manager.fetchall("SELECT COUNT(*) as count FROM image_features_clip_val")[0]['count']
        test_count = db_manager.fetchall("SELECT COUNT(*) as count FROM image_features_clip_test")[0]['count']
        
        total_count = train_count + val_count + test_count
        
        # 计算比例
        train_ratio = train_count / total_count if total_count > 0 else 0
        val_ratio = val_count / total_count if total_count > 0 else 0
        test_ratio = test_count / total_count if total_count > 0 else 0
        
        return {
            'training': train_count,
            'validation': val_count,
            'testing': test_count,
            'total': total_count,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio
        }
    except Exception as e:
        print(f"获取数据集划分统计信息时出错: {str(e)}")
        return None
    finally:
        db_manager.close()


def validate_feature_structures():
    """验证训练集、验证集和测试集的特征结构是否一致"""
    db_manager = DatabaseManager()
    
    try:
        # 从各个集合中各获取一个样本
        train_sample = db_manager.fetchall("SELECT features FROM image_features_clip_train LIMIT 1")
        val_sample = db_manager.fetchall("SELECT features FROM image_features_clip_val LIMIT 1")
        test_sample = db_manager.fetchall("SELECT features FROM image_features_clip_test LIMIT 1")
        
        if not train_sample or not val_sample or not test_sample:
            print("错误: 无法获取样本特征")
            return False
        
        # 转换为NumPy数组
        feature_dim = 640  # RN50x4的特征维度
        
        train_features = np.frombuffer(train_sample[0]['features'], dtype=np.float32).reshape(1, feature_dim)
        val_features = np.frombuffer(val_sample[0]['features'], dtype=np.float32).reshape(1, feature_dim)
        test_features = np.frombuffer(test_sample[0]['features'], dtype=np.float32).reshape(1, feature_dim)
        
        # 检查维度是否一致
        if (train_features.shape == val_features.shape == test_features.shape):
            return True
        
        return False
    except Exception as e:
        print(f"验证特征结构时出错: {str(e)}")
        return False
    finally:
        db_manager.close()


if __name__ == "__main__":
    print("开始划分数据集...")
    
    # 步骤1: 划分数据集
    split_stats = split_encoded_images(train_ratio=0.85, val_ratio=0.10, test_ratio=0.05)
    
    if split_stats:
        print("\n数据集划分结果:")
        print(f"训练集: {split_stats['train']} 图像 ({split_stats['train_ratio']*100:.1f}%)")
        print(f"验证集: {split_stats['val']} 图像 ({split_stats['val_ratio']*100:.1f}%)")
        print(f"测试集: {split_stats['test']} 图像 ({split_stats['test_ratio']*100:.1f}%)")
    
    # 步骤2: 验证特征结构
    if validate_feature_structures():
        print("✓ 训练集、验证集和测试集的特征结构一致")
    else:
        print("✗ 特征结构不一致，请检查数据")
    
    print("处理完成")