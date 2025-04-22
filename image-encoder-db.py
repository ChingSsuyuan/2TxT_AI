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
    """CLIP 编码器"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"加载 CLIP Model {CLIP_MODEL_TYPE}...")
        self.model, self.preprocess = clip.load(CLIP_MODEL_TYPE, device=self.device, jit=False)
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"CLIP 初始化完成: {self.device}")
    
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
    
    def prepare_features_table(self, table_name="image_features_clip"):
        """创建或验证存储图像特征的表"""
        # 检查表是否已存在
        cursor = self.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        
        if not cursor.fetchone():
            # 表不存在，创建新表
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
            print(f"已创建新表 {table_name}")
        else:
            print(f"表 {table_name} 已存在，将在后面添加新数据")
        
        return table_name
    
    def get_encoded_files(self, table_name="image_features_clip"):
        """获取已编码的文件名列表"""
        try:
            results = self.fetchall(f"SELECT file_name FROM {table_name}")
            return [row['file_name'] for row in results]
        except Exception as e:
            print(f"获取已编码文件名时出错: {str(e)}")
            return []
    
    def insert_features(self, table_name, file_name, features):
        """将特征向量插入数据库"""
        if features is None:
            return False
            
        # 将PyTorch张量转换为NumPy数组
        features_np = features.cpu().numpy()
        
        # 将NumPy数组序列化为二进制数据
        features_blob = features_np.tobytes()
        
        try:
            # 首先检查是否已存在
            cursor = self.execute(
                f"SELECT id FROM {table_name} WHERE file_name = ?",
                (file_name,)
            )
            exists = cursor.fetchone()
            
            if exists:
                print(f"文件 {file_name} 已在 {table_name} 中编码，跳过")
                return False
            
            # 不存在则插入
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
    
    # 准备特征表（不覆盖已有数据）
    main_table = db_manager.prepare_features_table("image_features_clip")
    
    # 获取已编码的文件列表
    encoded_files = set(db_manager.get_encoded_files(main_table))
    print(f"数据库中已有 {len(encoded_files)} 个编码文件")
    
    # 扫描图像目录
    all_image_files = scan_images_directory()
    if not all_image_files:
        print("没有找到图像文件，退出")
        db_manager.close()
        return []
    
    # 过滤出未编码的文件
    image_files = [f for f in all_image_files if f not in encoded_files]
    print(f"找到 {len(image_files)} 个未编码的图像文件")
    
    if not image_files:
        print("所有图像文件已编码，无需进一步处理")
        db_manager.close()
        return []
    
    # 显示前几个未编码图像文件的路径
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
    
    if success_count > 0:
        print(f"总用时: {elapsed_time:.2f} 秒, 平均每张图像 {elapsed_time/success_count:.4f} 秒")
    
    # 关闭数据库连接
    db_manager.close()
    
    # 返回新处理的文件列表
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


def append_to_split_dataset(new_files):
    """将新编码的图像追加到现有的训练集、验证集和测试集中"""
    if not new_files:
        print("没有新文件需要添加到数据集划分中")
        return None
        
    db_manager = DatabaseManager()
    
    try:
        # 检查训练/验证/测试表是否存在
        required_tables = ["image_features_clip_train", "image_features_clip_val", "image_features_clip_test"]
        existing_tables = []
        
        for table in required_tables:
            cursor = db_manager.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                existing_tables.append(table)
                
        if len(existing_tables) < 3:
            print(f"错误: 缺少必要的表。 现有: {existing_tables}")
            print("将创建新的表并进行完整的数据集划分")
            db_manager.close()
            from dataset_split import split_encoded_images
            # 获取所有已编码文件（包括新文件）
            db_manager = DatabaseManager()
            all_encoded = db_manager.get_encoded_files("image_features_clip")
            db_manager.close()
            return split_encoded_images(all_encoded, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05)
        
        # 获取现有表中的文件数量
        train_count = db_manager.fetchall("SELECT COUNT(*) as count FROM image_features_clip_train")[0]['count']
        val_count = db_manager.fetchall("SELECT COUNT(*) as count FROM image_features_clip_val")[0]['count']
        test_count = db_manager.fetchall("SELECT COUNT(*) as count FROM image_features_clip_test")[0]['count']
        
        total_existing = train_count + val_count + test_count
        print(f"现有数据集: 训练集 {train_count}, 验证集 {val_count}, 测试集 {test_count}")
        
        # 计算当前比例
        train_ratio = train_count / total_existing if total_existing > 0 else 0.85
        val_ratio = val_count / total_existing if total_existing > 0 else 0.10
        test_ratio = test_count / total_existing if total_existing > 0 else 0.05
        
        print(f"现有数据集比例: 训练集 {train_ratio:.2f}, 验证集 {val_ratio:.2f}, 测试集 {test_ratio:.2f}")
        
        # 随机打乱新文件
        random.shuffle(new_files)
        
        # 按照现有比例分配新文件
        new_total = len(new_files)
        new_train_end = int(new_total * train_ratio)
        new_val_end = new_train_end + int(new_total * val_ratio)
        
        new_train_files = new_files[:new_train_end]
        new_val_files = new_files[new_train_end:new_val_end]
        new_test_files = new_files[new_val_end:]
        
        print(f"新文件分配: 训练集 {len(new_train_files)}, 验证集 {len(new_val_files)}, 测试集 {len(new_test_files)}")
        
        # 追加到现有表
        append_to_split(db_manager, "image_features_clip", "image_features_clip_train", new_train_files, "训练集")
        append_to_split(db_manager, "image_features_clip", "image_features_clip_val", new_val_files, "验证集")
        append_to_split(db_manager, "image_features_clip", "image_features_clip_test", new_test_files, "测试集")
        
        # 更新后的统计信息
        train_count += len(new_train_files)
        val_count += len(new_val_files)
        test_count += len(new_test_files)
        total_count = train_count + val_count + test_count
        
        # 计算新的比例
        new_train_ratio = train_count / total_count if total_count > 0 else 0
        new_val_ratio = val_count / total_count if total_count > 0 else 0
        new_test_ratio = test_count / total_count if total_count > 0 else 0
        
        # 返回结果
        return {
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'total': total_count,
            'train_ratio': new_train_ratio,
            'val_ratio': new_val_ratio,
            'test_ratio': new_test_ratio,
            'new_train': len(new_train_files),
            'new_val': len(new_val_files),
            'new_test': len(new_test_files),
            'new_total': new_total
        }
            
    except Exception as e:
        print(f"追加到数据集划分时出错: {str(e)}")
        return None
    finally:
        db_manager.close()


def append_to_split(db_manager, source_table, target_table, file_names, split_name):
    """将新文件特征从源表复制到目标表"""
    if not file_names:
        print(f"{split_name}没有新文件需要添加")
        return
        
    print(f"正在向{split_name}添加 {len(file_names)} 个新文件...")
    
    # 批处理大小
    batch_size = 100
    success_count = 0
    
    for i in tqdm(range(0, len(file_names), batch_size), desc=f"处理{split_name}"):
        batch_files = file_names[i:i+batch_size]
        
        for file_name in batch_files:
            # 检查目标表中是否已存在
            check = db_manager.fetchall(
                f"SELECT id FROM {target_table} WHERE file_name = ?",
                (file_name,)
            )
            
            if check:
                # 文件已存在于目标表中，跳过
                continue
                
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
    
    print(f"{split_name}处理完成: {success_count}/{len(file_names)} 个文件成功添加")


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
    print(f"开始处理图像编码...")
    print(f"数据库路径: {DB_PATH}")
    print(f"图像目录: {IMAGES_DIR}")
    print(f"CLIP模型类型: {CLIP_MODEL_TYPE}")
    
    # 检查图像目录是否存在
    if not os.path.exists(IMAGES_DIR):
        print(f"错误: 图像目录不存在 {IMAGES_DIR}")
    else:
        # 步骤1: 仅编码新图像
        processed_files = encode_all_images()
        
        # 步骤2: 追加到现有的数据集划分中
        if processed_files:
            # 将新编码的图像追加到现有的训练/验证/测试集
            split_stats = append_to_split_dataset(processed_files)
            
            if split_stats:
                print("\n更新后的数据集划分结果:")
                print(f"训练集: {split_stats['train']} 图像 ({split_stats['train_ratio']*100:.1f}%)")
                print(f"验证集: {split_stats['val']} 图像 ({split_stats['val_ratio']*100:.1f}%)")
                print(f"测试集: {split_stats['test']} 图像 ({split_stats['test_ratio']*100:.1f}%)")
                print(f"\n本次新增文件分配:")
                print(f"训练集: +{split_stats.get('new_train', 0)} 图像")
                print(f"验证集: +{split_stats.get('new_val', 0)} 图像")
                print(f"测试集: +{split_stats.get('new_test', 0)} 图像")
            
            # 验证特征结构
            if validate_feature_structures():
                print("✓ 训练集、验证集和测试集的特征结构一致")
            else:
                print("✗ 特征结构不一致，请检查数据")
        else:
            print("没有新图像需要处理，跳过数据集划分")
    
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