import sqlite3
import numpy as np
from tqdm import tqdm

# 使用原有的常量配置
DB_PATH = "coco_image_title_data/image_title_database.db"

def create_features_tables():
    """
    根据Training_Set, Validation_Set和Test_Set表创建三个特征表
    image_features_clip, image_features_clip_V和image_features_clip_T
    """
    print(f"开始准备特征表...")
    
    # 连接到数据库
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    try:
        # 检查源表是否存在
        source_tables = ["Training_Set", "Validation_Set", "Test_Set"]
        for source in source_tables:
            cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{source}'")
            if not cursor.fetchone():
                print(f"错误: 源表 {source} 不存在")
                return
        
        # 定义源表和目标表的映射
        dataset_mapping = {
            "Training_Set": "image_features_clip",
            "Validation_Set": "image_features_clip_V",
            "Test_Set": "image_features_clip_T"
        }
        
        # 为每个数据集创建特征表
        for source_table, target_table in dataset_mapping.items():
            # 检查目标表是否已存在，如果存在则删除
            cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{target_table}'")
            if cursor.fetchone():
                conn.execute(f"DROP TABLE {target_table}")
            
            # 创建新的特征表
            conn.execute(f"""
            CREATE TABLE {target_table} (
                id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                features BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # 创建索引
            conn.execute(f"CREATE INDEX idx_{target_table}_file_name ON {target_table}(file_name)")
            
            # 获取源表中的图像文件名数量
            cursor = conn.execute(f"SELECT COUNT(*) as count FROM {source_table}")
            count = cursor.fetchone()['count']
            
            print(f"为 {source_table} 创建了特征表 {target_table}，准备编码 {count} 个图像")
            
        print("\n特征表创建完成，可以开始编码过程")
        
        # 显示每个源表的前几个示例
        for source_table in source_tables:
            print(f"\n{source_table} 表中的前5个图像文件:")
            cursor = conn.execute(f"SELECT file_name FROM {source_table} LIMIT 5")
            for idx, row in enumerate(cursor.fetchall()):
                print(f"{idx+1}. {row['file_name']}")
            
    except Exception as e:
        print(f"准备特征表时出错: {str(e)}")
    finally:
        # 关闭数据库连接
        conn.close()

def get_dataset_split_counts():
    """获取各数据集的图像计数"""
    conn = sqlite3.connect(DB_PATH)
    counts = {}
    
    try:
        # 检查各个特征表的记录数
        feature_tables = ["image_features_clip", "image_features_clip_V", "image_features_clip_T"]
        total = 0
        
        for table in feature_tables:
            try:
                cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                count = cursor.fetchone()[0]
                counts[table] = count
                total += count
            except:
                counts[table] = 0
        
        # 计算比例
        if total > 0:
            counts["training"] = counts.get("image_features_clip", 0)
            counts["validation"] = counts.get("image_features_clip_V", 0)
            counts["test"] = counts.get("image_features_clip_T", 0)
            counts["total"] = total
            counts["train_ratio"] = counts["training"] / total if total > 0 else 0
            counts["val_ratio"] = counts["validation"] / total if total > 0 else 0
            counts["test_ratio"] = counts["test"] / total if total > 0 else 0
        
        return counts
    finally:
        conn.close()

def validate_feature_structures():
    """验证不同特征表中的特征结构是一致的"""
    conn = sqlite3.connect(DB_PATH)
    tables = ["image_features_clip", "image_features_clip_V", "image_features_clip_T"]
    feature_shapes = {}
    
    try:
        for table in tables:
            try:
                # 检查表是否存在且有数据
                cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                if cursor.fetchone()[0] == 0:
                    print(f"表 {table} 中没有记录，无法验证结构")
                    feature_shapes[table] = None
                    continue
                
                # 从表获取一条记录
                cursor = conn.execute(f"SELECT features FROM {table} LIMIT 1")
                features_blob = cursor.fetchone()[0]
                
                # 将二进制数据转换回NumPy数组
                feature_dim = 640  # RN50x4的特征维度
                features = np.frombuffer(features_blob, dtype=np.float32).reshape(1, feature_dim)
                feature_shapes[table] = features.shape
            except Exception as e:
                print(f"验证表 {table} 结构时出错: {str(e)}")
                feature_shapes[table] = None
    
        print("\n特征结构验证:")
        for table, shape in feature_shapes.items():
            if shape is not None:
                print(f"- {table}: 特征形状 {shape}")
            else:
                print(f"- {table}: 无法验证")
        
        # 检查所有形状是否一致
        shapes = [shape for shape in feature_shapes.values() if shape is not None]
        if len(shapes) >= 2:
            if all(shape == shapes[0] for shape in shapes):
                print("✓ 所有数据集的特征结构一致")
                return True
            else:
                print("✗ 不同数据集的特征结构不一致，请检查数据")
                return False
        else:
            print("无法验证特征结构一致性，至少需要两个有效的数据集")
            return False
            
    except Exception as e:
        print(f"验证特征结构时出错: {str(e)}")
        return False
    finally:
        conn.close()

def create_training_data_from_files():
    """
    此方法用于与原始代码保持兼容
    原来的拆分训练/验证集的方法不再使用
    """
    print("提示: 现在使用预定义的Training_Set、Validation_Set和Test_Set表作为数据源")
    create_features_tables()
    return

# 保持旧方法名称以兼容现有代码
split_encoded_images = create_training_data_from_files

if __name__ == "__main__":
    print(f"开始准备特征表，从现有数据集表中获取文件名...")
    print(f"数据库路径: {DB_PATH}")
    
    # 创建特征表
    create_features_tables()
    
    # 获取数据集统计信息
    stats = get_dataset_split_counts()
    if stats and stats["total"] > 0:
        print("\n数据集统计信息:")
        print(f"训练集 (image_features_clip): {stats['training']} 图像 ({stats['train_ratio']*100:.1f}%)")
        print(f"验证集 (image_features_clip_V): {stats['validation']} 图像 ({stats['val_ratio']*100:.1f}%)")
        print(f"测试集 (image_features_clip_T): {stats['test']} 图像 ({stats['test_ratio']*100:.1f}%)")
        print(f"总计: {stats['total']} 图像")
    else:
        print("\n特征表已创建，但尚未填充数据。请运行图像编码过程填充数据。")
    
    print("\n处理完成")