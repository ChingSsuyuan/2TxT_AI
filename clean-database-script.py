import os
import sqlite3
import shutil
import sys

print("="*70)
print("清空COCO图像标题数据库及下载的图片")
print("="*70)

# 设置路径
base_dir = 'coco_image_title_data'
db_path = os.path.join(base_dir, 'image_title_database.db')
images_dir = os.path.join(base_dir, 'images')
vocab_dir = os.path.join(base_dir, 'vocabulary')

# 检查数据库是否存在
if not os.path.exists(db_path):
    print(f"警告: 数据库文件不存在: {db_path}")
    print("没有需要清空的数据库。")
    sys.exit(0)

# 确认操作
print("此操作将会:")
print(f"1. 清空数据库中的所有表 ({db_path})")
print(f"2. 删除所有下载的图片 ({images_dir})")
print(f"3. 清理词汇表数据 ({vocab_dir})")
print("\n警告: 此操作不可逆!")

confirmation = input("\n请输入 'yes' 确认执行清空操作: ")
if confirmation.lower() != 'yes':
    print("操作已取消。")
    sys.exit(0)

# 连接到数据库
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取当前表的信息
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    if tables:
        print("\n清空数据库表:")
        # 禁用外键约束以便顺利删除数据
        cursor.execute("PRAGMA foreign_keys = OFF")
        
        # 清空所有表
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DELETE FROM {table_name}")
            print(f"- 已清空表: {table_name}")
        
        # 重置自增ID
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table_name}'")
            except sqlite3.OperationalError:
                # 如果表没有自增列，可能会出错，忽略即可
                pass
        
        # 重新启用外键约束
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # 提交更改
        conn.commit()
        print("所有表已清空，自增ID已重置。")
    else:
        print("数据库中没有表。")
    
    # 关闭数据库连接
    conn.close()
    
except sqlite3.Error as e:
    print(f"清空数据库时出错: {e}")
    sys.exit(1)

# 清理图片目录
if os.path.exists(images_dir):
    try:
        # 删除并重新创建目录
        shutil.rmtree(images_dir)
        os.makedirs(images_dir)
        print(f"\n已清空图片目录: {images_dir}")
    except Exception as e:
        print(f"清空图片目录时出错: {e}")

# 清理词汇表目录 (如果存在)
if os.path.exists(vocab_dir):
    try:
        # 删除并重新创建目录
        shutil.rmtree(vocab_dir)
        os.makedirs(vocab_dir)
        print(f"已清空词汇表目录: {vocab_dir}")
    except Exception as e:
        print(f"清空词汇表目录时出错: {e}")

print("\n数据库清空操作完成!")
print("你现在可以重新运行流水线来处理新的数据。")
