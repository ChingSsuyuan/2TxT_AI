import os
import json
import sqlite3
import sys
from datetime import datetime
from pycocotools.coco import COCO

# 设置COCO API
dataDir = 'coco_data'
os.makedirs(os.path.join(dataDir, 'annotations'), exist_ok=True)
dataType = 'train2017'
annFile = f'{dataDir}/annotations/captions_{dataType}.json'

# 获取 COCO API 数据
def get_coco_data():
    try:
        coco = COCO(annFile)  # 使用注释文件加载COCO API
        print("COCO数据集加载成功")
        return coco
    except Exception as e:
        print(f"加载COCO数据集失败: {str(e)}")
        sys.exit(1)

# 获取所有图像 ID 和对应标题信息
def get_image_titles(coco, n=10):
    # 获取所有图像 ID
    imgIds = coco.getImgIds()
    print(f"数据集中的图像总数: {len(imgIds)}")
    
    # 选择前 n 张图像
    num_images = min(n, len(imgIds))
    selected_img_ids = imgIds[:num_images]
    print(f"选择了 {num_images} 张图像进行处理")
    
    # 获取选中图像的详细信息
    selected_imgs = coco.loadImgs(selected_img_ids)
    
    # 获取图像标题信息
    image_title_data = []
    for img_info in selected_imgs:
        img_id = img_info['id']
        file_name = img_info.get('file_name', f"image_{img_id}.jpg")
        
        # 获取该图像的所有标题
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        captions = [ann.get('caption', '') for ann in anns]
        image_title_data.append({
            "image_id": img_id,
            "file_name": file_name,
            "captions": captions
        })
        
    return image_title_data

# 初始化数据库
def init_db():
    db_path = 'coco_image_title_data/image_title_database.db'
    print(f"创建/连接数据库: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表 - 简化版本
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        coco_id INTEGER,
        file_name TEXT,
        width INTEGER,
        height INTEGER,
        created_at TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS captions (
        id INTEGER PRIMARY KEY,
        image_id INTEGER,
        caption TEXT,
        FOREIGN KEY (image_id) REFERENCES images (id)
    )
    ''')

    conn.commit()
    return conn, cursor

# 将数据存储到数据库
def store_data_to_db(cursor, image_title_data):
    for img_data in image_title_data:
        img_id = img_data['image_id']
        file_name = img_data['file_name']
        
        # 获取图像信息并存入数据库
        current_time = datetime.now()
        cursor.execute('''
        INSERT INTO images (coco_id, file_name, width, height, created_at)
        VALUES (?, ?, ?, ?, ?)
        ''', (img_id, file_name, 0, 0, current_time))
        
        # 获取插入的图像ID
        image_row_id = cursor.lastrowid
        
        # 将标题存入数据库
        for caption in img_data['captions']:
            cursor.execute('''
            INSERT INTO captions (image_id, caption)
            VALUES (?, ?)
            ''', (image_row_id, caption))

# 主程序
def main():
    print("="*70)
    print("从COCO数据集获取图片ID和标题信息并存入数据库")
    print("="*70)
    
    # 获取COCO数据
    coco = get_coco_data()
    
    # 获取图片ID和标题
    image_title_data = get_image_titles(coco, n=10)
    
    # 初始化数据库
    conn, cursor = init_db()
    
    # 存储数据到数据库
    store_data_to_db(cursor, image_title_data)
    conn.commit()
    
    # 验证数据库内容
    print("\n验证数据库内容:")
    print("-" * 50)
    
    # 查询图像数量
    cursor.execute("SELECT COUNT(*) FROM images")
    image_count = cursor.fetchone()[0]
    print(f"数据库中的图像数量: {image_count}")
    
    # 查询标题数量
    cursor.execute("SELECT COUNT(*) FROM captions")
    caption_count = cursor.fetchone()[0]
    print(f"数据库中的标题数量: {caption_count}")
    
    # 显示图像和标题的样例
    cursor.execute('''
    SELECT i.id, i.coco_id, i.file_name, c.caption
    FROM images i
    JOIN captions c ON i.id = c.image_id
    LIMIT 5
    ''')
    
    print("\n图像和标题样例:")
    print("-" * 50)
    print("ID | COCO_ID | 文件名 | 标题")
    print("-" * 50)
    
    for row in cursor.fetchall():
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    
    # 关闭数据库连接
    conn.close()
    
    print("\n=" * 50)
    print("处理完成。数据已保存到数据库。")
    print("=" * 50)

if __name__ == '__main__':
    main()