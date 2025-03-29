import os
import json
import sqlite3
import sys
from datetime import datetime

# 检查和安装必要的包
try:
    from pycocotools.coco import COCO
except ImportError:
    print("安装pycocotools...")
    import pip
    pip.main(['install', 'pycocotools'])
    from pycocotools.coco import COCO

print("="*70)
print("从COCO数据集获取图片ID和标题信息并存入数据库")
print("="*70)

# 创建目录结构
base_dir = 'coco_image_title_data'
os.makedirs(base_dir, exist_ok=True)

# 设置COCO API
dataDir = 'coco_data'
os.makedirs(os.path.join(dataDir, 'annotations'), exist_ok=True)
dataType = 'train2017'
annFile = f'{dataDir}/annotations/captions_{dataType}.json'

# 检查注释文件是否存在
if not os.path.exists(annFile):
    print(f"注释文件不存在: {annFile}")
    print("请从COCO官网下载注释文件，并放置在以下位置:")
    print(f"  {annFile}")
    print("下载链接: http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    sys.exit(1)

# 初始化数据库
db_path = os.path.join(base_dir, 'image_title_database.db')
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

# 加载COCO API
print("加载COCO数据集...")
try:
    coco = COCO(annFile)
    print("COCO数据集加载成功")
except Exception as e:
    print(f"加载COCO数据集失败: {str(e)}")
    sys.exit(1)

# 获取所有图像ID
imgIds = coco.getImgIds()
print(f"数据集中的图像总数: {len(imgIds)}")

# 选择前n张图像 (默认为10)
n = 10  # 这里可以修改样本数量
num_images = min(n, len(imgIds))
selected_img_ids = imgIds[:num_images]
print(f"选择了 {num_images} 张图像进行处理")

# 加载选中的图像信息
selected_imgs = coco.loadImgs(selected_img_ids)

# 处理每张图像
for i, img_info in enumerate(selected_imgs):
    img_id = img_info['id']
    file_name = img_info.get('file_name', f"image_{img_id}.jpg")
    width = img_info.get('width', 0)
    height = img_info.get('height', 0)
    
    print(f"\n处理图像 {i+1}/{num_images}, ID: {img_id}")
    
    # 将图像信息存入数据库 (不下载图片)
    current_time = datetime.now()
    cursor.execute('''
    INSERT INTO images (coco_id, file_name, width, height, created_at)
    VALUES (?, ?, ?, ?, ?)
    ''', (img_id, file_name, width, height, current_time))
    
    # 获取插入的图像ID
    image_row_id = cursor.lastrowid
    
    # 获取该图像的所有标题
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    print(f"找到 {len(anns)} 个标题")
    
    # 将标题存入数据库
    for j, ann in enumerate(anns):
        caption = ann.get('caption', '')
        print(f"  标题 {j+1}: {caption}")
        
        cursor.execute('''
        INSERT INTO captions (image_id, caption)
        VALUES (?, ?)
        ''', (image_row_id, caption))
    
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
print(f"数据库路径: {db_path}")
print("=" * 50)