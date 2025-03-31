import os
import json
import sqlite3
import sys
import requests
import argparse
from datetime import datetime
from tqdm import tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description='创建COCO数据库并下载图片')
parser.add_argument('--num-images', type=int, default=20,
                    help='要下载的图片数量 (默认: 20)')
args = parser.parse_args()

# 检查和安装必要的包
try:
    from pycocotools.coco import COCO
except ImportError:
    print("安装pycocotools...")
    import pip
    pip.main(['install', 'pycocotools'])
    from pycocotools.coco import COCO

# 尝试导入进度条库
try:
    from tqdm import tqdm
except ImportError:
    print("安装tqdm进度条...")
    import pip
    pip.main(['install', 'tqdm'])
    from tqdm import tqdm

print("="*70)
print("从COCO数据集获取图片ID和标题信息并存入数据库")
print("并下载图片到本地")
print("="*70)

# 创建目录结构
base_dir = 'coco_image_title_data'
os.makedirs(base_dir, exist_ok=True)

# 为图像创建目录
images_dir = os.path.join(base_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

# 设置COCO API
dataDir = 'coco_data'
os.makedirs(os.path.join(dataDir, 'annotations'), exist_ok=True)
dataType = 'train2017'
annFile = f'{dataDir}/annotations/captions_{dataType}.json'

# 检查注释文件是否存在
if not os.path.exists(annFile):
    print(f"注释文件不存在: {annFile}")
    print("请从COCO官网下载注释文件,并放置在以下位置:")
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

# 添加新列
try:
    cursor.execute("ALTER TABLE images ADD COLUMN local_path TEXT")
    conn.commit()
    print("已向images表添加local_path列")
except sqlite3.OperationalError as e:
    # 如果列已经存在，会抛出错误，可以安全忽略
    if "duplicate column name" in str(e):
        print("local_path列已存在")
    else:
        print(f"修改表结构时出错: {str(e)}")

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

# 获取用户指定的图片数量
n = args.num_images
num_images = min(n, len(imgIds))
selected_img_ids = imgIds[:num_images]
print(f"选择了 {num_images} 张图像进行处理")

# 加载选中的图像信息
selected_imgs = coco.loadImgs(selected_img_ids)

# 下载图像的函数
def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # 如果请求不成功则抛出异常
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        return True
    except Exception as e:
        print(f"下载图像失败: {str(e)}")
        return False

# 处理每张图像
print("\n开始处理图像和下载...")
for i, img_info in enumerate(tqdm(selected_imgs, desc="处理图像")):
    img_id = img_info['id']
    file_name = img_info.get('file_name', f"image_{img_id}.jpg")
    width = img_info.get('width', 0)
    height = img_info.get('height', 0)
    
    print(f"\n处理图像 {i+1}/{num_images}, ID: {img_id}")
    
    # 构建图像URL和保存路径
    # COCO 2017数据集图像URL格式
    img_url = f"http://images.cocodataset.org/train2017/{file_name}"
    local_path = os.path.join(images_dir, file_name)
    
    # 下载图像
    print(f"下载图像: {img_url}")
    download_success = download_image(img_url, local_path)
    
    if download_success:
        print(f"图像已下载并保存到: {local_path}")
    else:
        print(f"无法下载图像，仅保存元数据")
        local_path = None
    
    # 将图像信息存入数据库
    current_time = datetime.now()
    cursor.execute('''
    INSERT INTO images (coco_id, file_name, width, height, local_path, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (img_id, file_name, width, height, local_path, current_time))
    
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

# 查询成功下载的图像数量
cursor.execute("SELECT COUNT(*) FROM images WHERE local_path IS NOT NULL")
downloaded_count = cursor.fetchone()[0]
print(f"成功下载的图像数量: {downloaded_count}")

# 查询标题数量
cursor.execute("SELECT COUNT(*) FROM captions")
caption_count = cursor.fetchone()[0]
print(f"数据库中的标题数量: {caption_count}")

# 显示图像和标题的样例
cursor.execute('''
SELECT i.id, i.coco_id, i.file_name, i.local_path, c.caption
FROM images i
JOIN captions c ON i.id = c.image_id
LIMIT 5
''')

print("\n图像和标题样例:")
print("-" * 70)
print("ID | COCO_ID | 文件名 | 本地路径 | 标题")
print("-" * 70)

for row in cursor.fetchall():
    print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")

# 关闭数据库连接
conn.close()

print("\n处理完成。数据已保存到数据库,图像已下载到本地。")
print(f"数据库路径: {db_path}")
print(f"图像保存路径: {images_dir}")