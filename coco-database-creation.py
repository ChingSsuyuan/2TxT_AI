import os
import json
import sqlite3
import numpy as np
import requests
import sys
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
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
print("从COCO数据集获取图片和标题并存入数据库")
print("="*70)

# 创建目录结构
base_dir = 'coco_image_title_data'
image_dir = os.path.join(base_dir, 'images')
os.makedirs(image_dir, exist_ok=True)

# 设置COCO API
dataDir = 'coco_data'
os.makedirs(os.path.join(dataDir, 'annotations'), exist_ok=True)
dataType = 'train2017'
annFile = f'{dataDir}/annotations/captions_{dataType}.json'

# 如果注释文件不存在，尝试下载或创建样例
if not os.path.exists(annFile):
    # 尝试从COCO官网下载
    print(f"注释文件不存在: {annFile}")
    print("尝试下载COCO注释文件...")
    
    try:
        # 尝试下载captions标注文件
        annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        print(f"从 {annotation_url} 下载注释文件...")
        
        import urllib.request
        import zipfile
        
        # 下载zip文件
        zip_path = os.path.join(dataDir, "annotations.zip")
        urllib.request.urlretrieve(annotation_url, zip_path)
        
        # 解压文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataDir)
        
        # 删除zip文件
        os.remove(zip_path)
        print("成功下载并解压注释文件。")
        
    except Exception as e:
        print(f"下载失败: {str(e)}")
        print("创建样例注释文件用于测试...")
        
        # 创建一个简单的COCO格式注释文件
        sample_annotation = {
            "info": {"description": "COCO 2017 样例数据"},
            "images": [
                {
                    "id": 1, "file_name": "sample_1.jpg", "width": 640, "height": 480, 
                    "coco_url": "https://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg"
                },
                {
                    "id": 2, "file_name": "sample_2.jpg", "width": 640, "height": 480,
                    "coco_url": "https://farm7.staticflickr.com/6101/6267254182_66058a4f92_z.jpg"
                },
                {
                    "id": 3, "file_name": "sample_3.jpg", "width": 640, "height": 480,
                    "coco_url": "https://farm4.staticflickr.com/3777/8751975247_42015426a8_z.jpg"
                },
                {
                    "id": 4, "file_name": "sample_4.jpg", "width": 640, "height": 480,
                    "coco_url": "https://farm4.staticflickr.com/3736/9705628730_bf4b188de5_z.jpg"
                },
                {
                    "id": 5, "file_name": "sample_5.jpg", "width": 640, "height": 480,
                    "coco_url": "https://farm9.staticflickr.com/8375/8430648507_2b310c55fe_z.jpg"
                },
                {
                    "id": 6, "file_name": "sample_6.jpg", "width": 640, "height": 480,
                    "coco_url": "https://farm6.staticflickr.com/5580/14565176421_22d3976115_z.jpg"
                },
                {
                    "id": 7, "file_name": "sample_7.jpg", "width": 640, "height": 480,
                    "coco_url": "https://farm1.staticflickr.com/68/158234715_93dfd6b1b6_z.jpg"
                },
                {
                    "id": 8, "file_name": "sample_8.jpg", "width": 640, "height": 480,
                    "coco_url": "https://farm4.staticflickr.com/3011/2431862363_04191ab1eb_z.jpg"
                },
                {
                    "id": 9, "file_name": "sample_9.jpg", "width": 640, "height": 480,
                    "coco_url": "https://farm1.staticflickr.com/80/259391136_916775caff_z.jpg"
                },
                {
                    "id": 10, "file_name": "sample_10.jpg", "width": 640, "height": 480,
                    "coco_url": "https://farm4.staticflickr.com/3118/3094153621_780edecdfd_z.jpg"
                }
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "A black and white cat sitting on a chair."},
                {"id": 2, "image_id": 1, "caption": "A cat relaxing on a wooden chair."},
                {"id": 3, "image_id": 1, "caption": "The cat is sitting and looking at the camera."},
                {"id": 4, "image_id": 2, "caption": "A dog running in a green field."},
                {"id": 5, "image_id": 2, "caption": "A brown dog playing outdoors."},
                {"id": 6, "image_id": 2, "caption": "The dog is enjoying its time outside."},
                {"id": 7, "image_id": 3, "caption": "A person riding a bicycle on a path."},
                {"id": 8, "image_id": 3, "caption": "Someone cycling through a park."},
                {"id": 9, "image_id": 3, "caption": "A cyclist enjoying a sunny day outdoors."},
                {"id": 10, "image_id": 4, "caption": "A bowl of fresh fruits on a table."},
                {"id": 11, "image_id": 4, "caption": "Assorted fruits arranged in a bowl."},
                {"id": 12, "image_id": 4, "caption": "A colorful display of different fruits."},
                {"id": 13, "image_id": 5, "caption": "A red car parked on the street."},
                {"id": 14, "image_id": 5, "caption": "A vehicle parked near some buildings."},
                {"id": 15, "image_id": 5, "caption": "A sedan parked on an urban street."},
                {"id": 16, "image_id": 6, "caption": "People walking on a beach at sunset."},
                {"id": 17, "image_id": 6, "caption": "A group enjoying an evening at the beach."},
                {"id": 18, "image_id": 6, "caption": "Silhouettes of people against a sunset sky."},
                {"id": 19, "image_id": 7, "caption": "A tall building in a city skyline."},
                {"id": 20, "image_id": 7, "caption": "A skyscraper among other urban buildings."},
                {"id": 21, "image_id": 7, "caption": "A modern high-rise in a metropolitan area."},
                {"id": 22, "image_id": 8, "caption": "A cup of coffee with latte art."},
                {"id": 23, "image_id": 8, "caption": "A beautifully decorated coffee drink."},
                {"id": 24, "image_id": 8, "caption": "Frothy coffee with a design on top."},
                {"id": 25, "image_id": 9, "caption": "A person skiing down a snowy mountain."},
                {"id": 26, "image_id": 9, "caption": "A skier on a slope in winter."},
                {"id": 27, "image_id": 9, "caption": "Someone enjoying winter sports in the mountains."},
                {"id": 28, "image_id": 10, "caption": "A plate of pasta with sauce and herbs."},
                {"id": 29, "image_id": 10, "caption": "A delicious Italian dish ready to eat."},
                {"id": 30, "image_id": 10, "caption": "Spaghetti with tomato sauce garnished with herbs."}
            ]
        }
        
        with open(annFile, 'w') as f:
            json.dump(sample_annotation, f)
        
        print(f"已创建样例注释文件: {annFile}")

# 初始化数据库
db_path = os.path.join(base_dir, 'image_title_database.db')
print(f"创建/连接数据库: {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 创建表
cursor.execute('''
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    coco_id INTEGER,
    file_name TEXT,
    local_path TEXT,
    url TEXT,
    width INTEGER,
    height INTEGER,
    download_status BOOLEAN,
    download_time TIMESTAMP
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

# 选择前10张图像
num_images = min(10, len(imgIds))
selected_img_ids = imgIds[:num_images]
print(f"选择了 {num_images} 张图像进行处理")

# 加载选中的图像信息
selected_imgs = coco.loadImgs(selected_img_ids)

# 处理每张图像
for i, img_info in enumerate(selected_imgs):
    img_id = img_info['id']
    img_url = img_info.get('coco_url', '')
    file_name = img_info.get('file_name', f"image_{img_id}.jpg")
    width = img_info.get('width', 0)
    height = img_info.get('height', 0)
    
    print(f"\n处理图像 {i+1}/{num_images}, ID: {img_id}")
    
    # 下载图像
    local_path = os.path.join(image_dir, file_name)
    download_status = False
    
    try:
        if img_url:
            print(f"正在从 {img_url} 下载图像...")
            response = requests.get(img_url, timeout=10)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.save(local_path)
                download_status = True
                print(f"图像已保存到 {local_path}")
                
                # 显示图像
                plt.figure(figsize=(5, 5))
                plt.imshow(np.array(img))
                plt.axis('off')
                plt.title(f"Image ID: {img_id}")
                plt.savefig(os.path.join(base_dir, f"preview_{img_id}.png"))
                plt.close()
            else:
                print(f"下载失败,HTTP状态码: {response.status_code}")
        else:
            print("图像URL不可用")
    except Exception as e:
        print(f"下载图像时出错: {str(e)}")
    
    # 将图像信息存入数据库
    download_time = datetime.now() if download_status else None
    cursor.execute('''
    INSERT INTO images (coco_id, file_name, local_path, url, width, height, download_status, download_time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (img_id, file_name, local_path if download_status else None, 
         img_url, width, height, download_status, download_time))
    
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
SELECT i.id, i.coco_id, i.file_name, i.local_path, c.caption
FROM images i
JOIN captions c ON i.id = c.image_id
LIMIT 5
''')

print("\n图像和标题样例:")
print("-" * 50)
print("ID | COCO_ID | 文件名 | 本地路径 | 标题")
print("-" * 50)

for row in cursor.fetchall():
    print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")

# 关闭数据库连接
conn.close()

print("\n=" * 50)
print("处理完成。数据已保存到数据库和图像文件中。")
print(f"数据库路径: {db_path}")
print(f"图像目录: {image_dir}")
print("=" * 50)
