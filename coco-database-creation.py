import os
import json
import sqlite3
import sys
import requests
import argparse
import random
import time
from datetime import datetime
from tqdm import tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description='创建COCO数据库并下载图片')
parser.add_argument('--num-images', type=int, default=20,
                    help='要下载的图片数量 (默认: 20)')
parser.add_argument('--random-seed', type=int, default=None,
                    help='随机种子，用于图片选择 (默认: 使用系统时间)')
parser.add_argument('--force', action='store_true',
                    help='强制重新下载，即使数据库已有数据')
parser.add_argument('--debug', action='store_true',
                    help='启用调试输出')
cmd_args = parser.parse_args()

# 设置调试函数
def debug_print(*message_args, **kwargs):
    if cmd_args.debug:
        print("[DEBUG]", *message_args, **kwargs)

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
print("从COCO数据集获取动物图片并存入数据库")
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
instFile = f'{dataDir}/annotations/instances_{dataType}.json'

# 检查注释文件是否存在
if not os.path.exists(annFile):
    print(f"注释文件不存在: {annFile}")
    print("请从COCO官网下载注释文件,并放置在以下位置:")
    print(f"  {annFile}")
    print("下载链接: http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    sys.exit(1)

if not os.path.exists(instFile):
    print(f"实例分割注释文件不存在: {instFile}")
    print("请从COCO官网下载注释文件,并放置在以下位置:")
    print(f"  {instFile}")
    print("下载链接: http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    sys.exit(1)

# 初始化数据库
db_path = os.path.join(base_dir, 'image_title_database.db')
print(f"创建/连接数据库: {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 创建表 - 保持原始结构
cursor.execute('''
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    coco_id INTEGER UNIQUE,
    file_name TEXT,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP,
    local_path TEXT
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

# 添加索引以提高性能
try:
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_coco_id ON images(coco_id)")
    print("已添加索引")
except sqlite3.OperationalError as e:
    print(f"添加索引时出错: {e}")

conn.commit()

# 加载COCO API
print("加载COCO数据集...")
try:
    coco = COCO(annFile)
    coco_inst = COCO(instFile)
    print("COCO数据集加载成功")
except Exception as e:
    print(f"加载COCO数据集失败: {str(e)}")
    sys.exit(1)

# COCO数据集中的动物类别ID
animal_category_ids = [
    16,  # bird
    17,  # cat
    18,  # dog
    19,  # horse
    20,  # sheep
    21,  # cow
    22,  # elephant
    23,  # bear
    24,  # zebra
    25,  # giraffe
]

# 获取动物类别的名称映射
cat_info = coco_inst.loadCats(animal_category_ids)
animal_category_names = {cat['id']: cat['name'] for cat in cat_info}

# 获取包含动物的图像ID
print("查找包含动物的图像...")
animal_img_ids = []
for cat_id in animal_category_ids:
    cat_img_ids = coco_inst.getImgIds(catIds=[cat_id])
    debug_print(f"类别 {animal_category_names[cat_id]} (ID: {cat_id}) 包含 {len(cat_img_ids)} 张图像")
    animal_img_ids.extend(cat_img_ids)

# 移除重复的图像ID
animal_img_ids = list(set(animal_img_ids))
print(f"共找到 {len(animal_img_ids)} 张包含动物的唯一图像")

# 检查数据库中已有多少图片
cursor.execute("SELECT coco_id FROM images")
downloaded_ids = set(row[0] for row in cursor.fetchall())
print(f"数据库中已有 {len(downloaded_ids)} 张图片")

# 从所有ID中排除已下载的ID
available_ids = [img_id for img_id in animal_img_ids if img_id not in downloaded_ids]
debug_print(f"可用ID数量: {len(available_ids)}")

if not available_ids:
    print("警告: 所有动物图像已下载。如要重新下载，请先清空数据库或使用--force参数。")
    if cmd_args.force:
        print("使用--force参数，将重新下载部分图片...")
        available_ids = animal_img_ids  # 强制模式下使用所有ID
    else:
        sys.exit(0)

# 设置随机种子，保证不同运行之间选择不同图片
if cmd_args.random_seed is not None:
    random.seed(cmd_args.random_seed)
    print(f"使用指定的随机种子: {cmd_args.random_seed}")
else:
    # 使用当前时间戳作为种子，确保每次运行使用不同的种子
    seed = int(time.time())
    random.seed(seed)
    print(f"使用当前时间戳作为随机种子: {seed}")

# 获取用户指定的图片数量
n = min(cmd_args.num_images, len(available_ids))
if n < cmd_args.num_images:
    print(f"警告: 只有 {n} 张新图像可用，少于请求的 {cmd_args.num_images} 张")

# 随机选择图片IDs
selected_img_ids = random.sample(available_ids, n)
print(f"随机选择了 {n} 张动物图像进行处理")

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

# 获取图像包含的动物类别
def get_image_animal_categories(img_id):
    ann_ids = coco_inst.getAnnIds(imgIds=img_id)
    anns = coco_inst.loadAnns(ann_ids)
    
    category_ids = []
    for ann in anns:
        cat_id = ann.get('category_id')
        if cat_id in animal_category_ids:
            category_ids.append(cat_id)
    
    return list(set(category_ids))  # 移除重复的类别ID

# 处理每张图像
print("\n开始处理图像和下载...")
successful_downloads = 0
animal_stats = {}  # 用于统计动物出现次数

for i, img_info in enumerate(tqdm(selected_imgs, desc="处理图像")):
    img_id = img_info['id']
    file_name = img_info.get('file_name', f"image_{img_id}.jpg")
    width = img_info.get('width', 0)
    height = img_info.get('height', 0)
    
    print(f"\n处理图像 {i+1}/{n}, ID: {img_id}")
    
    # 再次检查该图片ID是否已存在（以防在此过程中有其他进程添加）
    cursor.execute("SELECT id FROM images WHERE coco_id = ?", (img_id,))
    existing = cursor.fetchone()
    
    if existing and not cmd_args.force:
        print(f"图像ID {img_id} 已在数据库中，跳过")
        continue
    
    # 获取图像包含的动物类别ID
    animal_category_ids_in_image = get_image_animal_categories(img_id)
    if not animal_category_ids_in_image:
        print(f"图像ID {img_id} 未找到动物类别，跳过")
        continue
    
    # 获取动物名称
    animal_names = [animal_category_names.get(cat_id, f"unknown") for cat_id in animal_category_ids_in_image]
    print(f"图像包含的动物: {', '.join(animal_names)}")
    
    # 更新动物统计信息
    for name in animal_names:
        animal_stats[name] = animal_stats.get(name, 0) + 1
    
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
    
    # 如果图片已存在且使用--force，则先删除旧记录
    if existing and cmd_args.force:
        try:
            # 先删除关联的标题
            cursor.execute("DELETE FROM captions WHERE image_id = ?", (existing[0],))
            # 再删除图片记录
            cursor.execute("DELETE FROM images WHERE id = ?", (existing[0],))
            print(f"已删除图像ID {img_id} 的旧记录")
        except sqlite3.Error as e:
            print(f"删除旧记录时出错: {e}")
            conn.rollback()
            continue
    
    # 将图像信息存入数据库
    current_time = datetime.now()
    try:
        cursor.execute('''
        INSERT INTO images (coco_id, file_name, width, height, local_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (img_id, file_name, width, height, local_path, current_time))
        
        # 获取插入的图像ID
        image_row_id = cursor.lastrowid
        
        # 保存动物名称到标题表中
        for name in animal_names:
            cursor.execute('''
            INSERT INTO captions (image_id, caption)
            VALUES (?, ?)
            ''', (image_row_id, name))  # 直接保存动物名称
            print(f"  添加动物名称: {name}")
        
        conn.commit()
        successful_downloads += 1
        
    except sqlite3.IntegrityError as e:
        print(f"插入数据时出错 (可能是重复键): {e}")
        conn.rollback()
        # 如果下载了图片但插入失败，删除图片文件
        if download_success and os.path.exists(local_path):
            try:
                os.remove(local_path)
                print(f"由于数据库插入失败，已删除图片文件: {local_path}")
            except:
                pass
    except Exception as e:
        print(f"处理图像时出错: {e}")
        conn.rollback()

# 验证数据库内容
print("\n验证数据库内容:")
print("-" * 50)

# 查询图像数量
cursor.execute("SELECT COUNT(*) FROM images")
image_count = cursor.fetchone()[0]
print(f"数据库中的图像总数量: {image_count}")

# 查询成功下载的图像数量
cursor.execute("SELECT COUNT(*) FROM images WHERE local_path IS NOT NULL")
downloaded_count = cursor.fetchone()[0]
print(f"成功下载的图像数量: {downloaded_count}")

# 查询动物名称数量
cursor.execute("SELECT COUNT(*) FROM captions")
caption_count = cursor.fetchone()[0]
print(f"数据库中的动物名称数量: {caption_count}")

# 查询本次运行新增的图片数量
print(f"本次运行成功添加 {successful_downloads} 张新图片")

# 显示动物统计信息
print("\n本次下载的动物图片统计:")
print("-" * 50)
for animal, count in sorted(animal_stats.items(), key=lambda x: x[1], reverse=True):
    print(f"  {animal}: {count}张图片")

# 统计数据库中各种动物的数量
cursor.execute('''
SELECT caption, COUNT(*) as count
FROM captions
GROUP BY caption
ORDER BY count DESC
''')

print("\n数据库中各动物类别统计:")
print("-" * 50)
for row in cursor.fetchall():
    if row[0] in animal_category_names.values():  # 确保是动物名称
        print(f"  {row[0]}: {row[1]}张图片")

# 显示图像和动物名称的样例
cursor.execute('''
SELECT i.id, i.coco_id, i.file_name, i.local_path, c.caption
FROM images i
JOIN captions c ON i.id = c.image_id
ORDER BY i.id DESC
LIMIT 10
''')

print("\n最近添加的图像和动物名称样例:")
print("-" * 70)
print("图像ID | COCO_ID | 文件名 | 本地路径 | 动物名称")
print("-" * 70)

for row in cursor.fetchall():
    print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")

# 关闭数据库连接
conn.close()

print("\n处理完成。数据已保存到数据库,图像已下载到本地。")
print(f"数据库路径: {db_path}")
print(f"图像保存路径: {images_dir}")