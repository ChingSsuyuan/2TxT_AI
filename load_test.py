import os
import json
import requests
from tqdm import tqdm

# ==== 可配置参数 ====
NUM_IMAGES = 20  # 要下载的图片数量
ANNOTATIONS_PATH = "coco_data/annotations/captions_val2017.json"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"
SAVE_DIR = "coco_image_title_data/test_images"

# ==== 创建保存目录 ====
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== 读取 COCO 标注文件 ====
with open(ANNOTATIONS_PATH, "r") as f:
    data = json.load(f)

# ==== 获取前 NUM_IMAGES 张图片 ====
images = data["images"][:NUM_IMAGES]

print(f"将从 {ANNOTATIONS_PATH} 中下载 {len(images)} 张图片到 {SAVE_DIR}")

# ==== 下载图片 ====
for img in tqdm(images):
    file_name = img["file_name"]  # 例如 '000000391895.jpg'
    url = IMAGE_BASE_URL + file_name
    save_path = os.path.join(SAVE_DIR, file_name)

    if os.path.exists(save_path):
        continue  # 已经下载则跳过

    try:
        resp = requests.get(url, stream=True)
        if resp.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
        else:
            print(f"❌ 下载失败: {file_name}")
    except Exception as e:
        print(f"⚠️ 下载出错: {file_name}，错误信息: {e}")

print("✅ 所有图片处理完成！")