


import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image
import requests
from io import BytesIO

# Set up paths for COCO dataset
# You would need to download the dataset files first from https://cocodataset.org/#download
dataDir = 'coco_data'
dataType = 'train2017'  # Can be 'train2017', 'val2017', 'test2017'
annFile = f'{dataDir}/annotations/captions_{dataType}.json'

# Initialize COCO API
coco = COCO(annFile)

# Get all image IDs
imgIds = coco.getImgIds()
print(f"Total number of images: {len(imgIds)}")

# Get some sample images
sample_img_ids = imgIds[:5]
sample_imgs = coco.loadImgs(sample_img_ids)

# Display sample image with its captions
img_id = sample_imgs[0]['id']
img_info = coco.loadImgs(img_id)[0]
img_url = img_info['coco_url']

# Get captions for this image
ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
anns = coco.loadAnns(ann_ids)

# Display the image
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))
plt.figure(figsize=(8, 8))
plt.imshow(np.array(img))
plt.axis('off')
plt.title(f"Image ID: {img_id}")
plt.show()

# Print captions
print("Captions:")
for i, ann in enumerate(anns):
    print(f"{i+1}. {ann['caption']}")

# Function to extract a dataset for image title generation
def create_image_caption_dataset(coco_instance, num_samples=10000):
    """
    Create a dataset for image title generation with image paths and captions
    
    Args:
        coco_instance: COCO API instance
        num_samples: Number of samples to extract
        
    Returns:
        List of dictionaries with image info and captions
    """
    img_ids = coco_instance.getImgIds()[:num_samples]
    dataset = []
    
    for img_id in img_ids:
        img_info = coco_instance.loadImgs(img_id)[0]
        
        # Get captions for this image
        ann_ids = coco_instance.getAnnIds(imgIds=img_id)
        anns = coco_instance.loadAnns(ann_ids)
        
        captions = [ann['caption'] for ann in anns]
        
        dataset.append({
            'image_id': img_id,
            'file_name': img_info['file_name'],
            'coco_url': img_info['coco_url'],
            'captions': captions
        })
    
    return dataset

# Create a smaller dataset for demonstration
small_dataset = create_image_caption_dataset(coco, num_samples=100)
print(f"Created dataset with {len(small_dataset)} images")
print("Sample entry:", small_dataset[0])
