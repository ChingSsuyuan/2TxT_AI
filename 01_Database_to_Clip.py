import os
import json
import torch
import clip
from PIL import Image
import pickle
import requests
import argparse
from tqdm import tqdm
import random

# Download Coco images
def download_coco_images(annotation_file, output_dir, dataset_type, max_images=None):
    print(f"Loading: {annotation_file}")
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    images_info = data['images']
    print(f"Total number of images in the labelled file: {len(images_info)}")

    if max_images and max_images < len(images_info):
        images_info = random.sample(images_info, max_images)
        print(f"Pick {max_images} images")

    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    failed_count = 0
    
    for img_info in tqdm(images_info, desc=f"Downloading {dataset_type} images"):
        img_id = img_info['id']
        file_name = f"{int(img_id):012d}.jpg"
        output_path = os.path.join(output_dir, file_name)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"If exists: {output_path}")
            success_count += 1
            continue
        
        urls = [
            f"http://images.cocodataset.org/{dataset_type}2017/{file_name}",
            f"http://images.cocodataset.org/train2017/{file_name}",
            f"http://images.cocodataset.org/val2017/{file_name}"
        ]
        
        download_success = False
        for url in urls:
            try:
                print(f"Downloading {url}")
                response = requests.get(url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)

                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"Successfully downloaded the image to: {output_path}")
                        download_success = True
                        break
                    else:
                        print(f"Invalid downloaded file: {output_path}")
            except Exception as e:
                print(f" {url} fail: {str(e)}")
        
        if download_success:
            success_count += 1
        else:
            failed_count += 1
            print(f"All URL attempts failed to download image IDs: {img_id}")
    
    print(f"Download complete! Success: {success_count}, Failure: {failed_count}")
    return success_count

# Encoder->CLIP
def encode_images(image_dir, clip_model, preprocess, batch_size, output_prefix, dataset_type):
    """Coding images using the CLIP model"""
    print(f"Start encoding {dataset_type} images...")

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"Find {len(image_files)} images")

    if not image_files:
        return 0

    image_ids = [int(f.split('.')[0]) for f in image_files]

    all_embeddings = []
    all_captions = []

    annotation_file = f'./coco_data/annotations/captions_{dataset_type}2017.json'
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    id_to_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in id_to_captions:
            id_to_captions[img_id] = []
        id_to_captions[img_id].append(ann['caption'])
    

    total_processed = 0
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_ids = image_ids[i:i+batch_size]
        
        batch_embeddings = []
        batch_captions = []
        
        for j, (file_name, img_id) in enumerate(zip(batch_files, batch_ids)):
            file_path = os.path.join(image_dir, file_name)
            
            try:
                image = Image.open(file_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input).cpu()
                captions = id_to_captions.get(img_id, ["No caption available"])

                for caption in captions:
                    caption_item = {
                        "image_id": img_id,
                        "id": total_processed + j,
                        "caption": caption,
                        "clip_embedding": total_processed + j,
                        "filename": file_name
                    }
                    
                    batch_embeddings.append(image_features)
                    batch_captions.append(caption_item)
                
                print(f"Successfully encoded image: {file_name}")
                
            except Exception as e:
                print(f"process image {file_path} Error: {str(e)}")
        if batch_embeddings:
            batch_tensor = torch.cat(batch_embeddings, dim=0)
            batch_file = f"{output_prefix}_{dataset_type}_batch_{i//batch_size + 1}.pkl"
            
            with open(batch_file, 'wb') as f:
                pickle.dump({
                    "clip_embedding": batch_tensor,
                    "captions": batch_captions
                }, f)
            
            print(f"Save Batch {i//batch_size + 1} to {batch_file}, including {len(batch_captions)} ")
            total_processed += len(batch_captions)
    
    print(f"Encoding Complete! Processing{total_processed} captions")
    return total_processed

# Output pkl files
def merge_pkl_files(output_prefix, dataset_type):

    print(f"Merge {dataset_type} Files...")
    import glob
    batch_files = glob.glob(f"{output_prefix}_{dataset_type}_batch_*.pkl")
    batch_files.sort(key=lambda x: int(x.split('_batch_')[1].split('.')[0]))
    
    if not batch_files:
        print(f"Fail to find {dataset_type} ")
        return False
    
    print(f"Find {len(batch_files)} ")
    
    all_embeddings = []
    all_captions = []

    for batch_file in batch_files:
        try:
            print(f"Loading {batch_file}...")
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            if 'clip_embedding' in batch_data and 'captions' in batch_data:
                all_embeddings.append(batch_data['clip_embedding'])
                all_captions.extend(batch_data['captions'])
            else:
                print(f"Warning: {batch_file} invalid")
        except Exception as e:
            print(f"Error {batch_file} when: {str(e)}")
    
    if all_embeddings and all_captions:
        try:
            combined_embedding = torch.cat(all_embeddings, dim=0)
            merged_file = f"{output_prefix}_{dataset_type}_merged.pkl"
            
            print(f"Merge {len(all_embeddings)} Shape: {combined_embedding.shape}")
            print(f"Merge {len(all_captions)} captions")
            
            with open(merged_file, 'wb') as f:
                pickle.dump({
                    "clip_embedding": combined_embedding,
                    "captions": all_captions
                }, f)
            
            print(f"Successfully merged all batches into: {merged_file}")
            return True
        
        except Exception as e:
            print(f"Error when meraging: {str(e)}")
            return False
    else:
        print("No valid batch data found")
        return False


if __name__ == "__main__": #Main control module
    parser = argparse.ArgumentParser(description='Downloading COCO images and encoding them in CLIP')
    parser.add_argument('--dataset', type=str, choices=['train', 'val'], default='val',
                       help='Type of dataset to be processed (default: train)')
    parser.add_argument('--batch-size', type=int, default=2000,
                       help='Batch size for coding (default: 200)')
    parser.add_argument('--output-prefix', type=str, default='./CLIP_Pro',
                       help='Prefix of the output file (default: ./CLIP_Pro)')
    parser.add_argument('--download-only', action='store_true',
                       help='Download images only, no encoding')
    parser.add_argument('--encode-only', action='store_true',
                       help='Only encode existing images, no downloads')
    parser.add_argument('--merge-only', action='store_true',
                       help='Merge only existing batch files')
    parser.add_argument('--max-images', type=int, default=200,
                       help='Maximum number of images to download (default:200)')
    args = parser.parse_args()
    
    dataset_type = args.dataset
    image_dir = f"./coco_data/images/{dataset_type}2017"
    annotation_file = f'./coco_data/annotations/captions_{dataset_type}2017.json'
    
    print(f"Handling {dataset_type} set")
    print(f"Image Source   : {image_dir}")
    print(f"annotation File: {annotation_file}")
    print(f"Batch Size     : {args.batch_size}")
    print(f"Output Prefix  : {args.output_prefix}")
    
    if not args.encode_only and not args.merge_only:
        print("\n===== Step 1: Download the image =====")
        download_coco_images(annotation_file, image_dir, dataset_type, args.max_images)
    

    if args.download_only:
        print("Download only mode, task complete!")
        exit(0)

    if not args.merge_only:
        print("\n===== Step 2: Encoding the image =====")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using: {device}")
        
        clip_model_type = "RN50x4"
        print(f"Loading the CLIP model: {clip_model_type}")
        clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        
        encode_images(image_dir, clip_model, preprocess, args.batch_size, 
                     args.output_prefix, dataset_type)
    print("\n===== Step 3: Merge batch files =====")
    merge_success = merge_pkl_files(args.output_prefix, dataset_type)
    
    if merge_success:
        print(f"\nAll processing complete! Final file: {args.output_prefix}_{dataset_type}_merged.pkl")
    else:
        print("\nMerge failed, please check the previous error message.")