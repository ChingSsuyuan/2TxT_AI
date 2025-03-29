import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests
from io import BytesIO

class COCOTitleDataset(Dataset):
    """
    Dataset for image title generation using COCO
    """
    def __init__(self, dataset_info, transform=None, download=False, local_img_dir=None):
        """
        Args:
            dataset_info: List of dictionaries with image info and captions
            transform: PyTorch image transforms
            download: Whether to download images from URLs
            local_img_dir: Directory for local images if not downloading
        """
        self.dataset_info = dataset_info
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        self.download = download
        self.local_img_dir = local_img_dir
        
    def __len__(self):
        return len(self.dataset_info)
    
    def __getitem__(self, idx):
        img_info = self.dataset_info[idx]
        
        # Get image
        if self.download:
            # Download image from URL
            response = requests.get(img_info['coco_url'])
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Load from local directory
            img_path = os.path.join(self.local_img_dir, img_info['file_name'])
            image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # For title generation, we'll select one caption as the title
        # In practice, you might want to process these captions further
        caption = img_info['captions'][0]  # Using the first caption as title
        
        return {
            'image_id': img_info['image_id'],
            'image': image,
            'caption': caption
        }

# Create a custom title vocabulary builder
def build_title_vocabulary(dataset_info, min_freq=5):
    """
    Build a vocabulary from captions
    
    Args:
        dataset_info: List of dictionaries with image info and captions
        min_freq: Minimum frequency for a word to be included
        
    Returns:
        word2idx: Word to index mapping
        idx2word: Index to word mapping
    """
    word_freq = {}
    
    # Count word frequencies
    for item in dataset_info:
        for caption in item['captions']:
            # Simple preprocessing - lowercase and split by space
            # In practice, you'd want more sophisticated tokenization
            words = caption.lower().split()
            for word in words:
                # Remove punctuation (simple approach)
                word = word.strip('.,!?;:()[]{}""\'')
                if word:
                    word_freq[word] = word_freq.get(word, 0) + 1
    
    # Filter by frequency and create vocabulary
    filtered_words = [word for word, freq in word_freq.items() 
                     if freq >= min_freq]
    
    # Add special tokens
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    vocab = special_tokens + sorted(filtered_words)
    
    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    
    return word2idx, idx2word, word_freq

# Example of usage:
# Assuming small_dataset is already created
word2idx, idx2word, word_freq = build_title_vocabulary(small_dataset, min_freq=2)

print(f"Vocabulary size: {len(word2idx)}")
print(f"Most common words: {sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]}")

# Create dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# In practice, you would have images downloaded to a local directory
# For demonstration, we'll use download=True (will be slower)
dataset = COCOTitleDataset(
    small_dataset, 
    transform=transform,
    download=True  # Set to False if you have local images
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# Example of extracting a batch
sample_batch = next(iter(dataloader))
print(f"Batch image shape: {sample_batch['image'].shape}")
print(f"Sample caption: {sample_batch['caption'][0]}")
