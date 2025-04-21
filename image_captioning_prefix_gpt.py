import os
import torch
import sqlite3
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
from tqdm import tqdm
import matplotlib.pyplot as plt

class PrefixGPTCaptionModel(nn.Module):
    def __init__(self, prefix_length=40, clip_dim=640, gpt_embedding_dim=768):
        super(PrefixGPTCaptionModel, self).__init__()
        # Mapping from CLIP image embedding dimension to GPT embedding dimension
        self.prefix_length = prefix_length
        self.clip_dim = clip_dim
        self.gpt_embedding_dim = gpt_embedding_dim
        
        # Linear projection from CLIP feature space to GPT-2 embedding space
        self.mapping_network = nn.Sequential(
            nn.Linear(clip_dim, gpt_embedding_dim * prefix_length),
            nn.ReLU(),
            nn.Linear(gpt_embedding_dim * prefix_length, gpt_embedding_dim * prefix_length)
        )
        
        # Initialize the GPT-2 model
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        # Freeze GPT-2 parameters for initial training
        for param in self.gpt.parameters():
            param.requires_grad = False
            
    def forward(self, clip_features, caption_ids, caption_attention_mask=None):
        # Transform CLIP features to prefix embeddings
        prefix_embeddings = self.mapping_network(clip_features)
        # Reshape to [batch_size, prefix_length, gpt_embedding_dim]
        prefix_embeddings = prefix_embeddings.view(-1, self.prefix_length, self.gpt_embedding_dim)
        
        # Get the GPT-2 embeddings for the captions
        caption_embeddings = self.gpt.transformer.wte(caption_ids)
        
        # Concatenate prefix embeddings with caption embeddings
        # Shape: [batch_size, prefix_length + caption_length, gpt_embedding_dim]
        combined_embeddings = torch.cat([prefix_embeddings, caption_embeddings], dim=1)
        
        # Create attention mask for prefix (always attend to prefix)
        prefix_attention_mask = torch.ones(caption_ids.shape[0], self.prefix_length, 
                                          device=caption_ids.device)
        
        # Combine prefix attention mask with caption attention mask
        if caption_attention_mask is None:
            caption_attention_mask = torch.ones(caption_ids.shape, device=caption_ids.device)
            
        combined_attention_mask = torch.cat([prefix_attention_mask, caption_attention_mask], dim=1)
        
        # Forward through GPT-2 with combined embeddings
        outputs = self.gpt(inputs_embeds=combined_embeddings, 
                          attention_mask=combined_attention_mask,
                          labels=None)
        
        # Get the logits
        logits = outputs.logits
        
        # The logits for predicting the next token are shifted
        # We need to ignore the prefix positions and the last position
        relevant_logits = logits[:, self.prefix_length-1:-1, :]
        relevant_labels = caption_ids
        
        return relevant_logits, relevant_labels
    
    def generate_caption(self, clip_features, tokenizer, max_length=50):
        """Generate caption from CLIP features"""
        with torch.no_grad():
            # Get prefix embeddings
            prefix_embeddings = self.mapping_network(clip_features)
            prefix_embeddings = prefix_embeddings.view(-1, self.prefix_length, self.gpt_embedding_dim)
            
            # Start with BOS token
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=clip_features.device)
            input_embeds = self.gpt.transformer.wte(input_ids)
            
            # Initialize with prefix embeddings
            embeds = prefix_embeddings.clone()
            
            # Generate tokens one by one
            for _ in range(max_length):
                # Forward pass through GPT-2
                outputs = self.gpt(inputs_embeds=embeds, labels=None)
                logits = outputs.logits
                
                # Get the next token prediction (last position)
                next_token_logits = logits[:, -1, :]
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # If EOS token is generated, stop
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Add token embedding to the sequence
                next_token_embed = self.gpt.transformer.wte(next_token)
                embeds = torch.cat([embeds, next_token_embed], dim=1)
                
                # Also keep track of tokens for decoding
                input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Decode the generated token IDs to text
            caption = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            return caption

class COCOImageCaptionDataset(Dataset):
    def __init__(self, db_path, image_folder, split="Training_Set", preprocess=None):
        self.db_path = db_path
        self.image_folder = image_folder
        self.split = split
        self.preprocess = preprocess
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all image-caption pairs from the specified split
        cursor.execute(f"SELECT id, file_name, caption FROM {split}")
        self.data = cursor.fetchall()
        conn.close()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_id, file_name, caption = self.data[idx]
        
        # Load and preprocess the image
        image_path = os.path.join(self.image_folder, file_name)
        image = Image.open(image_path).convert('RGB')
        
        if self.preprocess:
            image = self.preprocess(image)
        
        return {
            'img_id': img_id,
            'file_name': file_name,
            'image': image,
            'caption': caption
        }

def train_model(args):
    """Train the Prefix+GPT captioning model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load('RN50x4', device=device)
    clip_model.eval()  # Set to evaluation mode
    
    # Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset and dataloader
    train_dataset = COCOImageCaptionDataset(
        args.db_path, 
        args.image_folder,
        split="Training_Set",
        preprocess=preprocess
    )
    
    val_dataset = COCOImageCaptionDataset(
        args.db_path,
        args.image_folder,
        split="Validation_Set",
        preprocess=preprocess
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    model = PrefixGPTCaptionModel(
        prefix_length=args.prefix_length,
        clip_dim=640,  # RN50x4 CLIP model dimension
        gpt_embedding_dim=768  # GPT-2 small embedding dimension
    )
    model.to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(model.mapping_network.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            # Get image features from CLIP
            images = batch['image'].to(device)
            captions = batch['caption']
            
            with torch.no_grad():
                clip_features = clip_model.encode_image(images)
            
            # Tokenize captions
            tokenized = tokenizer(
                captions,
                padding="max_length",
                max_length=args.max_caption_length,
                truncation=True,
                return_tensors="pt"
            )
            caption_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, labels = model(clip_features, caption_ids, attention_mask)
            
            # Calculate loss
            # Reshape for cross entropy: [batch_size * seq_len, vocab_size]
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                captions = batch['caption']
                
                clip_features = clip_model.encode_image(images)
                
                tokenized = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=args.max_caption_length,
                    truncation=True,
                    return_tensors="pt"
                )
                caption_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)
                
                logits, labels = model(clip_features, caption_ids, attention_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"coco_prefix-{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'clip_dim': 640,
                'gpt_embedding_dim': 768,
                'prefix_length': args.prefix_length
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
            best_model_path = checkpoint_path
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "coco_prefix-trained-final.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'clip_dim': 640,
        'gpt_embedding_dim': 768,
        'prefix_length': args.prefix_length
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_curve.png'))
    
    return best_model_path, final_path

def find_best_model(checkpoint_dir):
    """Find the model with the lowest validation loss"""
    best_loss = float('inf')
    best_model_path = None
    
    # List all checkpoint files
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('coco_prefix-') and filename.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                val_loss = checkpoint.get('val_loss', float('inf'))
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_path = filepath
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    if best_model_path:
        print(f"Best model: {best_model_path} with validation loss: {best_loss:.4f}")
    else:
        print("No valid model checkpoints found")
    
    return best_model_path

def evaluate_model(model_path, db_path, image_folder, num_examples=5):
    """Evaluate the model by generating captions for sample images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CLIP model
    clip_model, preprocess = clip.load('RN50x4', device=device)
    clip_model.eval()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    prefix_length = checkpoint.get('prefix_length', 40)
    
    # Create model and load weights
    model = PrefixGPTCaptionModel(
        prefix_length=prefix_length,
        clip_dim=640,
        gpt_embedding_dim=768
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get some random test images
    cursor.execute("SELECT id, file_name, caption FROM Test_Set ORDER BY RANDOM() LIMIT ?", (num_examples,))
    examples = cursor.fetchall()
    
    results = []
    
    for img_id, file_name, gt_caption in examples:
        # Load and preprocess image
        image_path = os.path.join(image_folder, file_name)
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Get CLIP features
        with torch.no_grad():
            clip_features = clip_model.encode_image(image_input)
        
        # Generate caption
        generated_caption = model.generate_caption(clip_features, tokenizer)
        
        results.append({
            'image_id': img_id,
            'file_name': file_name,
            'ground_truth': gt_caption,
            'generated': generated_caption
        })
    
    conn.close()
    
    # Display results
    for i, result in enumerate(results):
        print(f"Example {i+1}:")
        print(f"Image: {result['file_name']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Generated: {result['generated']}")
        print("-" * 50)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate Prefix+GPT captioning model')
    
    # Training arguments
    parser.add_argument('--db_path', type=str, default='coco_image_title_data/image_title_database.db',
                       help='Path to the SQLite database')
    parser.add_argument('--image_folder', type=str, default='coco_image_title_data/images',
                       help='Path to image folder')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--prefix_length', type=int, default=40,
                       help='Length of prefix from image features')
    parser.add_argument('--max_caption_length', type=int, default=50,
                       help='Maximum caption length')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log interval for training')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'find_best', 'evaluate'],
                       help='Mode: train, find_best, or evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint for evaluation')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to evaluate')
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.mode == 'train':
        best_model_path, final_path = train_model(args)
        print(f"Training complete. Best model: {best_model_path}, Final model: {final_path}")
        
        # Evaluate the best model
        print("Evaluating best model...")
        evaluate_model(best_model_path, args.db_path, args.image_folder, args.num_examples)
        
    elif args.mode == 'find_best':
        best_model_path = find_best_model(args.checkpoint_dir)
        if best_model_path:
            print("Evaluating best model...")
            evaluate_model(best_model_path, args.db_path, args.image_folder, args.num_examples)
            
    elif args.mode == 'evaluate':
        if args.model_path is None:
            args.model_path = find_best_model(args.checkpoint_dir)
            
        if args.model_path:
            print(f"Evaluating model: {args.model_path}")
            evaluate_model(args.model_path, args.db_path, args.image_folder, args.num_examples)
        else:
            print("No model path provided or found")

if __name__ == '__main__':
    main()
