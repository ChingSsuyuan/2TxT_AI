import os
import sys
import json
import torch
import sqlite3
import numpy as np
import argparse
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Import your model classes
from image_captioning_prefix_gpt import (
    ClipCaptionModel, ClipCaptionPrefix, 
    MappingType
)

def parse_args():
    parser = argparse.ArgumentParser(description='Test trained CLIP-GPT model for caption generation')
    
    # Model parameters
    parser.add_argument('--model_path', default='./gpt2_checkpoints/GPT2_model_best.pt', 
                        help='path to the trained model')
    
    # Database parameters
    parser.add_argument('--db_path', default='coco_image_title_data/image_title_database.db', 
                        help='path to SQLite database')
    parser.add_argument('--test_table', default='image_features_clip_test', 
                        help='table name with test CLIP embeddings')
    
    # Generation parameters
    parser.add_argument('--prefix_length', type=int, default=40, 
                        help='prefix length')
    parser.add_argument('--prefix_size', type=int, default=640, 
                        help='prefix size (640 for RN50x4)')
    parser.add_argument('--mapping_type', type=str, default='transformer', 
                        help='type of architecture (mlp/transformer)')
    parser.add_argument('--num_layers', type=int, default=8, 
                        help='number of layers in the mapper')
    parser.add_argument('--max_length', type=int, default=20, 
                        help='maximum length of generated caption')
    
    # Output parameters
    parser.add_argument('--output_file', default='caption_results.json', 
                        help='path to save results')
    
    return parser.parse_args()

def get_original_captions(db_path, file_name):
    """Get the original captions for an image from the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # First get the image id from the file name
    cursor.execute("SELECT id FROM images WHERE file_name = ?", (file_name,))
    image_row = cursor.fetchone()
    
    if not image_row:
        conn.close()
        return []
    
    image_id = image_row[0]
    
    # Then get all captions for this image
    cursor.execute("SELECT caption FROM captions WHERE image_id = ?", (image_id,))
    captions = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return captions

def load_test_data(db_path, test_table):
    """Load test data from the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all test samples
    cursor.execute(f"SELECT id, file_name, features FROM {test_table}")
    test_data = []
    
    for id, file_name, features_blob in cursor.fetchall():
        # Convert BLOB to numpy array then to torch tensor
        features = np.frombuffer(features_blob, dtype=np.float32).reshape(1, -1)
        test_data.append({
            'id': id,
            'file_name': file_name,
            'features': torch.tensor(features)
        })
    
    conn.close()
    return test_data

def generate_caption(model, prefix, tokenizer, max_length=20, device='cpu'):
    """Generate caption using greedy decoding"""
    model.eval()
    prefix = prefix.to(device)
    
    # Get prefix projections (the hidden representation from the CLIP embeddings)
    with torch.no_grad():
        prefix_projections = model.clip_project(prefix).view(-1, model.prefix_length, model.gpt_embedding_size)
    
    # Initialize with start token
    tokens = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    # Create initial input embeddings
    embedding_text = model.gpt.transformer.wte(tokens)
    embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
    
    # Set up attention mask if needed
    attention_mask = None
    
    # Generate text token by token
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model.gpt(inputs_embeds=embedding_cat, attention_mask=attention_mask)
            
        # Get predictions for next token
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        
        # If we generate EOS, stop
        if next_token_id.item() == tokenizer.eos_token_id:
            break
            
        # Add token to sequence
        tokens = torch.cat((tokens, next_token_id), dim=1)
        
        # Update embeddings for next iteration
        next_embedding = model.gpt.transformer.wte(next_token_id)
        embedding_cat = torch.cat((embedding_cat, next_embedding), dim=1)
        
        # Update attention mask if needed
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
    
    # Decode tokens to text
    caption = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
    return caption

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Set the mapping type
    mapping_type = MappingType.MLP if args.mapping_type == 'mlp' else MappingType.Transformer
    
    # Initialize the model
    model = ClipCaptionModel(
        prefix_length=args.prefix_length,
        clip_length=args.prefix_length,
        prefix_size=args.prefix_size,
        num_layers=args.num_layers,
        mapping_type=mapping_type
    )
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load test data
    print(f"Loading test data from {args.db_path}, table: {args.test_table}")
    test_data = load_test_data(args.db_path, args.test_table)
    print(f"Loaded {len(test_data)} test samples")
    
    if not test_data:
        print("No test data found. Exiting.")
        sys.exit(1)
    
    # Generate captions and compare with original
    results = []
    
    print("Generating captions and comparing...")
    for sample in tqdm(test_data):
        # Get original captions
        original_captions = get_original_captions(args.db_path, sample['file_name'])
        
        # Generate caption
        generated_caption = generate_caption(
            model, 
            sample['features'], 
            tokenizer, 
            max_length=args.max_length,
            device=device
        )
        
        # Add to results
        results.append({
            'id': sample['id'],
            'file_name': sample['file_name'],
            'generated_caption': generated_caption,
            'original_captions': original_captions
        })
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    # Print summary
    print(f"\n===== Caption Generation Results =====")
    print(f"Number of test samples: {len(test_data)}")
    print(f"Results saved to {args.output_file}")
    
    # Print some examples
    print("\n===== Example Captions =====")
    for i in range(min(5, len(results))):
        print(f"\nImage: {results[i]['file_name']}")
        print(f"Generated: {results[i]['generated_caption']}")
        print(f"Original captions:")
        for j, ref in enumerate(results[i]['original_captions']):
            print(f"  {j+1}. {ref}")

if __name__ == "__main__":
    main()