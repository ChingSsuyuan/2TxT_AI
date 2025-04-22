import os
import sys
import json
import torch
import sqlite3
import numpy as np
import argparse
from tqdm import tqdm
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from rouge import Rouge

# Import your model classes
from modified_train_script import (
    ClipCaptionModel, ClipCaptionPrefix, 
    MappingType
)

# Try to download NLTK data needed for evaluation
try:
    nltk.download('wordnet')
    nltk.download('punkt')
except:
    print("NLTK data download failed, but we'll continue anyway")

def parse_args():
    parser = argparse.ArgumentParser(description='Test trained CLIP-GPT model for caption generation')
    
    # Model parameters
    parser.add_argument('--model_path', default='./best_model.pt', 
                        help='path to the trained model')
    parser.add_argument('--config_path', default='./best_hyperparams.json', 
                        help='path to model configuration')
    
    # Database parameters
    parser.add_argument('--db_path', default='coco_image_title_data/image_title_database.db', 
                        help='path to SQLite database')
    parser.add_argument('--test_table', default='image_features_clip_test', 
                        help='table name with test CLIP embeddings')
    
    # Generation parameters
    parser.add_argument('--prefix_length', type=int, default=40, 
                        help='prefix length')
    parser.add_argument('--num_layers', type=int, default=8, 
                        help='number of layers in the mapper')
    parser.add_argument('--mapping_type', type=str, default='transformer', 
                        help='type of architecture (mlp/transformer)')
    parser.add_argument('--only_prefix', action='store_true', default=True, 
                        help='if the model only trained the prefix mapper')
    parser.add_argument('--is_not_rn', action='store_true', default=False, 
                        help='CLIP backbone: False for RN, True for ViT')
    parser.add_argument('--beam_size', type=int, default=5, 
                        help='beam size for generation')
    parser.add_argument('--max_length', type=int, default=4, 
                        help='maximum length of generated caption')
    
    # Output parameters
    parser.add_argument('--output_file', default='caption_results.json', 
                        help='path to save results')
    
    return parser.parse_args()

def load_config_from_json(config_path):
    """Load configuration from JSON file if available"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    return None

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

def generate_caption(model, prefix, tokenizer, beam_size=5, max_length=20, device='cpu'):
    """Generate caption using beam search"""
    model.eval()
    prefix = prefix.to(device)
    
    # Get prefix projections
    prefix_projections = model.clip_project(prefix).view(-1, model.prefix_length, model.gpt_embedding_size)
    
    # Initialize beams with BOS token
    beams = [([tokenizer.bos_token_id], 0.0)]  # (sequence, score)
    complete_beams = []
    
    # Beam search
    for _ in range(max_length):
        candidates = []
        
        for seq, score in beams:
            if seq[-1] == tokenizer.eos_token_id:
                complete_beams.append((seq, score))
                continue
                
            tokens = torch.tensor([seq], device=device)
            embedding_text = model.gpt.transformer.wte(tokens)
            
            # Create input embeddings by concatenating prefix projections and text embeddings
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            
            # Get model predictions
            with torch.no_grad():
                outputs = model.gpt(inputs_embeds=embedding_cat)
                
            logits = outputs.logits[:, prefix_projections.size(1) + len(seq) - 1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top k next tokens
            top_probs, top_indices = probs.topk(beam_size)
            
            # Add new candidates
            for i in range(beam_size):
                token_id = top_indices[0, i].item()
                token_prob = top_probs[0, i].item()
                new_seq = seq + [token_id]
                new_score = score - np.log(token_prob)  # Using negative log likelihood as score
                candidates.append((new_seq, new_score))
        
        # If all beams are complete, break
        if len(candidates) == 0:
            break
            
        # Sort candidates by score and keep top beam_size
        candidates.sort(key=lambda x: x[1])
        beams = candidates[:beam_size]
        
        # If we have enough complete beams, we can stop
        if len(complete_beams) >= beam_size:
            break
    
    # Add incomplete beams to complete beams
    complete_beams.extend(beams)
    
    # Sort complete beams by score
    complete_beams.sort(key=lambda x: x[1])
    
    # Get the best sequence
    best_seq = complete_beams[0][0] if complete_beams else [tokenizer.bos_token_id]
    
    # Convert tokens to text
    caption = tokenizer.decode(best_seq, skip_special_tokens=True)
    return caption

def calculate_metrics(generated_caption, reference_captions):
    """Calculate evaluation metrics between generated and reference captions"""
    if not reference_captions:
        return {'bleu1': 0, 'bleu4': 0, 'meteor': 0, 'rouge_l': 0}
    
    # Tokenize captions
    tokenized_gen = nltk.word_tokenize(generated_caption.lower())
    tokenized_refs = [nltk.word_tokenize(ref.lower()) for ref in reference_captions]
    
    # Calculate BLEU scores
    try:
        bleu1 = sentence_bleu(tokenized_refs, tokenized_gen, weights=(1, 0, 0, 0))
        bleu4 = sentence_bleu(tokenized_refs, tokenized_gen, weights=(0.25, 0.25, 0.25, 0.25))
    except:
        bleu1 = 0
        bleu4 = 0
    
    # Calculate METEOR score
    try:
        meteor = np.mean([meteor_score([ref], generated_caption) for ref in reference_captions])
    except:
        meteor = 0
    
    # Calculate ROUGE-L score
    try:
        rouge = Rouge()
        rouge_scores = [rouge.get_scores(generated_caption, ref)[0]['rouge-l']['f'] for ref in reference_captions]
        rouge_l = np.mean(rouge_scores)
    except:
        rouge_l = 0
    
    return {
        'bleu1': float(bleu1),
        'bleu4': float(bleu4),
        'meteor': float(meteor),
        'rouge_l': float(rouge_l)
    }

def main():
    args = parse_args()
    
    # Load configuration if available
    config = load_config_from_json(args.config_path)
    if config:
        print(f"Loaded configuration from {args.config_path}")
        # Override command line arguments with config
        if 'num_layers' in config:
            args.num_layers = config['num_layers']
        if 'mapping_type' in config:
            args.mapping_type = config['mapping_type']
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Determine the feature dimension
    prefix_dim = 640 if not args.is_not_rn else 512
    
    # Set the mapping type
    mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    
    # Initialize the model
    if args.only_prefix:
        model = ClipCaptionPrefix(
            args.prefix_length, 
            prefix_size=prefix_dim,
            num_layers=args.num_layers, 
            mapping_type=mapping_type
        )
    else:
        model = ClipCaptionModel(
            args.prefix_length, 
            prefix_size=prefix_dim,
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
    
    # Generate captions and evaluate
    results = []
    avg_metrics = {'bleu1': 0, 'bleu4': 0, 'meteor': 0, 'rouge_l': 0}
    
    print("Generating captions and evaluating...")
    for sample in tqdm(test_data):
        # Get original captions
        original_captions = get_original_captions(args.db_path, sample['file_name'])
        
        # Generate caption
        generated_caption = generate_caption(
            model, 
            sample['features'], 
            tokenizer, 
            beam_size=args.beam_size, 
            max_length=args.max_length,
            device=device
        )
        
        # Calculate metrics
        metrics = calculate_metrics(generated_caption, original_captions)
        
        # Update average metrics
        for key in avg_metrics:
            avg_metrics[key] += metrics[key]
        
        # Add to results
        results.append({
            'id': sample['id'],
            'file_name': sample['file_name'],
            'generated_caption': generated_caption,
            'original_captions': original_captions,
            'metrics': metrics
        })
    
    # Calculate average metrics
    for key in avg_metrics:
        avg_metrics[key] /= len(test_data) if test_data else 1
    
    # Add average metrics to results
    results_with_avg = {
        'results': results,
        'average_metrics': avg_metrics
    }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results_with_avg, f, indent=2)
    
    # Print summary
    print("\n===== Caption Generation Results =====")
    print(f"Number of test samples: {len(test_data)}")
    print(f"Average BLEU-1: {avg_metrics['bleu1']:.4f}")
    print(f"Average BLEU-4: {avg_metrics['bleu4']:.4f}")
    print(f"Average METEOR: {avg_metrics['meteor']:.4f}")
    print(f"Average ROUGE-L: {avg_metrics['rouge_l']:.4f}")
    print(f"Results saved to {args.output_file}")
    
    # Print some examples
    print("\n===== Example Captions =====")
    for i in range(min(5, len(results))):
        print(f"\nImage: {results[i]['file_name']}")
        print(f"Generated: {results[i]['generated_caption']}")
        print(f"References:")
        for j, ref in enumerate(results[i]['original_captions']):
            print(f"  {j+1}. {ref}")

if __name__ == "__main__":
    main()
