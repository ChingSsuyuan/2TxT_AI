import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
import os
import sys
import argparse
import json
import numpy as np
import sqlite3
import clip
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
from train import ClipCaptionModel, MappingType

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ImageDatabase:
    """Class to handle loading images from a directory or database"""
    
    def __init__(self, images_path, is_db=False, table_name=None):
        self.images_path = images_path
        self.is_db = is_db
        self.table_name = table_name
        self.image_ids = []
        self.file_names = []
        self.features = []
        
        if is_db:
            self._load_from_database()
        else:
            self._load_from_directory()
            
    def _load_from_database(self):
        """Load image features from SQLite database"""
        print(f"Loading data from database: {self.images_path}, table: {self.table_name}")
        conn = sqlite3.connect(self.images_path)
        c = conn.cursor()
        
        # Get all CLIP features
        c.execute(f"SELECT id, file_name, features FROM {self.table_name}")
        data = c.fetchall()
        conn.close()
        
        for id, file_name, features_blob in data:
            self.image_ids.append(id)
            self.file_names.append(file_name)
            
            # Convert binary BLOB to numpy array, then to torch.Tensor
            features = np.frombuffer(features_blob, dtype=np.float32).reshape(1, -1)
            self.features.append(torch.from_numpy(features).to(device))
            
        print(f"Loaded {len(self.image_ids)} samples from database")
        
    def _load_from_directory(self):
        """Load images from directory"""
        print(f"Loading images from directory: {self.images_path}")
        valid_extensions = ['.jpg', '.jpeg', '.png']
        for file_name in os.listdir(self.images_path):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in valid_extensions:
                self.file_names.append(file_name)
                
        print(f"Found {len(self.file_names)} images in directory")
    
    def get_image(self, idx):
        """Get image as PIL Image object"""
        if self.is_db:
            return None  # No images in DB mode, only features
        else:
            image_path = os.path.join(self.images_path, self.file_names[idx])
            return Image.open(image_path).convert("RGB")
    
    def __len__(self):
        return len(self.file_names)


class ModelEvaluator:
    """Evaluate CLIP-based captioning model performance"""
    
    def __init__(self, model, clip_model, tokenizer, preprocess, args):
        self.model = model.to(device)
        self.model.eval()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.args = args
        self.metrics = {
            'bleu-1': 0.0,
            'bleu-4': 0.0,
            'rouge-l': 0.0,
            'inference_time': 0.0
        }
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load pre-calculated modality offset if needed
        if args.add_modality_offset:
            try:
                import pickle
                with open('others/CLIP_embeddings_centers_info.pkl', 'rb') as f:
                    self.modality_offset = pickle.load(f)['offset_to_add_in_inference'].to(device)
            except FileNotFoundError:
                print("Modality offset file not found. Proceeding without offset.")
                self.modality_offset = None
        else:
            self.modality_offset = None
    
    def generate_beam(self, prefix_embed, beam_size=5, entry_length=67, temperature=1.0):
        """Generate caption using beam search"""
        
        model = self.model
        tokenizer = self.tokenizer
        
        stop_token_index = tokenizer.encode('.')[0]
        tokens = None
        scores = None
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        
        # Get starting token embeddings
        generated = torch.tensor([[tokenizer.bos_token_id]], device=device)
        
        # Initialize prefix_embed for each beam
        prefix_embed = prefix_embed.expand(beam_size, -1, -1)
        
        for i in range(entry_length):
            # For the first iteration
            if i == 0:
                outputs = model.gpt(inputs_embeds=prefix_embed, past_key_values=None, attention_mask=None)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1)
                
                # Get top-k tokens and their probabilities
                probs = logits.log_softmax(-1)
                next_tokens = torch.topk(logits, beam_size, dim=-1).indices
                next_token_scores = torch.gather(probs, -1, next_tokens)
                
                # Initialize beam tokens and scores
                tokens = torch.cat([torch.full((beam_size, 1), tokenizer.bos_token_id, device=device), 
                                  next_tokens.unsqueeze(-1)], dim=1)
                scores = next_token_scores.squeeze(-1)
                
                # Check if sequences should stop
                is_stopped = is_stopped + (next_tokens == stop_token_index)
                
            else:
                # For subsequent iterations
                outputs = model.gpt(inputs_embeds=prefix_embed, past_key_values=None, attention_mask=None)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1)
                
                # Get next token probs
                probs = logits.log_softmax(-1)
                
                # Get top token for each beam
                next_tokens = torch.argmax(probs, dim=-1)
                next_token_scores = torch.gather(probs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)
                
                # Update sequences
                tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)
                scores = scores + next_token_scores * (~is_stopped).float()
                
                # Check if sequences should stop
                is_stopped = is_stopped + (next_tokens == stop_token_index)
            
            if is_stopped.all():
                break
                
        # Select best sequence
        best_score_idx = torch.argmax(scores)
        best_tokens = tokens[best_score_idx]
        
        # Convert tokens to text
        gen_seq = tokenizer.decode(best_tokens.tolist(), skip_special_tokens=True)
            
        return gen_seq
    
    def generate(self, prefix_embed, entry_length=67, temperature=1.0):
        """Generate caption using greedy decoding"""
        
        model = self.model
        tokenizer = self.tokenizer
        
        stop_token_index = tokenizer.encode('.')[0]
        tokens = torch.tensor([tokenizer.bos_token_id], device=device).unsqueeze(0)
        
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=prefix_embed, past_key_values=None, attention_mask=None)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)
            
            if next_token.item() == stop_token_index:
                break
                
        return tokenizer.decode(tokens.squeeze().tolist(), skip_special_tokens=True)
    
    def evaluate(self, test_images, ground_truth=None):
        """Evaluate model performance on test images"""
        
        results = []
        all_references = []
        all_candidates = []
        inference_times = []
        
        # Process each image
        for idx in tqdm(range(len(test_images)), desc="Evaluating"):
            image = test_images.get_image(idx)
            
            if image is None and test_images.is_db:
                # Use pre-extracted features from database
                prefix = test_images.features[idx].to(device, dtype=torch.float32)
            else:
                # Extract features from image
                image_tensor = self.preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    prefix = self.clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
            
            # Normalize if needed
            if not self.args.dont_normalize_prefix:
                prefix = prefix / prefix.norm(2, -1)
            
            # Add modality offset if specified
            if self.args.add_modality_offset and self.modality_offset is not None:
                prefix = prefix + self.modality_offset
            
            # Project CLIP embedding to model embedding
            with torch.no_grad():
                # Use time package for CPU compatibility
                import time
                
                start_time = time.time()
                prefix_embed = self.model.clip_project(prefix).reshape(1, self.args.prefix_length, -1)
                
                # Generate caption
                if self.args.beam:
                    generated_caption = self.generate_beam(prefix_embed)
                else:
                    generated_caption = self.generate(prefix_embed)
                end_time = time.time()
                
                # Convert to milliseconds to match CUDA timing format
                inference_time = (end_time - start_time) * 1000
                inference_times.append(inference_time)
            
            # Log result
            file_name = test_images.file_names[idx]
            results.append({
                'file_name': file_name,
                'generated_caption': generated_caption.lower()
            })
            
            # Compute metrics if ground truth available
            if ground_truth is not None and file_name in ground_truth:
                reference = ground_truth[file_name]
                
                # For BLEU
                all_references.append([reference.split()])
                all_candidates.append(generated_caption.lower().split())
                
                # For ROUGE
                rouge_scores = self.rouge_scorer.score(reference, generated_caption.lower())
                
                # Add metrics to result
                results[-1]['reference'] = reference
                results[-1]['rouge_scores'] = {
                    'rouge1': rouge_scores['rouge1'].fmeasure,
                    'rouge2': rouge_scores['rouge2'].fmeasure,
                    'rougeL': rouge_scores['rougeL'].fmeasure
                }
        
        # Calculate overall metrics
        if ground_truth is not None:
            # BLEU-1
            bleu1_weights = (1.0, 0, 0, 0)
            bleu1 = corpus_bleu(all_references, all_candidates, weights=bleu1_weights)
            
            # BLEU-4
            bleu4_weights = (0.25, 0.25, 0.25, 0.25)
            bleu4 = corpus_bleu(all_references, all_candidates, weights=bleu4_weights)
            
            # Average ROUGE-L
            avg_rouge_l = sum(result['rouge_scores']['rougeL'] for result in results if 'rouge_scores' in result) / len(results)
            
            self.metrics['bleu-1'] = bleu1
            self.metrics['bleu-4'] = bleu4
            self.metrics['rouge-l'] = avg_rouge_l
        
        # Average inference time
        self.metrics['inference_time'] = sum(inference_times) / len(inference_times)
        
        return results
    
    def save_results(self, results, output_path):
        """Save evaluation results to file"""
        
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'results': results
            }, f, indent=2)
        
        print(f"Results saved to {output_path}")
        print(f"Metrics: {self.metrics}")


def load_ground_truth(gt_path):
    """Load ground truth captions from file"""
    
    ground_truth = {}
    
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            if 'file_name' in item and 'caption' in item:
                ground_truth[item['file_name']] = item['caption']
            elif 'image_id' in item and 'caption' in item:
                # Convert image_id to file_name format if needed
                file_name = f"COCO_val2014_{int(item['image_id']):012d}.jpg"
                ground_truth[file_name] = item['caption']
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-based captioning model")
    parser.add_argument('--checkpoint', default='./checkpoints/coco_prefix-009.pt', help='path to model checkpoint')
    parser.add_argument('--images_path', default='coco_image_title_data/test_images', help='path to test images directory or database')
    parser.add_argument('--ground_truth', default='', help='path to ground truth captions file')
    parser.add_argument('--output', default='./evaluation_results.json', help='path to save evaluation results')
    parser.add_argument('--is_db', action='store_true', help='whether images_path is a database')
    parser.add_argument('--table_name', default='image_features_clip', help='table name in database')
    parser.add_argument('--beam', action='store_true', default=True, help='use beam search for caption generation')
    parser.add_argument('--is_rn', action='store_true', default=True, help='use ResNet CLIP model instead of ViT (default: True)')
    parser.add_argument('--dont_normalize_prefix', action='store_true', default=False, help='do not normalize CLIP embeddings')
    parser.add_argument('--add_modality_offset', action='store_true', default=False, help='add modality offset to CLIP embeddings')
    parser.add_argument('--prefix_length', type=int, default=40, help='prefix length')
    parser.add_argument('--prefix_length_clip', type=int, default=40, help='prefix length for clip')
    parser.add_argument('--num_layers', type=int, default=8, help='number of layers in the mapper')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='type of mapping: mlp or transformer')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Check model dimensions from checkpoint to determine CLIP type
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract the shape of the clip_project.linear.weight or first layer in MLP
    clip_dim = None
    for key in ckpt.keys():
        if 'clip_project' in key and 'weight' in key:
            weight_shape = ckpt[key].shape
            if len(weight_shape) == 2:  # It's a linear layer
                clip_dim = weight_shape[1]  # Input dimension
                print(f"Detected CLIP dimension from checkpoint: {clip_dim}")
                break
    
    # Set CLIP model based on detected dimension
    if clip_dim == 640 or args.is_rn:
        clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
        prefix_dim = 640
        print("Using RN50x4 CLIP model (640-dim features)")
    else:
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        prefix_dim = 512
        print("Using ViT-B/32 CLIP model (512-dim features)")
    
    # Set mapping type
    mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    
    # Initialize model
    model = ClipCaptionModel(
        args.prefix_length, 
        clip_length=args.prefix_length_clip,
        prefix_size=prefix_dim,
        num_layers=args.num_layers, 
        mapping_type=mapping_type
    )
    
    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Model loaded from {args.checkpoint}")
    
    # Load test images
    test_images = ImageDatabase(args.images_path, is_db=args.is_db, table_name=args.table_name)
    
    # Load ground truth captions if available
    ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else None
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, clip_model, tokenizer, preprocess, args)
    
    # Run evaluation
    results = evaluator.evaluate(test_images, ground_truth)
    
    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()