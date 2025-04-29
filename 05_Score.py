import os
import torch
import clip
import json
import random
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from collections import defaultdict
from transformers import GPT2Tokenizer

# Import necessary modules from mapping.py
from mapping import (
    MappingType,
    ClipCaptionModel,
    generate2
)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Function to download an image from URL
def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# Function to compute BLEU score
def compute_bleu(reference_captions, candidate_caption):
    smoothing = SmoothingFunction().method1
    # Tokenize captions
    tokenized_refs = [nltk.word_tokenize(cap.lower()) for cap in reference_captions]
    tokenized_candidate = nltk.word_tokenize(candidate_caption.lower())
    
    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
    bleu1 = sentence_bleu(tokenized_refs, tokenized_candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu(tokenized_refs, tokenized_candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu(tokenized_refs, tokenized_candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu(tokenized_refs, tokenized_candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4
    }

# Improved CIDEr implementation
def compute_cider(reference_captions, candidate_caption):
    # Tokenize captions
    tokenized_refs = [nltk.word_tokenize(cap.lower()) for cap in reference_captions]
    tokenized_candidate = nltk.word_tokenize(candidate_caption.lower())
    
    # Calculate CIDEr for different n-gram sizes and take the average
    cider_scores = []
    for n in range(1, 5):  # Use n-grams from 1 to 4
        cider_n = compute_cider_n(tokenized_refs, tokenized_candidate, n)
        cider_scores.append(cider_n)
    
    # Average CIDEr score across different n-gram sizes
    return sum(cider_scores) / len(cider_scores)

def compute_cider_n(tokenized_refs, tokenized_candidate, n):
    # Document frequency counter for all n-grams in the reference captions
    df = defaultdict(float)
    
    # For each reference, count its n-grams
    ref_ngram_counts = []
    
    # Count total number of references
    num_refs = len(tokenized_refs)
    
    # Process all reference captions to count n-grams
    for ref in tokenized_refs:
        # Count n-grams in this reference
        ref_ngrams = defaultdict(int)
        
        # Skip references that are too short for the current n-gram size
        if len(ref) < n:
            ref_ngram_counts.append(ref_ngrams)
            continue
            
        # Count n-grams
        for i in range(len(ref) - n + 1):
            ngram = tuple(ref[i:i+n])
            ref_ngrams[ngram] += 1
            # Mark this n-gram as observed in at least one reference
            df[ngram] = 1
        
        # Add this reference's n-gram counts to our collection
        ref_ngram_counts.append(ref_ngrams)
    
    # If candidate is too short for the n-gram size, return 0
    if len(tokenized_candidate) < n:
        return 0
    
    # Count n-grams in the candidate caption
    cand_ngrams = defaultdict(int)
    for i in range(len(tokenized_candidate) - n + 1):
        ngram = tuple(tokenized_candidate[i:i+n])
        cand_ngrams[ngram] += 1
    
    # Calculate the CIDEr score
    cider_score = 0
    
    # Calculate the idf values for all n-grams
    # Add 1 to denominator to avoid division by zero
    idf = {ngram: np.log(num_refs / (df_val + 1e-12)) for ngram, df_val in df.items()}
    
    # For each reference, calculate the CIDEr score
    for ref_ngrams in ref_ngram_counts:
        # Calculate the numerator (dot product of tf-idf vectors)
        numerator = 0
        for ngram, count in cand_ngrams.items():
            if ngram in ref_ngrams:
                # TF-IDF for this n-gram in both candidate and reference
                cand_tfidf = count * (idf.get(ngram, 0))
                ref_tfidf = ref_ngrams[ngram] * (idf.get(ngram, 0))
                numerator += cand_tfidf * ref_tfidf
        
        # Calculate the denominator (product of vector magnitudes)
        ref_norm = 0
        for ngram, count in ref_ngrams.items():
            ref_tfidf = count * (idf.get(ngram, 0))
            ref_norm += ref_tfidf ** 2
        ref_norm = np.sqrt(ref_norm) if ref_norm > 0 else 1e-12
        
        cand_norm = 0
        for ngram, count in cand_ngrams.items():
            cand_tfidf = count * (idf.get(ngram, 0))
            cand_norm += cand_tfidf ** 2
        cand_norm = np.sqrt(cand_norm) if cand_norm > 0 else 1e-12
        
        # Compute the cosine similarity
        if numerator > 0:
            cider_score += numerator / (ref_norm * cand_norm)
    
    # Normalize by the number of references
    cider_score = cider_score / num_refs * 10  # Scale for readability
    
    return cider_score

def evaluate_model_on_coco():
    # Create directories for data
    os.makedirs('coco_sample', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load('RN50x4', device=device, jit=False)
    
    # Load GPT2 tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load your model
    print("Loading caption generation model...")
    model_path = './checkpoints/clip_pro_prefix-best.pt'
    
    # Initialize model
    mapping_type = MappingType.Transformer
    model = ClipCaptionModel(
        prefix_length=40,
        clip_length=40,
        prefix_size=640,
        mapping_type=mapping_type
    )
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    model.to(device)
    
    # Load COCO annotations
    print("Loading COCO annotations...")
    anno_file = 'coco_data/annotations/captions_val2017.json'
    with open(anno_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create a mapping from image_id to captions
    image_to_captions = defaultdict(list)
    for anno in coco_data['annotations']:
        image_to_captions[anno['image_id']].append(anno['caption'])
    
    # Get image IDs with at least 5 captions
    valid_image_ids = [img_id for img_id, captions in image_to_captions.items() 
                      if len(captions) >= 5]
    
    # Randomly select 5 images
    random.seed(42)  # For reproducibility
    selected_image_ids = random.sample(valid_image_ids, 5)
    
    # Get image info (file paths, URLs)
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Download selected images and prepare data
    image_data = []
    
    print("Downloading and processing images...")
    for img_id in tqdm(selected_image_ids):
        img_info = image_id_to_info[img_id]
        img_url = img_info['coco_url']
        file_name = img_info['file_name']
        save_path = os.path.join('coco_sample', file_name)
        
        # Download image if it doesn't exist
        if not os.path.exists(save_path):
            if download_image(img_url, save_path):
                print(f"Downloaded: {file_name}")
            else:
                print(f"Failed to download: {file_name}")
                continue
        else:
            print(f"Image already exists: {file_name}")
        
        # Get 5 captions for this image
        captions = image_to_captions[img_id][:5]
        
        image_data.append({
            'image_id': img_id,
            'file_name': file_name,
            'local_path': save_path,
            'captions': captions
        })
    
    # Generate captions using the model and evaluate
    print("\nGenerating captions and evaluating...")
    results = []
    
    # Metrics for overall performance
    overall_metrics = {
        'BLEU-1': 0.0,
        'BLEU-2': 0.0,
        'BLEU-3': 0.0,
        'BLEU-4': 0.0,
        'CIDEr': 0.0
    }
    
    for img_data in tqdm(image_data):
        # Load and preprocess image
        image_path = img_data['local_path']
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Generate caption using the model
        with torch.no_grad():
            prefix = clip_model.encode_image(image_input).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, 40, -1)
        
        generated_caption = generate2(
            model, tokenizer,
            embed=prefix_embed,
            entry_length=30,
            temperature=1.0
        )
        
        # Clean up generated caption (some models include tokens like <|endoftext|>)
        generated_caption = generated_caption.replace("<|endoftext|>", "").strip()
        
        # Compute BLEU and CIDEr scores
        reference_captions = img_data['captions']
        bleu_scores = compute_bleu(reference_captions, generated_caption)
        cider_score = compute_cider(reference_captions, generated_caption)
        
        # Add scores to overall metrics
        for metric, value in bleu_scores.items():
            overall_metrics[metric] += value
        overall_metrics['CIDEr'] += cider_score
        
        # Save results
        result = {
            'image_id': img_data['image_id'],
            'file_name': img_data['file_name'],
            'reference_captions': reference_captions,
            'generated_caption': generated_caption,
            'bleu_scores': bleu_scores,
            'cider_score': cider_score
        }
        results.append(result)
    
    # Calculate average metrics
    num_images = len(results)
    for metric in overall_metrics:
        overall_metrics[metric] /= num_images
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 80)
    
    for result in results:
        print(f"Image: {result['file_name']}")
        print("\nReference Captions:")
        for i, caption in enumerate(result['reference_captions']):
            print(f"  {i+1}. {caption}")
        
        print(f"\nGenerated Caption: {result['generated_caption']}")
        
        print("\nScores:")
        for bleu_name, bleu_value in result['bleu_scores'].items():
            print(f"  {bleu_name}: {bleu_value:.4f}")
        
        print(f"  CIDEr: {result['cider_score']:.4f}")
        print("=" * 80)
    
    # Print average metrics
    print("\nAverage Metrics:")
    for metric_name, avg_value in overall_metrics.items():
        print(f"  {metric_name}: {avg_value:.4f}")
    
    # Also run leave-one-out evaluation of original captions
    print("\nLeave-one-out evaluation of original captions:")
    
    loo_metrics = {
        'BLEU-1': 0.0,
        'BLEU-2': 0.0,
        'BLEU-3': 0.0,
        'BLEU-4': 0.0,
        'CIDEr': 0.0
    }
    total_captions = 0
    
    for img_data in image_data:
        print(f"\nImage: {img_data['file_name']}")
        
        for i, candidate in enumerate(img_data['captions']):
            # Use all other captions as references (leave-one-out)
            references = [img_data['captions'][j] for j in range(len(img_data['captions'])) if j != i]
            
            # Compute BLEU scores
            bleu_scores = compute_bleu(references, candidate)
            
            # Compute CIDEr score
            cider_score = compute_cider(references, candidate)
            
            print(f"\nOriginal Caption {i+1}:")
            print(f"  \"{candidate}\"")
            print("  Scores (compared to other original captions):")
            for bleu_name, bleu_value in bleu_scores.items():
                print(f"    {bleu_name}: {bleu_value:.4f}")
                loo_metrics[bleu_name] += bleu_value
            
            print(f"    CIDEr: {cider_score:.4f}")
            loo_metrics['CIDEr'] += cider_score
            
            total_captions += 1
    
    # Calculate average leave-one-out metrics
    for metric in loo_metrics:
        loo_metrics[metric] /= total_captions
    
    print("\nAverage Leave-One-Out Metrics (Original Captions):")
    for metric_name, avg_value in loo_metrics.items():
        print(f"  {metric_name}: {avg_value:.4f}")
    
    # Compare model performance vs. human performance
    print("\nComparison: Model vs. Human Performance:")
    for metric in overall_metrics:
        model_score = overall_metrics[metric]
        human_score = loo_metrics[metric]
        diff = model_score - human_score
        print(f"  {metric}: Model: {model_score:.4f}, Human: {human_score:.4f}, Diff: {diff:.4f}")
    
    # Save results to JSON
    output_file = 'model_evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'images': results,
            'model_average_metrics': overall_metrics,
            'human_leave_one_out_metrics': loo_metrics
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Images saved in: coco_sample/")

if __name__ == "__main__":
    evaluate_model_on_coco()