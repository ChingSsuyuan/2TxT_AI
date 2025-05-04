import os
import torch
import clip
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from transformers import GPT2Tokenizer
from torch import nn
import torch.nn.functional as nnf
from enum import Enum
from typing import Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
"""

This module can be used to evaluate generated captions  

"""
from mapping import (
    MappingType,
    MLP,
    MlpTransformer,
    MultiHeadAttention,
    TransformerLayer,
    Transformer,
    TransformerMapper,
    ClipCaptionModel,
    generate2
)

def evaluate_test_set(model_path, test_dir, device='cpu'):
    """Evaluating model performance on test sets"""
    
    print("Loading...")
    device = torch.device(device)
    clip_model, preprocess = clip.load('RN50x4', device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    mapping_type = MappingType.Transformer
    model = ClipCaptionModel(
        prefix_length=40,
        clip_length=40,
        prefix_size=640,
        mapping_type=mapping_type
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    model.to(device)
    
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend([f for f in os.listdir(test_dir) if f.lower().endswith(ext[1:])])
    
    print(f"Find {len(test_images)} images")
    generated_captions = []
    
    for img_file in tqdm(test_images, desc="Generated captions:"):
        img_path = os.path.join(test_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prefix = clip_model.encode_image(image_input).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, 40, -1)
        caption = generate2(
            model, tokenizer,
            embed=prefix_embed,
            entry_length=30,
            temperature=1.0
        )
        
        generated_captions.append({
            'image': img_file,
            'caption': caption
        })
    
    print("\n Assessment of indicators:")
    print("-" * 30)
    
    avg_length = np.mean([len(item['caption'].split()) for item in generated_captions])
    print(f"Average caption length: {avg_length:.2f} words")
    
    all_words = []
    for item in generated_captions:
        all_words.extend(item['caption'].lower().split())
    vocab_diversity = len(set(all_words)) / len(all_words)
    print(f"Lexical diversity: {vocab_diversity:.3f}")

    consistency_scores = []
    print("\n Calculating generation consistency...")
    for i in range(min(10, len(test_images))):  
        img_path = os.path.join(test_dir, test_images[i])
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        captions = []
        for _ in range(3):  
            with torch.no_grad():
                prefix = clip_model.encode_image(image_input).to(device, dtype=torch.float32)
                prefix_embed = model.clip_project(prefix).reshape(1, 40, -1)
            
            caption = generate2(
                model, tokenizer,
                embed=prefix_embed,
                entry_length=30,
                temperature=1.0
            )
            captions.append(caption)
        smoothing = SmoothingFunction().method1
        scores = []
        for i in range(len(captions)):
            for j in range(i+1, len(captions)):
                score = sentence_bleu([captions[i].split()], captions[j].split(), smoothing_function=smoothing)
                scores.append(score)
        
        if scores:
            consistency_scores.append(np.mean(scores))
    
    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    print(f"Generating consistency: {avg_consistency:.3f}")
    
    proper_endings = sum(1 for item in generated_captions if item['caption'].strip().endswith('.'))
    grammar_score = proper_endings / len(generated_captions)
    print(f"Grammatical integrity: {grammar_score:.3f}")
    appropriate_length = sum(1 for item in generated_captions 
                           if 5 <= len(item['caption'].split()) <= 20)
    length_score = appropriate_length / len(generated_captions)
    print(f"Appropriateness of length: {length_score:.3f}")
    results = {
        'metrics': {
            'Average caption length': avg_length,
            'Lexical diversity': vocab_diversity,
            'Generating consistency': avg_consistency,
            'Grammatical integrity': grammar_score,
            'Appropriateness of length': length_score
        },
        'generated_captions': generated_captions
    }
    overall_score = (vocab_diversity + avg_consistency + grammar_score + length_score) / 4
    results['metrics']['Aggregate score'] = overall_score
    output_file = 'test_evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nAggregate score: {overall_score:.3f}")
    print(f"Save to: {output_file}")
    
    print("\nExample generated caption:")
    print("-" * 50)
    for i in range(min(5, len(generated_captions))):
        print(f"Image: {generated_captions[i]['image']}")
        print(f"Caption: {generated_captions[i]['caption']}")
        print("-" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluating image caption generation models')
    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/clip_pro_prefix-best.pt',
                       help='model path')
    parser.add_argument('--test_dir', type=str, 
                       default='./Test_Set',
                       help='Test image file path')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Use device (cpu/cuda)')
    
    args = parser.parse_args()
    
    results = evaluate_test_set(args.model_path, args.test_dir, args.device)

if __name__ == "__main__":
    main()