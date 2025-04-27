import os
import json
import torch
import clip
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import subprocess
import tempfile

# Import evaluation metrics
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Import your model
from 03_predict import ClipCaptionModel, MappingType, generate2, generate_beam
from transformers import GPT2Tokenizer

class CaptionEvaluator:
    def __init__(self, model_path, test_dir, reference_captions=None, device='cpu'):
        """
        Initialize evaluator
        Args:
            model_path: Path to trained model checkpoint
            test_dir: Directory containing test images
            reference_captions: Dict of reference captions or path to JSON file
            device: Device to use (cpu/cuda)
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.test_dir = test_dir
        
        # Load model and necessary components
        self.clip_model, self.preprocess = clip.load('RN50x4', device=self.device, jit=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Load caption model
        mapping_type = MappingType.Transformer  # Match training config
        self.model = ClipCaptionModel(
            prefix_length=40,
            clip_length=40,
            prefix_size=640,
            mapping_type=mapping_type
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.eval()
        self.model.to(self.device)
        
        # Load or create reference captions
        if isinstance(reference_captions, str) and os.path.exists(reference_captions):
            with open(reference_captions, 'r') as f:
                self.reference_captions = json.load(f)
        elif isinstance(reference_captions, dict):
            self.reference_captions = reference_captions
        else:
            print("Warning: No reference captions provided. Evaluation will be limited.")
            self.reference_captions = {}
    
    def generate_caption(self, image_path, use_beam_search=True, beam_size=5, 
                        temperature=1.0, entry_length=30):
        """Generate caption for a single image"""
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image_input).to(self.device, dtype=torch.float32)
            prefix_embed = self.model.clip_project(prefix).reshape(1, 40, -1)
        
        if use_beam_search:
            generated_text = generate_beam(
                self.model, self.tokenizer,
                embed=prefix_embed,
                beam_size=beam_size,
                entry_length=entry_length,
                temperature=temperature
            )[0]
        else:
            generated_text = generate2(
                self.model, self.tokenizer,
                embed=prefix_embed,
                entry_length=entry_length,
                temperature=temperature
            )
        
        return generated_text
    
    def evaluate_bleu(self, candidates: List[str], references: List[List[str]]) -> Dict:
        """Calculate BLEU scores"""
        # Tokenize candidates and references
        candidate_tokens = [caption.split() for caption in candidates]
        reference_tokens = [[ref.split() for ref in refs] for refs in references]
        
        # Calculate BLEU scores (1-4)
        bleu_scores = {}
        for i in range(1, 5):
            weights = [1.0/i] * i + [0.0] * (4-i)
            bleu_scores[f'BLEU-{i}'] = corpus_bleu(reference_tokens, candidate_tokens, weights=weights)
        
        return bleu_scores
    
    def evaluate_meteor(self, candidates: List[str], references: List[List[str]]) -> float:
        """Calculate METEOR score"""
        meteor_scores = []
        for cand, refs in zip(candidates, references):
            # METEOR expects single reference, so we take the average
            score = np.mean([meteor_score([ref], cand) for ref in refs])
            meteor_scores.append(score)
        
        return np.mean(meteor_scores)
    
    def evaluate_rouge(self, candidates: List[str], references: List[List[str]]) -> Dict:
        """Calculate ROUGE scores"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for cand, refs in zip(candidates, references):
            # Get best score among references
            best_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
            for ref in refs:
                scores = scorer.score(ref, cand)
                for metric in best_scores:
                    best_scores[metric] = max(best_scores[metric], scores[metric].fmeasure)
            
            for metric in rouge_scores:
                rouge_scores[metric].append(best_scores[metric])
        
        # Average scores
        return {metric: np.mean(scores) for metric, scores in rouge_scores.items()}
    
    def evaluate_cider(self, candidates: Dict[str, str], references: Dict[str, List[str]]) -> float:
        """Calculate CIDEr score using external script"""
        # Save temporary files for evaluation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_res:
            json.dump([{'image_id': img_id, 'caption': caption} 
                      for img_id, caption in candidates.items()], f_res)
            res_file = f_res.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_ann:
            anns = []
            for img_id, refs in references.items():
                for i, ref in enumerate(refs):
                    anns.append({'image_id': img_id, 'id': i, 'caption': ref})
            json.dump(anns, f_ann)
            ann_file = f_ann.name
        
        # Try to use pycocoevalcap if available
        try:
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.eval import COCOEvalCap
            from pycocotools.coco import COCO
            
            coco = COCO(ann_file)
            coco_res = coco.loadRes(res_file)
            coco_eval = COCOEvalCap(coco, coco_res)
            coco_eval.params['image_id'] = list(candidates.keys())
            coco_eval.evaluate()
            
            cider_score = coco_eval.eval['CIDEr']
        except ImportError:
            print("Warning: CIDEr computation requires pycocoevalcap package")
            cider_score = 0.0
        
        # Clean up temporary files
        os.unlink(res_file)
        os.unlink(ann_file)
        
        return cider_score
    
    def evaluate_all(self, use_beam_search=True, beam_size=5, temperature=1.0, entry_length=30):
        """Run complete evaluation on test set"""
        # Get all test images
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend([f for f in os.listdir(self.test_dir) if f.lower().endswith(ext[1:])])
        
        print(f"Found {len(test_images)} test images")
        
        # Generate captions for all images
        generated_captions = {}
        candidates = []
        references = []
        
        for img_file in tqdm(test_images, desc="Generating captions"):
            img_path = os.path.join(self.test_dir, img_file)
            caption = self.generate_caption(
                img_path, 
                use_beam_search=use_beam_search,
                beam_size=beam_size,
                temperature=temperature,
                entry_length=entry_length
            )
            
            generated_captions[img_file] = caption
            candidates.append(caption)
            
            # Get reference captions if available
            if img_file in self.reference_captions:
                refs = self.reference_captions[img_file]
                if isinstance(refs, str):
                    refs = [refs]
                references.append(refs)
        
        # Calculate metrics
        results = {}
        
        if references:
            # BLEU scores
            bleu_scores = self.evaluate_bleu(candidates, references)
            results.update(bleu_scores)
            
            # METEOR score
            results['METEOR'] = self.evaluate_meteor(candidates, references)
            
            # ROUGE scores
            rouge_scores = self.evaluate_rouge(candidates, references)
            results.update(rouge_scores)
            
            # CIDEr score
            if self.reference_captions:
                results['CIDEr'] = self.evaluate_cider(generated_captions, self.reference_captions)
        
        # Save generated captions
        output_file = 'test_evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'metrics': results,
                'generated_captions': generated_captions
            }, f, indent=4)
        
        print("\nEvaluation Results:")
        print("-" * 30)
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
        
        return results, generated_captions


def main():
    parser = argparse.ArgumentParser(description='Evaluate image captioning model on test set')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--reference_captions', type=str, help='Path to JSON file with reference captions')
    parser.add_argument('--use_beam_search', action='store_true', help='Use beam search for generation')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--entry_length', type=int, default=30, help='Maximum caption length')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    evaluator = CaptionEvaluator(
        model_path=args.model_path,
        test_dir=args.test_dir,
        reference_captions=args.reference_captions,
        device=args.device
    )
    
    results, captions = evaluator.evaluate_all(
        use_beam_search=args.use_beam_search,
        beam_size=args.beam_size,
        temperature=args.temperature,
        entry_length=args.entry_length
    )
    
    print(f"\nResults saved to test_evaluation_results.json")


if __name__ == "__main__":
    main()
