import os
import torch
import clip
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import requests
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch import nn
import torch.nn.functional as nnf
from enum import Enum
from typing import Optional, List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from collections import defaultdict
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class MLP(nn.Module):
    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention

class TransformerLayer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:
                x = layer(x, y)
            elif self.enc_dec:  
                x = layer(x, x, mask)
            else:  
                x = layer(x, y, mask)
        return x

class TransformerMapper(nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device):
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(self, tokens, prefix, mask=None, labels=None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, 
                prefix_size: int = 640, num_layers: int = 8, 
                mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP(
                (prefix_size, (self.gpt_embedding_size * prefix_length) // 2, 
                self.gpt_embedding_size * prefix_length)
            )
        else:
            if clip_length is None:
                clip_length = 10
            self.clip_project = TransformerMapper(
                prefix_size, self.gpt_embedding_size, prefix_length,
                clip_length, num_layers
            )

def generate2(model, tokenizer, tokens=None, prompt=None, embed=None, entry_count=1,
             entry_length=30, top_p=0.8, temperature=1.0, stop_token: str = "."):
    model.eval()
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

# CIDEr Score Implementation
def calculate_cider(candidate, references, n=4):
    """
    Calculate CIDEr score between a candidate sentence and reference sentences
    
    Args:
        candidate: a list of tokens of the candidate sentence
        references: a list of lists of tokens, each list is a reference sentence
        n: maximum n-gram used for CIDEr score calculation
    
    Returns:
        CIDEr score
    """
    from collections import Counter
    import math
    
    def compute_tf(tokens, n):
        """Compute term frequency for a list of tokens"""
        n_grams = {}
        for k in range(1, n+1):
            for i in range(len(tokens)-k+1):
                g = tuple(tokens[i:i+k])
                n_grams[g] = n_grams.get(g, 0) + 1
        return n_grams
    
    def compute_doc_freq(refs, n):
        """Compute document frequency for references"""
        doc_freq = {}
        for ref in refs:
            # Compute n-grams for this reference
            n_grams = set()
            for k in range(1, n+1):
                for i in range(len(ref)-k+1):
                    n_grams.add(tuple(ref[i:i+k]))
            
            # Update document frequency
            for g in n_grams:
                doc_freq[g] = doc_freq.get(g, 0) + 1
        return doc_freq
    cand_tfs = compute_tf(candidate, n)
    doc_freq = compute_doc_freq(references, n)
    
    # Compute IDF weights
    num_refs = len(references)
    ref_len_avg = sum(len(ref) for ref in references) / num_refs
    
    cider_scores = []
    for k in range(1, n+1):
        # Compute TF-IDF for candidate
        tfidf_cand = {}
        for g, tf in cand_tfs.items():
            if len(g) == k:
                df = doc_freq.get(g, 0)
                tfidf_cand[g] = tf * math.log(num_refs / (1.0 + df))
        
        # Compute TF-IDF for references
        tfidf_refs = []
        for ref in references:
            ref_tfs = compute_tf(ref, n)
            tfidf_ref = {}
            for g, tf in ref_tfs.items():
                if len(g) == k:
                    df = doc_freq.get(g, 0)
                    tfidf_ref[g] = tf * math.log(num_refs / (1.0 + df))
            tfidf_refs.append(tfidf_ref)
        
        # Compute cosine similarity
        cider_k = 0.0
        for ref_tfidf in tfidf_refs:
            # Compute numerator
            numerator = 0.0
            for g, w in tfidf_cand.items():
                if len(g) == k and g in ref_tfidf:
                    numerator += w * ref_tfidf[g]
            
            # Compute norm for candidate
            norm_cand = math.sqrt(sum(w * w for g, w in tfidf_cand.items() if len(g) == k))
            # Compute norm for reference
            norm_ref = math.sqrt(sum(w * w for g, w in ref_tfidf.items() if len(g) == k))
            
            # Compute similarity
            if norm_cand > 0 and norm_ref > 0:
                cider_k += numerator / (norm_cand * norm_ref)
        
        if len(tfidf_refs) > 0:
            cider_k /= len(tfidf_refs)
        else:
            cider_k = 0.0
        
        cider_scores.append(cider_k)
    
    # Average over k
    return sum(cider_scores) / len(cider_scores)

def load_coco_captions(captions_file):
    """
    Load COCO captions from json file and organize them by image_id
    
    Args:
        captions_file: path to the captions json file (e.g., captions_val2017.json)
        
    Returns:
        A dictionary mapping from image_id to a list of captions for that image
        A dictionary mapping from filename to image_id
    """
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    
    # Map from image_id to captions
    captions_by_image_id = defaultdict(list)
    for annotation in captions_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        captions_by_image_id[image_id].append(caption)
    
    # Map from filename to image_id
    filename_to_image_id = {}
    for image in captions_data['images']:
        filename_to_image_id[image['file_name']] = image['id']
        # Also map from zeroed filename format (used in some COCO downloads)
        filename_to_image_id[f"{int(image['id']):012d}.jpg"] = image['id']
    
    # Create reverse mapping
    image_id_to_filename = {v: k for k, v in filename_to_image_id.items()}
    
    return captions_by_image_id, filename_to_image_id, image_id_to_filename

def download_coco_images(annotation_file, output_dir, max_images=5):
    """
    Download COCO images based on the annotation file
    
    Args:
        annotation_file: Path to the COCO annotations JSON file
        output_dir: Directory to save downloaded images
        max_images: Maximum number of images to download (default: 5)
    
    Returns:
        List of successfully downloaded image filenames
    """
    print(f"Loading: {annotation_file}")
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Load captions and create mapping
    captions_by_image_id, _, image_id_to_filename = load_coco_captions(annotation_file)
    
    # Find image IDs with exactly 5 captions
    image_ids_with_5_captions = [
        img_id for img_id, captions in captions_by_image_id.items() 
        if len(captions) == 5 and img_id in image_id_to_filename
    ]
    
    print(f"Found {len(image_ids_with_5_captions)} images with exactly 5 captions")
    
    # Select random subset if we have more than needed
    if max_images < len(image_ids_with_5_captions):
        selected_image_ids = random.sample(image_ids_with_5_captions, max_images)
    else:
        selected_image_ids = image_ids_with_5_captions[:max_images]
    
    print(f"Selected {len(selected_image_ids)} images for download")
    
    # Get image info for selected IDs
    images_info = []
    for img_id in selected_image_ids:
        # Find corresponding image info
        for img_info in data['images']:
            if img_info['id'] == img_id:
                images_info.append(img_info)
                break
    
    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    failed_count = 0
    downloaded_files = []
    
    for img_info in tqdm(images_info, desc=f"Downloading images"):
        img_id = img_info['id']
        file_name = f"{int(img_id):012d}.jpg"
        output_path = os.path.join(output_dir, file_name)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Image already exists: {output_path}")
            downloaded_files.append(file_name)
            success_count += 1
            continue
        
        urls = [
            f"http://images.cocodataset.org/val2017/{file_name}",
            f"http://images.cocodataset.org/train2017/{file_name}",
            img_info.get('coco_url', '')
        ]
        
        download_success = False
        for url in urls:
            if not url:  # Skip empty URLs
                continue
                
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
                        downloaded_files.append(file_name)
                        download_success = True
                        break
                    else:
                        print(f"Invalid downloaded file: {output_path}")
            except Exception as e:
                print(f"Failed to download {url}: {str(e)}")
        
        if download_success:
            success_count += 1
        else:
            failed_count += 1
            print(f"All URL attempts failed for image ID: {img_id}")
    
    print(f"Download complete! Success: {success_count}, Failure: {failed_count}")
    return downloaded_files

def evaluate_test_set(model_path, test_dir, captions_file=None, device='cpu', download_images=True, num_images=5):
    """
    Evaluating model performance on test sets using BLEU and CIDEr scores
    
    Args:
        model_path: Path to the trained model
        test_dir: Directory containing test images
        captions_file: Path to COCO captions JSON file
        device: Device to run evaluation on (cpu/cuda)
        download_images: Whether to download COCO images
        num_images: Number of images to evaluate
    """
    
    print("Loading model and CLIP...")
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
    
    # Download COCO images if requested
    downloaded_images = []
    if download_images and captions_file and os.path.exists(captions_file):
        downloaded_images = download_coco_images(captions_file, test_dir, num_images)
    
    # Load COCO captions
    captions_by_image_id = None
    filename_to_image_id = None
    if captions_file and os.path.exists(captions_file):
        print(f"Loading captions from {captions_file}")
        captions_by_image_id, filename_to_image_id, _ = load_coco_captions(captions_file)
        print(f"Loaded captions for {len(captions_by_image_id)} images")
    
    # Get test images - either downloaded ones or from directory
    test_images = []
    if downloaded_images:
        test_images = downloaded_images
        print(f"Using {len(test_images)} downloaded images for evaluation")
    else:
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            import glob
            pattern = os.path.join(test_dir, ext)
            test_images.extend([os.path.basename(f) for f in glob.glob(pattern)])
        
        if len(test_images) > num_images:
            test_images = test_images[:num_images]
        print(f"Using {len(test_images)} existing images in directory for evaluation")
    
    generated_captions = []
    
    for img_file in tqdm(test_images, desc="Generating captions:"):
        img_path = os.path.join(test_dir, img_file)
        
        # Make sure the image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_file} not found at {img_path}")
            continue
        
        # Generate caption
        try:
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
        except Exception as e:
            print(f"Error processing image {img_file}: {str(e)}")
    
    print("\nEvaluation Metrics:")
    print("-" * 30)
    
    # Calculate metrics if we have ground truth captions
    metrics = {}
    if captions_by_image_id and filename_to_image_id:
        bleu1_scores = []
        bleu4_scores = []
        cider_scores = []
        
        smoothing = SmoothingFunction().method1
        
        # For each generated caption
        for item in generated_captions:
            img_file = item['image']
            if img_file in filename_to_image_id:
                image_id = filename_to_image_id[img_file]
                if image_id in captions_by_image_id:
                    # Get ground truth references
                    references = captions_by_image_id[image_id]
                    
                    # Print out the references for each image
                    print(f"\nImage: {img_file}")
                    print(f"Generated caption: {item['caption']}")
                    print("Ground truth captions:")
                    for i, ref in enumerate(references):
                        print(f"  {i+1}: {ref}")
                    
                    # Tokenize references and candidate
                    tokenized_refs = [nltk.word_tokenize(ref.lower()) for ref in references]
                    tokenized_candidate = nltk.word_tokenize(item['caption'].lower())
                    
                    # Calculate BLEU-1
                    bleu1 = sentence_bleu(tokenized_refs, tokenized_candidate, 
                                         weights=(1, 0, 0, 0), smoothing_function=smoothing)
                    
                    # Calculate BLEU-4
                    bleu4 = sentence_bleu(tokenized_refs, tokenized_candidate, 
                                         weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                    
                    # Calculate CIDEr
                    cider = calculate_cider(tokenized_candidate, tokenized_refs)
                    
                    print(f"BLEU-1: {bleu1:.4f}")
                    print(f"BLEU-4: {bleu4:.4f}")
                    print(f"CIDEr: {cider:.4f}")
                    
                    bleu1_scores.append(bleu1)
                    bleu4_scores.append(bleu4)
                    cider_scores.append(cider)
        
        if bleu1_scores:
            avg_bleu1 = np.mean(bleu1_scores)
            avg_bleu4 = np.mean(bleu4_scores)
            avg_cider = np.mean(cider_scores)
            
            print("\nAverage Metrics:")
            print(f"BLEU-1: {avg_bleu1:.4f}")
            print(f"BLEU-4: {avg_bleu4:.4f}")
            print(f"CIDEr: {avg_cider:.4f}")
            
            metrics = {
                'BLEU-1': avg_bleu1,
                'BLEU-4': avg_bleu4,
                'CIDEr': avg_cider
            }
        else:
            print("Warning: No matching images found in ground truth captions")
    else:
        print("Warning: No ground truth captions provided or no matching images found.")
    
    # Original metrics
    print("\nOriginal Metrics:")
    print("-" * 30)
    
    avg_length = np.mean([len(item['caption'].split()) for item in generated_captions])
    print(f"Average caption length: {avg_length:.2f} words")
    
    all_words = []
    for item in generated_captions:
        all_words.extend(item['caption'].lower().split())
    vocab_diversity = len(set(all_words)) / len(all_words)
    print(f"Lexical diversity: {vocab_diversity:.3f}")

    consistency_scores = []
    print("\nCalculating generation consistency...")
    for i in range(min(num_images, len(generated_captions))):  
        img_file = generated_captions[i]['image']
        img_path = os.path.join(test_dir, img_file)
        
        try:
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
            for j in range(len(captions)):
                for k in range(j+1, len(captions)):
                    score = sentence_bleu([captions[j].split()], captions[k].split(), smoothing_function=smoothing)
                    scores.append(score)
            
            if scores:
                consistency_scores.append(np.mean(scores))
        except Exception as e:
            print(f"Error calculating consistency for image {img_file}: {str(e)}")
    
    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    print(f"Generating consistency: {avg_consistency:.3f}")
    
    proper_endings = sum(1 for item in generated_captions if item['caption'].strip().endswith('.'))
    grammar_score = proper_endings / len(generated_captions)
    print(f"Grammatical integrity: {grammar_score:.3f}")
    
    appropriate_length = sum(1 for item in generated_captions 
                           if 5 <= len(item['caption'].split()) <= 20)
    length_score = appropriate_length / len(generated_captions)
    print(f"Appropriateness of length: {length_score:.3f}")
    
    # Combine all metrics
    all_metrics = {
        'Average caption length': avg_length,
        'Lexical diversity': vocab_diversity,
        'Generating consistency': avg_consistency,
        'Grammatical integrity': grammar_score,
        'Appropriateness of length': length_score
    }
    
    # Add BLEU and CIDEr scores if available
    if metrics:
        all_metrics.update(metrics)
    
    # Calculate overall score
    original_score = (vocab_diversity + avg_consistency + grammar_score + length_score) / 4
    all_metrics['Original aggregate score'] = original_score
    
    # If we have BLEU and CIDEr, calculate a new aggregate score
    if 'BLEU-1' in metrics and 'CIDEr' in metrics:
        new_score = (metrics['BLEU-1'] + metrics['BLEU-4'] + metrics['CIDEr'] + grammar_score) / 4
        all_metrics['New aggregate score (with BLEU/CIDEr)'] = new_score
        print(f"New aggregate score (with BLEU/CIDEr): {new_score:.3f}")
    
    print(f"Original aggregate score: {original_score:.3f}")
    results = {
        'metrics': all_metrics,
        'generated_captions': generated_captions
    }
    
    output_file = 'test_evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to: {output_file}")
    
    return results