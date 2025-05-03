import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import skimage.io as io
import PIL.Image
from PIL import Image
from enum import Enum
import argparse
import glob
import json


N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device("cpu")
from mapping import (
    MappingType,
    MLP,
    MlpTransformer,
    TransformerLayer,
    MultiHeadAttention,
    Transformer,
    TransformerMapper,
    ClipCaptionModel,
    generate2
)

class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
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
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=30,  
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
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


def main():
    parser = argparse.ArgumentParser(description='Generate captions for images')
    parser.add_argument('--img_dir', type=str, default='./test_images', 
                       help='Test Image Catalogue (default: ./test_images)')
    parser.add_argument('--weights', type=str, default='./checkpoints/clip_pro_prefix-002.pt',
                       help='Model weights file path')
    parser.add_argument('--prefix_length', type=int, default=40,
                       help='prefix length (default: 40)')
    parser.add_argument('--use_beam_search', action='store_true',
                       help='Generate using beam search')
    parser.add_argument('--beam_size', type=int, default=15,
                       help='beam_size ')
    parser.add_argument('--temperature', type=float, default=1.6,
                       help='temperature ')
    parser.add_argument('--entry_length', type=int, default=20,
                       help='entry_length ')
    parser.add_argument('--output_file', type=str, default='./generated_captions.json',
                       help='Output file path')
    parser.add_argument('--clip_model', type=str, default='RN50x4',
                       help='CLIP model type ')
    parser.add_argument('--mapping_type', type=str, default='transformer', 
                       help='Mapping Type (mlp/transformer) (default: mlp)')
    parser.add_argument('--use_cpu', action='store_true',
                       help='Forced CPU usage')
    parser.add_argument('--normalize_prefix', action='store_true',
                   help='Normalize prefix embeddings (should match training)')

    args = parser.parse_args()
    
    if args.use_cpu:
        device = torch.device('cpu')
        print("Reasoning with the CPU")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Use device: {device}")

    print(f"Loading the CLIP model: {args.clip_model}")
    clip_model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    

    print(f"Loading: {args.weights}")
    mapping_type = MappingType.MLP if args.mapping_type == 'mlp' else MappingType.Transformer
    prefix_length = args.prefix_length
    clip_length = 40
    model = ClipCaptionModel(prefix_length=prefix_length, 
                            clip_length=clip_length,
                            prefix_size=640,  
                            mapping_type=mapping_type)

    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model = model.eval()
    model = model.to(device)
    print("Loading Model Successfully")

    img_files = glob.glob(os.path.join(args.img_dir, "*.jpg")) + \
                glob.glob(os.path.join(args.img_dir, "*.jpeg")) + \
                glob.glob(os.path.join(args.img_dir, "*.png"))
    
    print(f"Find {len(img_files)} images")

    results = {}

    for img_file in img_files:
        print(f"Processing: {img_file}")
        try:

            image = Image.open(img_file).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image_input).to(device, dtype=torch.float32)
                prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
            if args.normalize_prefix: 
                prefix = prefix.float() 
                prefix = prefix / prefix.norm(2, -1)

            if args.use_beam_search:
                generated_text = generate_beam(model, tokenizer, 
                                             embed=prefix_embed,
                                             beam_size=args.beam_size,
                                             entry_length=args.entry_length,
                                             temperature=args.temperature)[0]
            else:
                generated_text = generate2(model, tokenizer, 
                                         embed=prefix_embed,
                                         entry_length=args.entry_length,
                                         temperature=args.temperature)
            filename = os.path.basename(img_file)
            results[filename] = generated_text
            print(f"Generated Caption: {generated_text}")
            
        except Exception as e:
            print(f"Error {img_file} : {str(e)}")
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Save captions to: {args.output_file}")


if __name__ == "__main__":
    main()