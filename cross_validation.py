import os
import sys
import json
import torch
import numpy as np
import argparse
from itertools import product
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from pathlib import Path

# Import your existing modules
# assuming modified_train_script.py is in the same directory
from modified_train_script import (
    DatabaseClipDataset, ClipCaptionModel, ClipCaptionPrefix, 
    MappingType, noise_injection, train
)

def parse_args():
    parser = argparse.ArgumentParser(description='Cross-validation for CLIP-GPT model')
    
    # Database paths
    parser.add_argument('--db_path', default='coco_image_title_data/image_title_database.db', 
                        help='path to SQLite database with CLIP embeddings')
    parser.add_argument('--table_name', default='image_features_clip', 
                        help='table name in database containing CLIP embeddings')
    parser.add_argument('--val_db_path', default='coco_image_title_data/image_title_database.db', 
                        help='path to validation database')
    parser.add_argument('--val_table_name', default='image_features_clip_V', 
                        help='validation table name')
    
    # Hyperparameter search space
    parser.add_argument('--epochs_list', nargs='+', type=int, default=[10,15,25,30], 
                        help='list of epoch values to try')
    parser.add_argument('--num_layers_list', nargs='+', type=int, default=[4,8,12], 
                        help='list of layer numbers to try')
    parser.add_argument('--lr_list', nargs='+', type=float, default=[1e-6,1e-5,5e-5,5e-4,5e-3], 
                        help='list of learning rates to try')
    parser.add_argument('--bs_list', nargs='+', type=int, default=[5,15,30,60], 
                        help='list of batch sizes to try')
    
    # Other model parameters (fixed)
    parser.add_argument('--prefix_length', type=int, default=40, 
                        help='prefix length')
    parser.add_argument('--prefix_length_clip', type=int, default=40, 
                        help='prefix length for clip')
    parser.add_argument('--noise_variance', type=float, default=0.0, 
                        help='noise variance')
    parser.add_argument('--uniform_noise', action='store_true', default=False, 
                        help='use uniform noise instead of gaussian')
    parser.add_argument('--dont_norm', action='store_true', default=False, 
                        help='dont normalize CLIP embeddings')
    parser.add_argument('--mapping_type', type=str, default='transformer', 
                        help='type of architecture between CLIP and GPT (mlp/transformer)')
    parser.add_argument('--only_prefix', action='store_true', default=True, 
                        help='train only the mapper between CLIP and GPT, while GPT is frozen')
    parser.add_argument('--is_not_rn', action='store_true', default=False, 
                        help='Choose the CLIP backbone: False for RN, True for ViT')
    parser.add_argument('--add_modality_offset', action='store_true', default=False, 
                        help='train with modality offset')
    
    # Output directory
    parser.add_argument('--out_dir', default='./cv_checkpoints', 
                        help='path to output directory for cross validation')
    parser.add_argument('--final_model_path', default='./best_model.pt', 
                        help='path to save the best model')
    
    # Maximum number of combinations to try
    parser.add_argument('--max_combinations', type=int, default=10, 
                        help='maximum number of hyperparameter combinations to try')
    
    return parser.parse_args()

def evaluate_model(model, val_dataset, batch_size, device):
    """Evaluate the model on the validation dataset and return the validation loss."""
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for tokens, mask, prefix in val_dataloader:
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, val_dataset.prefix_length - 1: -1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), 
                tokens.flatten(), 
                ignore_index=0
            )
            total_loss += loss.item() * tokens.size(0)
    
    return total_loss / len(val_dataset)

def main():
    args = parse_args()
    
    # Create the output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Determine the feature dimension
    prefix_dim = 640 if not args.is_not_rn else 512
    
    # Set the mapping type
    mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    
    # Load validation dataset
    print(f"Loading validation dataset from {args.val_db_path}, table: {args.val_table_name}")
    val_dataset = DatabaseClipDataset(
        args.val_db_path, 
        args.val_table_name, 
        args.prefix_length, 
        normalize_prefix=not args.dont_norm
    )
    
    # Generate hyperparameter combinations
    param_grid = list(product(
        args.epochs_list,
        args.num_layers_list,
        args.lr_list,
        args.bs_list
    ))
    
    # Limit the number of combinations if needed
    if len(param_grid) > args.max_combinations:
        print(f"Limiting to {args.max_combinations} random combinations out of {len(param_grid)}")
        np.random.shuffle(param_grid)
        param_grid = param_grid[:args.max_combinations]
    
    # Dictionary to store results
    results = []
    best_val_loss = float('inf')
    best_params = None
    best_model_path = None
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Cross-validation loop
    for i, (epochs, num_layers, lr, bs) in enumerate(param_grid):
        print(f"\n===== Combination {i+1}/{len(param_grid)} =====")
        print(f"Epochs: {epochs}, Layers: {num_layers}, LR: {lr}, Batch Size: {bs}")
        
        # Create a unique prefix for this configuration
        run_prefix = f"cv_e{epochs}_l{num_layers}_lr{lr}_bs{bs}"
        run_dir = os.path.join(args.out_dir, run_prefix)
        os.makedirs(run_dir, exist_ok=True)
        
        # Create modified args for this run
        run_args = argparse.Namespace(
            db_path=args.db_path,
            table_name=args.table_name,
            val_db_path=args.val_db_path,
            val_table_name=args.val_table_name,
            prefix=run_prefix,
            noise_variance=args.noise_variance,
            uniform_noise=args.uniform_noise,
            dont_norm=args.dont_norm,
            lr=lr,
            epochs=epochs,
            save_every=epochs,  # Save only at the end
            prefix_length=args.prefix_length,
            prefix_length_clip=args.prefix_length_clip,
            bs=bs,
            only_prefix=args.only_prefix,
            mapping_type=mapping_type,
            num_layers=num_layers,
            is_not_rn=args.is_not_rn,
            add_modality_offset=args.add_modality_offset,
            out_dir=run_dir
        )
        
        # Create and initialize the model
        if args.only_prefix:
            model = ClipCaptionPrefix(
                args.prefix_length, 
                clip_length=args.prefix_length_clip, 
                prefix_size=prefix_dim,
                num_layers=num_layers, 
                mapping_type=mapping_type
            )
            print("Training only the prefix mapper")
        else:
            model = ClipCaptionModel(
                args.prefix_length, 
                clip_length=args.prefix_length_clip, 
                prefix_size=prefix_dim,
                num_layers=num_layers, 
                mapping_type=mapping_type
            )
            print("Training both prefix mapper and GPT")
        
        # Load the training dataset
        print(f"Loading training dataset from {args.db_path}, table: {args.table_name}")
        train_dataset = DatabaseClipDataset(
            args.db_path, 
            args.table_name, 
            args.prefix_length, 
            normalize_prefix=not args.dont_norm
        )
        
        # Train the model
        try:
            print(f"Starting training with {len(train_dataset)} samples")
            train(train_dataset, model, run_args, output_dir=run_dir, output_prefix=run_prefix)
            
            # Load the trained model for evaluation
            model_path = os.path.join(run_dir, f"{run_prefix}_latest.pt")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            
            # Evaluate on validation set
            val_loss = evaluate_model(model, val_dataset, bs, device)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Record results
            result = {
                'epochs': epochs,
                'num_layers': num_layers,
                'learning_rate': lr,
                'batch_size': bs,
                'validation_loss': val_loss,
                'model_path': model_path
            }
            results.append(result)
            
            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = result
                best_model_path = model_path
                
            # Save results after each run
            with open(os.path.join(args.out_dir, 'cv_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error during training: {e}")
            continue
    
    # Print summary of results
    print("\n===== Cross-Validation Results =====")
    results.sort(key=lambda x: x['validation_loss'])
    
    for i, result in enumerate(results):
        print(f"{i+1}. Loss: {result['validation_loss']:.4f}, Epochs: {result['epochs']}, "
              f"Layers: {result['num_layers']}, LR: {result['learning_rate']}, BS: {result['batch_size']}")
    
    # Save the best model
    if best_model_path:
        print(f"\nBest hyperparameters: {best_params}")
        print(f"Copying best model to {args.final_model_path}")
        shutil.copy(best_model_path, args.final_model_path)
        
        # Also save the best hyperparameters
        with open(os.path.join(Path(args.final_model_path).parent, 'best_hyperparams.json'), 'w') as f:
            json.dump(best_params, f, indent=2)
    else:
        print("No valid model was found during cross-validation")

if __name__ == "__main__":
    main()
