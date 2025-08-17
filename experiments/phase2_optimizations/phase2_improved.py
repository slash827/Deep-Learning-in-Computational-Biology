#!/usr/bin/env python3
"""
Phase 2 - High Accuracy RNA-Protein Binding Prediction with Bug Fixes
Improved version with better Ctrl+C handling and anti-stalling measures.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.dataset import RNAProteinDataset
from src.data.preprocessing import create_vocab_and_encoders
from src.models.lstm_attention_fast import FastAttentionLSTM
from src.training.trainer_improved import ImprovedRNAProteinTrainer  # Use improved trainer
from src.utils.helpers import set_random_seed, create_run_directory, save_experiment_config


def load_data(data_dir="data"):
    """Load training and test data with comprehensive error handling."""
    try:
        # Load training data
        train_rna_file = os.path.join(data_dir, "training_seqs.txt")
        train_protein_file = os.path.join(data_dir, "training_RBPs2.txt")
        train_scores_file = os.path.join(data_dir, "training_data2.txt")
        
        # Load test data
        test_rna_file = os.path.join(data_dir, "test_seqs.txt")
        test_protein_file = os.path.join(data_dir, "test_RBPs2.txt")
        test_scores_file = os.path.join(data_dir, "test_RBPs2.txt")
        
        # Check all files exist
        for file_path in [train_rna_file, train_protein_file, train_scores_file, 
                         test_rna_file, test_protein_file, test_scores_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        print(f"Loading training data from {data_dir}...")
        
        # Read training RNA sequences
        with open(train_rna_file, 'r', encoding='utf-8', errors='replace') as f:
            train_rna_sequences = [line.strip() for line in f if line.strip()]
        
        # Read training protein sequences
        with open(train_protein_file, 'r', encoding='utf-8', errors='replace') as f:
            train_protein_sequences = [line.strip() for line in f if line.strip()]
        
        # Read training scores
        with open(train_scores_file, 'r', encoding='utf-8', errors='replace') as f:
            train_scores = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        score = float(line)
                        train_scores.append(score)
                    except ValueError:
                        print(f"Warning: Could not parse score: {line}")
        
        # Read test data
        print(f"Loading test data from {data_dir}...")
        
        with open(test_rna_file, 'r', encoding='utf-8', errors='replace') as f:
            test_rna_sequences = [line.strip() for line in f if line.strip()]
        
        with open(test_protein_file, 'r', encoding='utf-8', errors='replace') as f:
            test_protein_sequences = [line.strip() for line in f if line.strip()]
        
        # For test scores, we'll use dummy values or load from appropriate file
        test_scores = [0.0] * len(test_rna_sequences)  # Dummy scores for testing
        
        # Validate data consistency
        print(f"Training data: {len(train_rna_sequences)} RNA, {len(train_protein_sequences)} proteins, {len(train_scores)} scores")
        print(f"Test data: {len(test_rna_sequences)} RNA, {len(test_protein_sequences)} proteins")
        
        if not (len(train_rna_sequences) == len(train_protein_sequences) == len(train_scores)):
            raise ValueError(f"Training data size mismatch: RNA={len(train_rna_sequences)}, Protein={len(train_protein_sequences)}, Scores={len(train_scores)}")
        
        return {
            'train_rna': train_rna_sequences,
            'train_protein': train_protein_sequences,
            'train_scores': train_scores,
            'test_rna': test_rna_sequences,
            'test_protein': test_protein_sequences,
            'test_scores': test_scores
        }
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure the following files exist in '{data_dir}':")
        print("  - training_seqs.txt")
        print("  - training_RBPs2.txt") 
        print("  - training_data2.txt")
        print("  - test_seqs.txt")
        print("  - test_RBPs2.txt")
        raise


def create_model(vocab_sizes, config):
    """Create model with the specified configuration."""
    model = FastAttentionLSTM(
        rna_vocab_size=vocab_sizes['rna_vocab_size'],
        protein_vocab_size=vocab_sizes['protein_vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_attention_heads=config['num_attention_heads'],
        attention_dropout=config['attention_dropout'],
        mixed_precision=config['mixed_precision']
    )
    return model


def main():
    """Main training function with improved error handling and bug fixes."""
    parser = argparse.ArgumentParser(description='Phase 2 RNA-Protein Binding Prediction (Bug Fixed)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--max_seq_length', type=int, default=1000, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create run directory
    run_dir = create_run_directory("phase2_improved")
    print(f"Run directory: {run_dir}")
    
    # Determine device
    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Load data
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        data = load_data(args.data_dir)
        
        # Create vocabularies and encoders
        print("Creating vocabularies...")
        vocab_info = create_vocab_and_encoders(
            data['train_rna'], 
            data['train_protein'],
            max_seq_length=args.max_seq_length
        )
        
        print(f"RNA vocabulary size: {vocab_info['rna_vocab_size']}")
        print(f"Protein vocabulary size: {vocab_info['protein_vocab_size']}")
        
        # Create datasets
        print("Creating datasets...")
        
        # Full training dataset for splitting
        full_dataset = RNAProteinDataset(
            data['train_rna'],
            data['train_protein'], 
            data['train_scores'],
            vocab_info['rna_vocab'],
            vocab_info['protein_vocab'],
            max_seq_length=args.max_seq_length
        )
        
        # Split into train and validation
        train_size = int((1 - args.validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders with improved settings for Windows
        num_workers = 0 if os.name == 'nt' else 2  # Windows compatibility
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=False  # Better for Windows
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=False
        )
        
        # Model configuration
        config = {
            'embedding_dim': args.embedding_dim,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'num_attention_heads': args.num_attention_heads,
            'attention_dropout': args.attention_dropout,
            'mixed_precision': args.mixed_precision and device.type == 'cuda',
            'max_seq_length': args.max_seq_length
        }
        
        # Disable mixed precision for better stability if requested
        if not args.mixed_precision:
            config['mixed_precision'] = False
            print("🔒 Mixed precision disabled for stability")
        
        print("="*60)
        print("CREATING MODEL")
        print("="*60)
        
        # Create model
        model = create_model(vocab_info, config)
        
        # Print model info
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            print(f"Model: {model.__class__.__name__}")
            print(f"Parameters: {model_info['total_parameters']:,}")
            print(f"Memory estimate: {model_info['memory_estimate_mb']:.1f} MB")
        
        print("="*60)
        print("STARTING IMPROVED TRAINING")
        print("="*60)
        
        # Create improved trainer
        trainer = ImprovedRNAProteinTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.learning_rate,
            patience=args.patience,
            output_dir=run_dir,
            warmup_epochs=3  # Add warmup for better training stability
        )
        
        # Save configuration
        full_config = {
            **vars(args),
            **config,
            **vocab_info,
            'model_class': model.__class__.__name__,
            'device': str(device),
            'run_directory': run_dir,
            'improvements': [
                'improved_ctrl_c_handling',
                'anti_stalling_measures', 
                'learning_rate_warmup',
                'better_memory_management',
                'enhanced_error_handling'
            ]
        }
        
        config_path = os.path.join(run_dir, 'config.json')
        save_experiment_config(full_config, config_path)
        
        # Train model
        model_save_path = os.path.join(run_dir, 'models', 'best_model.pth')
        
        training_summary = trainer.train(
            num_epochs=args.epochs,
            save_path=model_save_path
        )
        
        # Save training summary
        summary_path = os.path.join(run_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            clean_summary = {}
            for k, v in training_summary.items():
                if isinstance(v, np.floating):
                    clean_summary[k] = float(v)
                elif isinstance(v, np.integer):
                    clean_summary[k] = int(v)
                else:
                    clean_summary[k] = v
            json.dump(clean_summary, f, indent=2)
        
        print(f"\n✅ Training completed!")
        print(f"📁 Results saved to: {run_dir}")
        print(f"📊 Best validation correlation: {training_summary['best_val_correlation']:.4f}")
        
        # Test the model if not interrupted
        if not training_summary.get('interrupted', False) and not training_summary.get('force_stopped', False):
            print("\n" + "="*60)
            print("TESTING MODEL")
            print("="*60)
            
            # Load best model for testing
            if os.path.exists(model_save_path):
                checkpoint = torch.load(model_save_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
            
            # Create test dataset if we have proper test scores
            if len(set(data['test_scores'])) > 1:  # More than just dummy values
                test_dataset = RNAProteinDataset(
                    data['test_rna'],
                    data['test_protein'],
                    data['test_scores'],
                    vocab_info['rna_vocab'],
                    vocab_info['protein_vocab'],
                    max_seq_length=args.max_seq_length
                )
                
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device.type == 'cuda'
                )
                
                # Evaluate on test set
                from src.training.evaluation_lightweight import evaluate_model
                test_metrics, predictions, targets = evaluate_model(model, test_loader, device)
                
                print(f"Test Results:")
                print(f"  MSE: {test_metrics['mse']:.4f}")
                print(f"  Pearson Correlation: {test_metrics['pearson_correlation']:.4f}")
                print(f"  Spearman Correlation: {test_metrics['spearman_correlation']:.4f}")
                print(f"  R²: {test_metrics['r2_score']:.4f}")
                
                # Add test results to summary
                training_summary['test_metrics'] = test_metrics
                
                # Re-save updated summary
                with open(summary_path, 'w') as f:
                    clean_summary = {}
                    for k, v in training_summary.items():
                        if isinstance(v, np.floating):
                            clean_summary[k] = float(v)
                        elif isinstance(v, np.integer):
                            clean_summary[k] = int(v)
                        else:
                            clean_summary[k] = v
                    json.dump(clean_summary, f, indent=2)
            else:
                print("Skipping test evaluation (no proper test scores available)")
        else:
            print(f"\n⚠️  Training was interrupted/stopped - skipping test evaluation")
        
        return training_summary
        
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
