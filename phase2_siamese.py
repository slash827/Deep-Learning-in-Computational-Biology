#!/usr/bin/env python3
"""
Phase 2 Siamese: RNA-Protein Binding Prediction with Siamese Contrastive Learning

This script implements Siamese contrastive learning for RNA-protein binding prediction,
building on the existing ProteinBERT embeddings and adding contrastive learning to
improve the model's ability to distinguish binding vs non-binding pairs.

Key Features:
- Siamese neural network architecture with shared weights
- Contrastive learning with positive/negative pair sampling
- Multiple loss functions: contrastive, InfoNCE, hybrid (contrastive + regression)
- Built on existing ProteinEmbeddingFusion backbone
- Support for both pure contrastive learning and hybrid training
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Any

from src.models.siamese_protein_bert import SiameseProteinBERT
from src.data.siamese_dataset import create_siamese_dataloaders
from src.training.siamese_trainer import SiameseTrainer


def create_config(args) -> Dict[str, Any]:
    """Create configuration dictionary from arguments."""
    
    config = {
        # Run configuration
        'run_name': f"siamese_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'device': 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu',
        
        # Data configuration
        'data_dir': args.data_dir,
        'protein_embedding_path': args.protein_embedding_path,
        'subset_size': args.subset_size,
        'max_rna_length': args.max_rna_length,
        'max_protein_length': args.max_protein_length,
        
        # Model configuration
        'rna_input_size': 5,
        'rna_hidden_size': args.rna_hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'protein_embedding_dim': args.protein_embedding_dim,
        'embedding_dim': args.embedding_dim,
        'temperature': args.temperature,
        
        # Training configuration
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'max_grad_norm': args.max_grad_norm,
        
        # Siamese learning configuration
        'loss_type': args.loss_type,
        'contrastive_weight': args.contrastive_weight,
        'regression_weight': args.regression_weight,
        'positive_threshold': args.positive_threshold,
        'negative_threshold': args.negative_threshold,
        'pair_sampling_ratio': args.pair_sampling_ratio,
        'hard_negative_ratio': args.hard_negative_ratio,
        
        # Data loading
        'num_workers': args.num_workers,
        'pin_memory': not args.force_cpu,
        
        # Optimization flags
        'force_cpu': args.force_cpu,
        'train_ratio': args.train_ratio,
        'split_strategy': args.split_strategy,
        
        # Timestamp
        'timestamp': datetime.now().isoformat()
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Siamese Contrastive Learning for RNA-Protein Binding')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='src/dataset', 
                       help='Directory containing training data')
    parser.add_argument('--protein_embedding_path', type=str, 
                       default='emb_cache/protein_bert.pt',
                       help='Path to cached protein embeddings')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Use subset of data for testing (default: use all data)')
    parser.add_argument('--max_rna_length', type=int, default=60,
                       help='Maximum RNA sequence length')
    parser.add_argument('--max_protein_length', type=int, default=300,
                       help='Maximum protein sequence length')
    
    # Model arguments
    parser.add_argument('--rna_hidden_size', type=int, default=128,
                       help='Hidden size for RNA LSTM')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--protein_embedding_dim', type=int, default=1024,
                       help='Protein embedding dimension (ProteinBERT)')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Final embedding dimension for contrastive learning')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature parameter for contrastive loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=8,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                       help='Minimum change for early stopping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Siamese learning arguments
    parser.add_argument('--loss_type', type=str, default='hybrid',
                       choices=['contrastive', 'infonce', 'hybrid'],
                       help='Type of loss function to use')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                       help='Weight for contrastive loss in hybrid training')
    parser.add_argument('--regression_weight', type=float, default=0.5,
                       help='Weight for regression loss in hybrid training')
    parser.add_argument('--positive_threshold', type=float, default=0.7,
                       help='Threshold for positive pairs (similarity)')
    parser.add_argument('--negative_threshold', type=float, default=0.3,
                       help='Threshold for negative pairs (dissimilarity)')
    parser.add_argument('--pair_sampling_ratio', type=float, default=1.5,
                       help='Ratio of pairs to generate relative to dataset size')
    parser.add_argument('--hard_negative_ratio', type=float, default=0.3,
                       help='Ratio of hard negatives to include')
    
    # System arguments
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if CUDA is available')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--split_strategy', type=str, default='realistic_test',
                       choices=['random', 'protein', 'rna', 'mixed', 'test_simulation', 'realistic_test'],
                       help='Strategy for train/validation split')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create configuration
    config = create_config(args)
    
    print("=" * 80)
    print("SIAMESE CONTRASTIVE LEARNING FOR RNA-PROTEIN BINDING PREDICTION")
    print("=" * 80)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Set device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Check if protein embeddings exist
    if not os.path.exists(config['protein_embedding_path']):
        print(f"ERROR: Protein embeddings not found at {config['protein_embedding_path']}")
        print("Please run the embedding caching script first:")
        print("python scripts/cache_proteinbert_embeddings.py")
        return
    
    try:
        # Create data loaders
        print("\nCreating Siamese data loaders...")
        train_loader, val_loader = create_siamese_dataloaders(
            data_dir=config['data_dir'],
            protein_embedding_path=config['protein_embedding_path'],
            subset_size=config['subset_size'],
            batch_size=config['batch_size'],
            max_rna_length=config['max_rna_length'],
            max_protein_length=config['max_protein_length'],
            train_ratio=config['train_ratio'],
            positive_threshold=config['positive_threshold'],
            negative_threshold=config['negative_threshold'],
            num_workers=config['num_workers'],
            split_strategy=config['split_strategy']
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Validation loader: {len(val_loader)} batches")
        
        # Create model
        print("\nCreating Siamese model...")
        model = SiameseProteinBERT(
            rna_input_size=config['rna_input_size'],
            rna_hidden_size=config['rna_hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            protein_embedding_dim=config['protein_embedding_dim'],
            embedding_dim=config['embedding_dim'],
            temperature=config['temperature']
        )
        
        model_info = model.get_model_info()
        print(f"Model created: {model_info}")
        
        # Create trainer
        print("\nInitializing trainer...")
        trainer = SiameseTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        # Train model
        print("\nStarting training...")
        training_summary = trainer.train()
        
        # Print final results
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Output directory: {trainer.output_dir}")
        print(f"Total training time: {training_summary['total_training_time']:.1f}s")
        print(f"Best epoch: {training_summary['best_epoch']}")
        print(f"Best validation loss: {training_summary['best_val_loss']:.6f}")
        
        if config['loss_type'] == 'hybrid':
            print(f"Best validation correlation: {training_summary['best_val_correlation']:.4f}")
        
        print("\nFinal evaluation metrics:")
        for metric, value in training_summary['final_metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

