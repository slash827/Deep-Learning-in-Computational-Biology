#!/usr/bin/env python3
"""
Simple Phase 2 Main - No Unicode for Batch Processing
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from src.data.dataset import load_training_data, create_data_loaders
from src.models.lstm_attention import AttentionLSTM
from src.training.trainer import RNAProteinTrainer
from src.training.evaluation import plot_predictions_vs_targets
from src.utils.helpers import create_run_directory, save_training_config, save_training_summary, save_comprehensive_run_report


def main():
    parser = argparse.ArgumentParser(description='Phase 2: BiLSTM + Attention for RNA-Protein Binding')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing data files')
    parser.add_argument('--subset_size', type=int, default=100,
                       help='Number of RNA sequences to use (default: 100)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM hidden size (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability (default: 0.3)')
    parser.add_argument('--attention_heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                       help='Attention dropout probability (default: 0.1)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda (default: auto)')
    parser.add_argument('--save_model', type=str, default='phase2_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom name for this training run')
    parser.add_argument('--output_dir', type=str, default='runs',
                       help='Base directory for saving outputs (default: runs)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping (default: 1.0)')
    parser.add_argument('--max_protein_length', type=int, default=200,
                       help='Maximum protein sequence length (default: 200)')
    parser.add_argument('--max_rna_length', type=int, default=50,
                       help='Maximum RNA sequence length (default: 50)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if CUDA is available')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Number of warmup epochs (default: 0)')
    parser.add_argument('--use_positional_encoding', action='store_true',
                       help='Use positional encoding in attention')
    
    args = parser.parse_args()
    
    # Create run directory
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"phase2_{timestamp}"
    
    run_dir = create_run_directory(args.output_dir, args.run_name)
    
    # Determine device
    if args.force_cpu:
        device = torch.device('cpu')
    elif args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    print(f"Run name: {args.run_name}")
    print(f"Subset size: {args.subset_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Attention heads: {args.attention_heads}")
    
    # Save configuration
    config = {
        'run_name': args.run_name,
        'device': str(device),
        'subset_size': args.subset_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'attention_heads': args.attention_heads,
        'attention_dropout': args.attention_dropout,
        'patience': args.patience,
        'max_grad_norm': args.max_grad_norm,
        'max_protein_length': args.max_protein_length,
        'max_rna_length': args.max_rna_length,
        'force_cpu': args.force_cpu,
        'warmup_epochs': args.warmup_epochs,
        'use_positional_encoding': args.use_positional_encoding,
        'timestamp': datetime.now().isoformat()
    }
    save_training_config(config, run_dir)
    
    # Load training data
    print("Loading training data...")
    rna_sequences, protein_sequences, binding_scores = load_training_data(
        args.data_dir, subset_size=args.subset_size
    )
    
    print(f"Loaded {len(rna_sequences)} RNA-protein pairs")
    print(f"Binding scores range: {binding_scores.min():.3f} - {binding_scores.max():.3f}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, _, _ = create_data_loaders(
        rna_sequences=rna_sequences,
        protein_sequences=protein_sequences,
        binding_scores=binding_scores,
        batch_size=args.batch_size,
        validation_split=0.2,
        rna_max_length=args.max_rna_length,
        protein_max_length=args.max_protein_length
    )
    
    # Create model
    print("Creating Phase 2 model with self-attention...")
    model = AttentionLSTM(
        rna_input_size=5,      # A, U, G, C, N
        protein_input_size=21, # 20 amino acids + unknown
        rna_hidden_size=args.hidden_size,
        protein_hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        attention_heads=args.attention_heads,
        attention_dropout=args.attention_dropout,
        use_positional_encoding=args.use_positional_encoding
    )
    
    model_info = model.get_model_info()
    print(f"Model created: {model_info['model_name']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    
    # Create trainer
    print("Setting up enhanced trainer for Phase 2...")
    trainer = RNAProteinTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        patience=args.patience,
        min_delta=1e-4,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_patience=7,
        lr_scheduler_factor=0.5,
        output_dir=run_dir,
        warmup_epochs=args.warmup_epochs
    )
    
    # Train model
    print("Starting Phase 2 training...")
    try:
        training_summary = trainer.train(
            num_epochs=args.epochs,
            save_path=args.save_model
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
        training_summary = trainer.metrics_tracker.get_summary()
        training_summary.update({
            'interrupted': True,
            'total_epochs': len(trainer.metrics_tracker.train_losses),
            'model_info': model.get_model_info()
        })
    
    # Print training summary
    print("Phase 2 Training Summary:")
    print("-" * 40)
    print(f"Best validation correlation: {training_summary['best_val_correlation']:.4f}")
    print(f"Best epoch: {training_summary['best_epoch'] + 1}")
    print(f"Total epochs trained: {training_summary.get('total_epochs', 'Unknown')}")
    if 'final_train_correlation' in training_summary:
        print(f"Final training correlation: {training_summary['final_train_correlation']:.4f}")
        print(f"Final validation correlation: {training_summary['final_val_correlation']:.4f}")
    print(f"Total training time: {training_summary.get('total_training_time', 0):.1f}s")
    
    # Save training summary
    save_training_summary(training_summary, run_dir)
    
    # Plot training history
    print("Plotting training history...")
    trainer.metrics_tracker.plot_history(output_dir=os.path.join(run_dir, 'plots'))
    
    # Evaluate on validation set
    print("Evaluating Phase 2 model...")
    from src.training.evaluation import evaluate_model
    
    val_metrics, val_predictions, val_targets = evaluate_model(
        trainer.model, val_loader, device
    )
    
    print("Validation Metrics:")
    print("-" * 20)
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot predictions vs targets
    plot_predictions_vs_targets(
        val_targets, val_predictions,
        title="Phase 2: BiLSTM + Attention - Validation Set",
        output_dir=os.path.join(run_dir, 'plots')
    )
    
    # Save comprehensive experiment report
    print("Generating comprehensive Phase 2 experiment report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_comprehensive_run_report(
        config=config,
        model_info=model_info,
        training_summary=training_summary,
        validation_metrics=val_metrics,
        run_dir=run_dir,
        timestamp=timestamp,
        phase=2
    )
    
    print("Phase 2 completed successfully!")
    print(f"All outputs saved to: {run_dir}")
    print(f"Best validation correlation: {training_summary['best_val_correlation']:.4f}")

if __name__ == "__main__":
    main()
