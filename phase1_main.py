#!/usr/bin/env python3
"""
Phase 1: Basic BiLSTM Implementation
RNA-Protein Binding Prediction Project

This script implements and tests a basic bidirectional LSTM model for predicting
RNA-protein binding intensities.
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
from src.models.lstm_basic import BasicLSTM
from src.training.trainer import RNAProteinTrainer
from src.training.evaluation import plot_predictions_vs_targets
from src.utils.helpers import create_run_directory, save_training_config, save_training_summary, save_comprehensive_run_report


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Basic BiLSTM for RNA-Protein Binding')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing data files')
    parser.add_argument('--subset_size', type=int, default=1000,
                       help='Number of RNA sequences to use for initial testing (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='LSTM hidden size (default: 64)')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of LSTM layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout probability (default: 0.2)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda (default: auto)')
    parser.add_argument('--save_model', type=str, default='phase1_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom name for this training run')
    parser.add_argument('--output_dir', type=str, default='runs',
                       help='Base directory for saving outputs (default: runs)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                       help='Minimum improvement for early stopping (default: 1e-4)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping (default: 1.0)')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                       help='Learning rate scheduler patience (default: 5)')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                       help='Learning rate reduction factor (default: 0.5)')
    parser.add_argument('--max_protein_length', type=int, default=300,
                       help='Maximum protein sequence length for memory efficiency (default: 300)')
    parser.add_argument('--max_rna_length', type=int, default=50,
                       help='Maximum RNA sequence length (default: 50)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if CUDA is available')
    
    args = parser.parse_args()
    
    # Create run directory
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"phase1_{timestamp}"
    
    run_dir = create_run_directory(args.output_dir, args.run_name)
    
    # Determine device
    if args.force_cpu:
        device = torch.device('cpu')
        print("⚠️ Forcing CPU usage as requested")
    elif args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Check GPU memory if using CUDA
    if device.type == 'cuda':
        try:
            torch.cuda.empty_cache()  # Clear cache
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated_memory
            print(f"🖥️ GPU Memory: {free_memory / 1024**3:.1f}GB free / {total_memory / 1024**3:.1f}GB total")
            
            # Warn if free memory is low
            if free_memory < 2 * 1024**3:  # Less than 2GB
                print("⚠️ Warning: Low GPU memory detected. Consider using --force_cpu or reducing --batch_size")
        except Exception as e:
            print(f"⚠️ Could not check GPU memory: {e}")
    
    print("="*60)
    print("RNA-Protein Binding Prediction - Phase 1: Basic BiLSTM")
    print("="*60)
    print(f"🏃 Run name: {args.run_name}")
    print(f"📁 Output directory: {run_dir}")
    print(f"💻 Device: {device}")
    print(f"📊 Subset size: {args.subset_size}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🧠 Hidden size: {args.hidden_size}")
    print(f"🏗️ Layers: {args.num_layers}")
    print(f"🛡️ Dropout: {args.dropout}")
    print(f"⏹️ Early stopping patience: {args.patience}")
    print(f"✂️ Gradient clipping: {args.max_grad_norm}")
    print(f"📏 Max protein length: {args.max_protein_length}")
    print(f"📏 Max RNA length: {args.max_rna_length}")
    print()
    
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
        'patience': args.patience,
        'min_delta': args.min_delta,
        'max_grad_norm': args.max_grad_norm,
        'lr_scheduler_patience': args.lr_scheduler_patience,
        'lr_scheduler_factor': args.lr_scheduler_factor,
        'max_protein_length': args.max_protein_length,
        'max_rna_length': args.max_rna_length,
        'force_cpu': args.force_cpu,
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
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    from src.data.preprocessing import find_optimal_sequence_lengths
    
    # Use custom max lengths if specified, otherwise use data-driven approach
    if args.max_protein_length and args.max_rna_length:
        rna_max_length = args.max_rna_length
        protein_max_length = args.max_protein_length
        print(f"Using custom sequence lengths: RNA={rna_max_length}, Protein={protein_max_length}")
    else:
        rna_max_length, protein_max_length = find_optimal_sequence_lengths(
            rna_sequences, protein_sequences
        )
        print(f"Using data-driven sequence lengths: RNA={rna_max_length}, Protein={protein_max_length}")
    
    train_loader, val_loader, _, _ = create_data_loaders(
        rna_sequences=rna_sequences,
        protein_sequences=protein_sequences,
        binding_scores=binding_scores,
        batch_size=args.batch_size,
        validation_split=0.2,
        rna_max_length=rna_max_length,
        protein_max_length=protein_max_length
    )
    print()
    
    # Create model
    print("Creating model...")
    
    # Calculate fusion hidden size based on hidden size for memory efficiency
    fusion_hidden_size = max(args.hidden_size * 2, 128)
    
    model = BasicLSTM(
        rna_input_size=5,      # A, U, G, C, N
        protein_input_size=21, # 20 amino acids + unknown
        rna_hidden_size=args.hidden_size,
        protein_hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        fusion_hidden_size=fusion_hidden_size
    )
    
    model_info = model.get_model_info()
    print(f"Model created: {model_info['model_name']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print()
    
    # Create trainer
    print("Setting up trainer...")
    trainer = RNAProteinTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        patience=args.patience,
        min_delta=args.min_delta,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        output_dir=run_dir
    )
    print()
    
    print("🎯 Training can be stopped anytime with Ctrl+C")
    print("📊 All outputs will be saved to:", run_dir)
    print("="*60)
    print()
    
    # Train model
    print("🚀 Starting training...")
    try:
        training_summary = trainer.train(
            num_epochs=args.epochs,
            save_path=args.save_model
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        training_summary = trainer.metrics_tracker.get_summary()
        training_summary.update({
            'interrupted': True,
            'total_epochs': len(trainer.metrics_tracker.train_losses),
            'model_info': model.get_model_info()
        })
    print()
    
    # Print training summary
    print("📊 Training Summary:")
    print("-" * 40)
    print(f"Best validation correlation: {training_summary['best_val_correlation']:.4f}")
    print(f"Best epoch: {training_summary['best_epoch'] + 1}")
    print(f"Total epochs trained: {training_summary.get('total_epochs', 'Unknown')}")
    if training_summary.get('interrupted', False):
        print("⚠️ Training was interrupted")
    if training_summary.get('early_stopped', False):
        print("🛑 Training stopped early")
    if 'final_train_correlation' in training_summary:
        print(f"Final training correlation: {training_summary['final_train_correlation']:.4f}")
        print(f"Final validation correlation: {training_summary['final_val_correlation']:.4f}")
    print(f"Total training time: {training_summary.get('total_training_time', 0):.1f}s")
    print()
    
    # Save training summary
    save_training_summary(training_summary, run_dir)
    
    # Plot training history
    print("📈 Plotting training history...")
    trainer.metrics_tracker.plot_history(output_dir=os.path.join(run_dir, 'plots'))
    
    # Evaluate on validation set and plot predictions
    print("🔍 Evaluating model...")
    from src.training.evaluation import evaluate_model
    
    val_metrics, val_predictions, val_targets = evaluate_model(
        trainer.model, val_loader, device
    )
    
    print("Validation Metrics:")
    print("-" * 20)
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
    
    # Plot predictions vs targets
    plot_predictions_vs_targets(
        val_targets, val_predictions,
        title="Phase 1: Basic BiLSTM - Validation Set",
        output_dir=os.path.join(run_dir, 'plots')
    )
    
    # Save comprehensive experiment report
    print("📋 Generating comprehensive experiment report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_comprehensive_run_report(
        config=config,
        model_info=model_info,
        training_summary=training_summary,
        validation_metrics=val_metrics,
        run_dir=run_dir,
        timestamp=timestamp
    )
    
    # Test model speed
    print("⚡ Testing inference speed...")
    import time
    
    trainer.model.eval()
    with torch.no_grad():
        # Get a batch for timing
        test_batch = next(iter(val_loader))
        rna = test_batch['rna'].to(device)
        protein = test_batch['protein'].to(device)
        
        # Warmup
        for _ in range(10):
            _ = trainer.model(rna, protein)
        
        # Time inference
        start_time = time.time()
        num_runs = 100
        for _ in range(num_runs):
            _ = trainer.model(rna, protein)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        samples_per_second = rna.size(0) / avg_time
        
    print(f"Average inference time: {avg_time*1000:.2f}ms per batch")
    print(f"Throughput: {samples_per_second:.1f} samples/second")
    print()
    
    print("🎉 Phase 1 completed successfully!")
    print(f"📁 All outputs saved to: {run_dir}")
    print(f"💾 Model saved to: {os.path.join(run_dir, 'models', args.save_model)}")
    print(f"📊 Plots saved to: {os.path.join(run_dir, 'plots')}")
    print(f"📋 Summary saved to: {os.path.join(run_dir, 'training_summary.json')}")
    print(f"📋 Comprehensive report saved to: {os.path.join(run_dir, 'configs')}")
    
    # Recommendations for next phase
    print()
    print("🚀 Recommendations for Phase 2:")
    print("- Current best validation correlation:", f"{training_summary['best_val_correlation']:.4f}")
    if training_summary['best_val_correlation'] > 0.3:
        print("✅ Good performance! Ready to add self-attention mechanism")
    elif training_summary['best_val_correlation'] > 0.2:
        print("🔶 Moderate performance. Consider tuning hyperparameters or adding attention")
    else:
        print("⚠️ Low performance. Consider:")
        print("   - Increasing model size (--hidden_size)")
        print("   - Using more training data (--subset_size)")
        print("   - Adjusting learning rate (--learning_rate)")
        print("   - Adding more regularization")
    
    print()
    print("🔧 Memory optimization tips:")
    print("- For CUDA memory issues: use --force_cpu or reduce --batch_size")
    print("- Reduce --max_protein_length to limit sequence processing")
    print("- Use smaller --hidden_size and --num_layers for less memory usage")
    print("- Current model parameters:", f"{model_info['total_parameters']:,}")


if __name__ == "__main__":
    main()
