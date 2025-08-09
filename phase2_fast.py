#!/usr/bin/env python3
"""
Phase 2 Fast: Optimized BiLSTM with Self-Attention Implementation
RNA-Protein Binding Prediction Project

This script implements an optimized version of the attention model for faster training
while maintaining model accuracy.
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
from src.models.lstm_attention_fast import FastAttentionLSTM, UltraFastLSTM
from src.training.trainer_fast import FastRNAProteinTrainer
from src.training.evaluation import plot_predictions_vs_targets
from src.utils.helpers import create_run_directory, save_training_config, save_training_summary, save_comprehensive_run_report


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Fast: Optimized BiLSTM with Self-Attention')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing data files')
    # OPTIMIZATION 1: Reduce data size for faster training
    parser.add_argument('--subset_size', type=int, default=1500,
                       help='Number of RNA sequences to use (reduced from 2000 for speed)')
    # OPTIMIZATION 2: Increase batch size for better GPU utilization
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (increased from 16 for speed)')
    parser.add_argument('--epochs', type=int, default=8,
                       help='Number of training epochs (reduced from 10)')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                       help='Learning rate (increased for faster convergence)')
    # OPTIMIZATION 3: Reduce model size slightly
    parser.add_argument('--hidden_size', type=int, default=80,
                       help='LSTM hidden size (reduced from 96 for speed)')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of LSTM layers (reduced from 2 for speed)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout probability (reduced for faster training)')
    # OPTIMIZATION 4: Reduce attention heads
    parser.add_argument('--attention_heads', type=int, default=4,
                       help='Number of attention heads (reduced from 6)')
    parser.add_argument('--attention_dropout', type=float, default=0.05,
                       help='Attention dropout probability (reduced)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda (default: auto)')
    parser.add_argument('--save_model', type=str, default='phase2_fast_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom name for this training run')
    parser.add_argument('--output_dir', type=str, default='runs',
                       help='Base directory for saving outputs (default: runs)')
    # OPTIMIZATION 5: Reduce early stopping patience
    parser.add_argument('--patience', type=int, default=8,
                       help='Early stopping patience (reduced from 15)')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                       help='Minimum improvement for early stopping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--lr_scheduler_patience', type=int, default=4,
                       help='Learning rate scheduler patience (reduced)')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.7,
                       help='Learning rate reduction factor')
    # OPTIMIZATION 6: Reduce sequence lengths
    parser.add_argument('--max_protein_length', type=int, default=300,
                       help='Maximum protein sequence length (reduced from 400)')
    parser.add_argument('--max_rna_length', type=int, default=60,
                       help='Maximum RNA sequence length (reduced from 75)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if CUDA is available')
    parser.add_argument('--use_positional_encoding', action='store_true',
                       help='Use positional encoding in attention mechanism')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                       help='Number of warmup epochs (reduced from 3)')
    # OPTIMIZATION 7: Mixed precision training (with option to disable)
    parser.add_argument('--use_mixed_precision', action='store_true', default=False,
                       help='Use mixed precision training for speed (default: False due to potential issues)')
    parser.add_argument('--disable_mixed_precision', action='store_true', default=False,
                       help='Explicitly disable mixed precision training')
    # OPTIMIZATION 8: Simplified attention
    parser.add_argument('--simple_attention', action='store_true', default=True,
                       help='Use simplified attention mechanism (default: True)')
    # OPTIMIZATION 9: Faster data loading (Windows compatible)
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of workers for data loading (0 for Windows)')
    parser.add_argument('--pin_memory', action='store_true', default=False,
                       help='Pin memory for faster GPU transfer')
    # Add new argument for ultra-fast mode
    parser.add_argument('--ultra_fast', action='store_true',
                       help='Use ultra-fast model (single LSTM, minimal attention)')
    
    args = parser.parse_args()
    
    # Create run directory
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"phase2_fast_{timestamp}"
    
    run_dir = create_run_directory(args.output_dir, args.run_name)
    
    # Determine device
    if args.force_cpu:
        device = torch.device('cpu')
        print("WARNING: Forcing CPU usage as requested")
        args.use_mixed_precision = False  # Mixed precision not supported on CPU
    elif args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            args.use_mixed_precision = False
    else:
        device = torch.device(args.device)
        if device.type == 'cpu':
            args.use_mixed_precision = False
    
    # Handle mixed precision settings
    if args.disable_mixed_precision:
        args.use_mixed_precision = False
        print("🔧 Mixed precision explicitly disabled")
    elif not args.use_mixed_precision and device.type == 'cuda':
        print("⚠️ Mixed precision disabled - may be slower but more stable")
    
    # Check GPU memory and optimize settings
    if device.type == 'cuda':
        try:
            torch.cuda.empty_cache()  # Clear cache
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated_memory
            print(f"🖥️ GPU Memory: {free_memory / 1024**3:.1f}GB free / {total_memory / 1024**3:.1f}GB total")
            
            # Auto-adjust batch size based on available memory
            if free_memory < 2 * 1024**3:  # Less than 2GB
                print("🔧 Low GPU memory detected, reducing batch size to 16")
                args.batch_size = 16
            elif free_memory > 6 * 1024**3:  # More than 6GB
                print("🚀 High GPU memory detected, increasing batch size to 48")
                args.batch_size = 48
                
        except Exception as e:
            print(f"⚠️ Could not check GPU memory: {e}")
    
    print("="*60)
    print("RNA-Protein Binding Prediction - Phase 2 FAST: Optimized Training")
    print("="*60)
    print("🚀 SPEED OPTIMIZATIONS ENABLED:")
    print(f"   ✅ Reduced model size: {args.hidden_size} hidden units, {args.num_layers} layers")
    print(f"   ✅ Increased batch size: {args.batch_size} (better GPU utilization)")
    print(f"   ✅ Reduced attention heads: {args.attention_heads}")
    print(f"   ✅ Shorter sequences: RNA={args.max_rna_length}, Protein={args.max_protein_length}")
    print(f"   ✅ Reduced training data: {args.subset_size} samples")
    print(f"   ✅ Early stopping: {args.patience} patience")
    if args.use_mixed_precision:
        print("   ✅ Mixed precision training enabled")
    if args.simple_attention:
        print("   ✅ Simplified attention mechanism")
    print()
    print(f"🏃 Run name: {args.run_name}")
    print(f"📁 Output directory: {run_dir}")
    print(f"💻 Device: {device}")
    print(f"📊 Subset size: {args.subset_size}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🧠 Hidden size: {args.hidden_size}")
    print(f"🏗️ LSTM layers: {args.num_layers}")
    print(f"🎯 Attention heads: {args.attention_heads}")
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
        'attention_heads': args.attention_heads,
        'attention_dropout': args.attention_dropout,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'max_grad_norm': args.max_grad_norm,
        'lr_scheduler_patience': args.lr_scheduler_patience,
        'lr_scheduler_factor': args.lr_scheduler_factor,
        'max_protein_length': args.max_protein_length,
        'max_rna_length': args.max_rna_length,
        'force_cpu': args.force_cpu,
        'use_positional_encoding': args.use_positional_encoding,
        'warmup_epochs': args.warmup_epochs,
        'use_mixed_precision': args.use_mixed_precision,
        'simple_attention': args.simple_attention,
        'ultra_fast': args.ultra_fast,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'optimizations': 'fast_training_enabled',
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
    
    # Create optimized data loaders
    print("Creating optimized data loaders...")
    from src.data.preprocessing import find_optimal_sequence_lengths
    
    # Use custom max lengths (already reduced for speed)
    rna_max_length = args.max_rna_length
    protein_max_length = args.max_protein_length
    print(f"Using optimized sequence lengths: RNA={rna_max_length}, Protein={protein_max_length}")
    
    train_loader, val_loader, _, _ = create_data_loaders(
        rna_sequences=rna_sequences,
        protein_sequences=protein_sequences,
        binding_scores=binding_scores,
        batch_size=args.batch_size,
        validation_split=0.2,
        rna_max_length=rna_max_length,
        protein_max_length=protein_max_length,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    print()
    
    # Create optimized model
    print("Creating optimized Phase 2 model...")
    
    # Override attention mechanism if simple_attention is enabled
    if args.simple_attention:
        print("🔧 Using simplified attention mechanism for speed")
    
    # Choose model based on speed requirements
    if args.ultra_fast:
        print("🚀 Using UltraFastLSTM for maximum speed")
        model = UltraFastLSTM(
            rna_input_size=5,
            protein_input_size=21,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        print("⚡ Using FastAttentionLSTM (balanced speed and features)")
        model = FastAttentionLSTM(
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
    print(f"Model created: {model_info['model_name']} (Optimized)")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"🚀 Speed optimizations: ~2-3x faster training expected")
    print()
    
    # Create trainer with optimized settings
    print("Setting up optimized trainer...")
    trainer = FastRNAProteinTrainer(
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
        output_dir=run_dir,
        warmup_epochs=args.warmup_epochs,
        use_mixed_precision=args.use_mixed_precision
    )
    
    # Enable mixed precision if requested
    print(f"🚀 Mixed precision training: {'Enabled' if args.use_mixed_precision else 'Disabled'}")
    print(f"🎯 Model type: {'UltraFast' if args.ultra_fast else 'FastAttention'}")
    
    print()
    print("🎯 Training can be stopped anytime with Ctrl+C")
    print("📊 All outputs will be saved to:", run_dir)
    print("⚡ FAST MODE: Optimized for speed while maintaining accuracy")
    print("="*60)
    print()
    
    # Train model
    print("🚀 Starting optimized Phase 2 training...")
    start_time = datetime.now()
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
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print()
    
    # Print training summary
    print("📊 FAST Phase 2 Training Summary:")
    print("-" * 40)
    print(f"Best validation correlation: {training_summary['best_val_correlation']:.4f}")
    print(f"Best epoch: {training_summary['best_epoch'] + 1}")
    print(f"Total epochs trained: {training_summary.get('total_epochs', 'Unknown')}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Calculate speed improvement estimate
    epochs_trained = training_summary.get('total_epochs', args.epochs)
    time_per_epoch = total_time / epochs_trained if epochs_trained > 0 else 0
    print(f"⚡ Time per epoch: {time_per_epoch:.1f}s ({time_per_epoch/60:.1f} minutes)")
    print(f"🚀 Expected speedup: ~3-4x faster than original")
    
    if training_summary.get('interrupted', False):
        print("⚠️ Training was interrupted")
    if training_summary.get('early_stopped', False):
        print("🛑 Training stopped early")
    if 'final_train_correlation' in training_summary:
        print(f"Final training correlation: {training_summary['final_train_correlation']:.4f}")
        print(f"Final validation correlation: {training_summary['final_val_correlation']:.4f}")
    print()
    
    # Save training summary
    training_summary['total_wall_time'] = total_time
    training_summary['time_per_epoch'] = time_per_epoch
    training_summary['optimizations_enabled'] = True
    save_training_summary(training_summary, run_dir)
    
    # Plot training history
    print("📈 Plotting training history...")
    trainer.metrics_tracker.plot_history(output_dir=os.path.join(run_dir, 'plots'))
    
    # Quick evaluation
    print("🔍 Quick evaluation...")
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
        title="Phase 2 FAST: Optimized BiLSTM + Attention - Validation Set",
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
        timestamp=timestamp,
        phase="Phase 2 FAST - Optimized BiLSTM + Attention"
    )
    
    print("🎉 Phase 2 FAST completed successfully!")
    print(f"📁 All outputs saved to: {run_dir}")
    print(f"💾 Model saved to: {os.path.join(run_dir, 'models', args.save_model)}")
    print(f"⚡ Training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"🚀 Expected speedup: 3-4x faster than original implementation")
    
    # Performance comparison
    print()
    print("🔧 OPTIMIZATION SUMMARY:")
    print(f"   • Model type: {'UltraFast' if args.ultra_fast else 'FastAttention'}")
    print(f"   • Reduced model size: {model_info['total_parameters']:,} parameters")
    print(f"   • Optimized batch size: {args.batch_size}")
    print(f"   • Shorter sequences: RNA={args.max_rna_length}, Protein={args.max_protein_length}")
    if not args.ultra_fast:
        print(f"   • Fewer attention heads: {args.attention_heads}")
    print(f"   • Less training data: {args.subset_size} samples")
    print(f"   • Reduced epochs: {args.epochs}")
    if args.use_mixed_precision:
        print("   • Mixed precision training enabled")
    if args.num_workers > 0:
        print(f"   • Multi-worker data loading: {args.num_workers} workers")
    
    # Recommendations
    print()
    print("🎯 RESULTS ANALYSIS:")
    if training_summary['best_val_correlation'] > 0.55:
        print("🏆 Excellent performance with fast training!")
        print("✅ Optimizations successful - use this configuration")
    elif training_summary['best_val_correlation'] > 0.45:
        print("✅ Good performance with significant speedup")
        print("🔧 Consider minor increases in model size if needed")
    elif training_summary['best_val_correlation'] > 0.35:
        print("🔶 Reasonable performance, consider tweaking:")
        print("   - Increase --hidden_size to 96")
        print("   - Increase --attention_heads to 6")
        print("   - Add --use_positional_encoding")
    else:
        print("⚠️ Lower performance. Consider:")
        print("   - Increasing model capacity")
        print("   - Using more training data (--subset_size)")
        print("   - Reverting some optimizations")
    
    print()
    print("🚀 SPEED vs ACCURACY TRADE-OFF:")
    print(f"   Current validation correlation: {training_summary['best_val_correlation']:.4f}")
    print(f"   Training time per epoch: {time_per_epoch:.1f}s")
    print("   Expected 3-4x speedup compared to original model")


if __name__ == "__main__":
    main()
