#!/usr/bin/env python3
"""
Phase 2 Ultra Fast: Maximum Speed RNA-Protein Binding Prediction

This script demonstrates the key optimizations to make training 3-5x faster
while maintaining reasonable model accuracy.
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import time

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from src.data.dataset import load_training_data, create_data_loaders
from src.models.lstm_attention_fast import UltraFastLSTM
from src.training.trainer import RNAProteinTrainer
from src.training.evaluation import plot_predictions_vs_targets
from src.utils.helpers import create_run_directory, save_training_config, save_training_summary


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Ultra Fast: Maximum Speed Training')
    
    # Key optimization parameters
    parser.add_argument('--subset_size', type=int, default=1000,
                       help='Training data size (default: 1000 for speed)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64 for speed)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs (default: 5)')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Hidden size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                       help='Learning rate (default: 0.003)')
    
    # Optional parameters
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom run name')
    
    args = parser.parse_args()
    
    # Setup
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"phase2_ultra_fast_{timestamp}"
    
    run_dir = create_run_directory("runs", args.run_name)
    
    # Device
    if args.force_cpu:
        device = torch.device('cpu')
        print("⚠️ Using CPU (slow)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("🚀 ULTRA FAST RNA-Protein Binding Prediction")
    print("="*60)
    print("KEY SPEED OPTIMIZATIONS:")
    print(f"✅ Minimal model: {args.hidden_size} hidden units, 1 layer")
    print(f"✅ Large batch size: {args.batch_size}")
    print(f"✅ Small dataset: {args.subset_size} samples")
    print(f"✅ Few epochs: {args.epochs}")
    print(f"✅ No complex attention")
    print(f"✅ Shorter sequences")
    print(f"💻 Device: {device}")
    print()
    
    # Load data quickly
    print("⚡ Loading data...")
    start_time = time.time()
    
    rna_sequences, protein_sequences, binding_scores = load_training_data(
        args.data_dir, subset_size=args.subset_size
    )
    
    print(f"Loaded {len(rna_sequences)} RNA-protein pairs in {time.time() - start_time:.1f}s")
    
    # Create data loaders with speed optimizations
    print("⚡ Creating data loaders...")
    start_time = time.time()
    
    train_loader, val_loader, _, _ = create_data_loaders(
        rna_sequences=rna_sequences,
        protein_sequences=protein_sequences,
        binding_scores=binding_scores,
        batch_size=args.batch_size,
        validation_split=0.2,
        rna_max_length=50,      # Reduced for speed
        protein_max_length=200, # Reduced for speed
        num_workers=0,          # Windows compatible
        pin_memory=False
    )
    
    print(f"Created data loaders in {time.time() - start_time:.1f}s")
    print()
    
    # Create ultra-fast model
    print("⚡ Creating ultra-fast model...")
    model = UltraFastLSTM(
        rna_input_size=5,
        protein_input_size=21,
        hidden_size=args.hidden_size,
        num_layers=1,           # Single layer for speed
        dropout=0.1             # Minimal dropout
    )
    
    model_info = model.get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"🚀 Expected speedup: 5-10x faster than original")
    print()
    
    # Create trainer
    print("⚡ Setting up trainer...")
    trainer = RNAProteinTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        patience=5,                    # Reduced patience
        min_delta=1e-4,
        max_grad_norm=1.0,
        lr_scheduler_patience=3,       # Aggressive LR reduction
        lr_scheduler_factor=0.5,
        output_dir=run_dir,
        warmup_epochs=0                # No warmup for speed
    )
    
    # Save configuration
    config = {
        'model': 'UltraFastLSTM',
        'subset_size': args.subset_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'hidden_size': args.hidden_size,
        'learning_rate': args.learning_rate,
        'device': str(device),
        'optimizations': 'ultra_fast_mode',
        'timestamp': datetime.now().isoformat()
    }
    save_training_config(config, run_dir)
    
    print("🚀 Starting ULTRA FAST training...")
    print("="*60)
    
    # Train with timing
    overall_start = time.time()
    
    try:
        training_summary = trainer.train(
            num_epochs=args.epochs,
            save_path="ultra_fast_model.pth"
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted")
        training_summary = trainer.metrics_tracker.get_summary()
        training_summary['interrupted'] = True
    
    total_time = time.time() - overall_start
    
    print("\n" + "="*60)
    print("🎉 ULTRA FAST TRAINING COMPLETED!")
    print("="*60)
    
    # Results
    print("📊 RESULTS:")
    print(f"Best validation correlation: {training_summary['best_val_correlation']:.4f}")
    print(f"Best epoch: {training_summary['best_epoch'] + 1}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    epochs_trained = len(trainer.metrics_tracker.train_losses)
    time_per_epoch = total_time / epochs_trained if epochs_trained > 0 else 0
    print(f"⚡ Time per epoch: {time_per_epoch:.1f}s")
    
    # Compare to original
    original_time_estimate = time_per_epoch * 12  # Original took ~20 minutes per epoch
    speedup = 1200 / time_per_epoch if time_per_epoch > 0 else 0  # 20 minutes = 1200s
    print(f"🚀 Estimated speedup: {speedup:.1f}x faster than original!")
    
    # Save results
    training_summary.update({
        'total_wall_time': total_time,
        'time_per_epoch': time_per_epoch,
        'estimated_speedup': speedup,
        'model_info': model_info
    })
    save_training_summary(training_summary, run_dir)
    
    # Quick evaluation
    print("\n🔍 Final evaluation...")
    from src.training.evaluation import evaluate_model
    
    val_metrics, val_predictions, val_targets = evaluate_model(
        trainer.model, val_loader, device
    )
    
    print("Validation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Create plots
    plot_predictions_vs_targets(
        val_targets, val_predictions,
        title="Ultra Fast Training Results",
        output_dir=os.path.join(run_dir, 'plots')
    )
    
    print(f"\n📁 All outputs saved to: {run_dir}")
    
    # Recommendations
    print("\n💡 OPTIMIZATION ANALYSIS:")
    if speedup > 5:
        print("🏆 Excellent speedup achieved!")
    elif speedup > 3:
        print("✅ Good speedup achieved!")
    else:
        print("🔶 Moderate speedup - consider more optimizations")
    
    if training_summary['best_val_correlation'] > 0.4:
        print("✅ Good accuracy maintained")
    elif training_summary['best_val_correlation'] > 0.3:
        print("🔶 Reasonable accuracy - consider slight model increase")
    else:
        print("⚠️ Low accuracy - may need larger model")
    
    print("\n🔧 TO IMPROVE ACCURACY (with some speed cost):")
    print("  • Increase --hidden_size to 96")
    print("  • Increase --subset_size to 2000")
    print("  • Add more epochs")
    print("  • Use FastAttentionLSTM instead of UltraFastLSTM")
    
    print("\n🚀 TO MAXIMIZE SPEED:")
    print("  • Reduce --batch_size if GPU memory limited")
    print("  • Reduce --hidden_size to 32")
    print("  • Use even smaller --subset_size")
    
    print(f"\n⚡ Current configuration achieved {speedup:.1f}x speedup!")


if __name__ == "__main__":
    main()
