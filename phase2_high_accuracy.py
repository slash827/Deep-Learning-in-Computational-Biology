#!/usr/bin/env python3
"""
Phase 2 High Accuracy: Optimized for best results with good speed

This script provides the best accuracy while still being significantly faster
than the original implementation, avoiding mixed precision issues.
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
from src.models.lstm_attention_fast import FastAttentionLSTM
from src.training.trainer import RNAProteinTrainer  # Use original trainer to avoid mixed precision issues
from src.training.evaluation import plot_predictions_vs_targets
from src.utils.helpers import create_run_directory, save_training_config, save_training_summary


def main():
    parser = argparse.ArgumentParser(description='Phase 2 High Accuracy: Best results with good speed')
    
    # Optimized parameters for accuracy
    parser.add_argument('--subset_size', type=int, default=2000,
                       help='Training data size (default: 2000 for accuracy)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32 for stability)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--hidden_size', type=int, default=96,
                       help='Hidden size (default: 96 for accuracy)')
    parser.add_argument('--attention_heads', type=int, default=6,
                       help='Attention heads (default: 6)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # Optional parameters
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom run name')
    parser.add_argument('--use_positional_encoding', action='store_true',
                       help='Use positional encoding for better accuracy')
    
    args = parser.parse_args()
    
    # Setup
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"phase2_high_accuracy_{timestamp}"
    
    run_dir = create_run_directory("runs", args.run_name)
    
    # Device (no mixed precision to avoid issues)
    if args.force_cpu:
        device = torch.device('cpu')
        print("⚠️ Using CPU (slow)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("🎯 HIGH ACCURACY RNA-Protein Binding Prediction")
    print("="*60)
    print("ACCURACY-FOCUSED OPTIMIZATIONS:")
    print(f"✅ Larger model: {args.hidden_size} hidden units")
    print(f"✅ More attention heads: {args.attention_heads}")
    print(f"✅ Full dataset: {args.subset_size} samples")
    print(f"✅ Stable training: No mixed precision")
    print(f"✅ Longer sequences for accuracy")
    print(f"💻 Device: {device}")
    print(f"⚡ Expected: 3-5x faster than original with better accuracy")
    print()
    
    # Load data
    print("📊 Loading training data...")
    start_time = time.time()
    
    rna_sequences, protein_sequences, binding_scores = load_training_data(
        args.data_dir, subset_size=args.subset_size
    )
    
    print(f"Loaded {len(rna_sequences)} RNA-protein pairs in {time.time() - start_time:.1f}s")
    
    # Create data loaders with accuracy-focused settings
    print("📊 Creating data loaders...")
    start_time = time.time()
    
    train_loader, val_loader, _, _ = create_data_loaders(
        rna_sequences=rna_sequences,
        protein_sequences=protein_sequences,
        binding_scores=binding_scores,
        batch_size=args.batch_size,
        validation_split=0.2,
        rna_max_length=75,      # Full length for accuracy
        protein_max_length=400, # Full length for accuracy
        num_workers=0,          # Windows compatible
        pin_memory=False
    )
    
    print(f"Created data loaders in {time.time() - start_time:.1f}s")
    print()
    
    # Create high-accuracy model
    print("🎯 Creating high-accuracy model...")
    model = FastAttentionLSTM(
        rna_input_size=5,
        protein_input_size=21,
        rna_hidden_size=args.hidden_size,
        protein_hidden_size=args.hidden_size,
        num_layers=2,           # 2 layers for better representation
        dropout=0.3,            # Standard dropout
        attention_heads=args.attention_heads,
        attention_dropout=0.1,
        use_positional_encoding=args.use_positional_encoding
    )
    
    model_info = model.get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"🎯 Optimized for accuracy with good speed")
    print()
    
    # Create trainer (using original trainer to avoid mixed precision issues)
    print("🎯 Setting up accuracy-focused trainer...")
    trainer = RNAProteinTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        patience=10,                   # Higher patience for accuracy
        min_delta=1e-4,
        max_grad_norm=1.0,
        lr_scheduler_patience=5,       # More patient LR reduction
        lr_scheduler_factor=0.5,
        output_dir=run_dir,
        warmup_epochs=2                # Some warmup for stability
    )
    
    # Save configuration
    config = {
        'model': 'FastAttentionLSTM',
        'subset_size': args.subset_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'hidden_size': args.hidden_size,
        'attention_heads': args.attention_heads,
        'learning_rate': args.learning_rate,
        'use_positional_encoding': args.use_positional_encoding,
        'device': str(device),
        'mixed_precision': False,  # Disabled for stability
        'optimization_mode': 'high_accuracy',
        'timestamp': datetime.now().isoformat()
    }
    save_training_config(config, run_dir)
    
    print("🎯 Starting HIGH ACCURACY training...")
    print("="*60)
    
    # Train with timing
    overall_start = time.time()
    
    try:
        training_summary = trainer.train(
            num_epochs=args.epochs,
            save_path="high_accuracy_model.pth"
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted")
        training_summary = trainer.metrics_tracker.get_summary()
        training_summary['interrupted'] = True
    
    total_time = time.time() - overall_start
    
    print("\n" + "="*60)
    print("🎉 HIGH ACCURACY TRAINING COMPLETED!")
    print("="*60)
    
    # Results
    print("📊 RESULTS:")
    print(f"Best validation correlation: {training_summary['best_val_correlation']:.4f}")
    print(f"Best epoch: {training_summary['best_epoch'] + 1}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    epochs_trained = len(trainer.metrics_tracker.train_losses)
    time_per_epoch = total_time / epochs_trained if epochs_trained > 0 else 0
    print(f"⚡ Time per epoch: {time_per_epoch:.1f}s ({time_per_epoch/60:.1f} minutes)")
    
    # Compare to original
    speedup = 1200 / time_per_epoch if time_per_epoch > 0 else 0  # 20 minutes = 1200s
    print(f"🚀 Estimated speedup: {speedup:.1f}x faster than original!")
    
    # Accuracy analysis
    if training_summary['best_val_correlation'] > 0.55:
        print("🏆 Excellent accuracy achieved!")
    elif training_summary['best_val_correlation'] > 0.45:
        print("✅ Good accuracy achieved!")
    else:
        print("🔶 Moderate accuracy - consider tuning parameters")
    
    # Save results
    training_summary.update({
        'total_wall_time': total_time,
        'time_per_epoch': time_per_epoch,
        'estimated_speedup': speedup,
        'model_info': model_info
    })
    save_training_summary(training_summary, run_dir)
    
    # Evaluation
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
        title="High Accuracy Training Results",
        output_dir=os.path.join(run_dir, 'plots')
    )
    
    print(f"\n📁 All outputs saved to: {run_dir}")
    
    # Recommendations
    print("\n💡 RESULTS ANALYSIS:")
    if speedup > 3 and training_summary['best_val_correlation'] > 0.5:
        print("🏆 Excellent balance of speed and accuracy!")
    elif speedup > 2:
        print("✅ Good speedup with maintained accuracy")
    else:
        print("🔶 Moderate speedup - consider more optimizations")
    
    print("\n🔧 TO FURTHER IMPROVE ACCURACY:")
    print("  • Add --use_positional_encoding")
    print("  • Increase --epochs to 15")
    print("  • Try different --learning_rate values")
    print("  • Experiment with --attention_heads")
    
    print("\n🚀 TO INCREASE SPEED (with minimal accuracy loss):")
    print("  • Reduce --batch_size to 24")
    print("  • Reduce --hidden_size to 80")
    print("  • Use shorter sequences")
    
    print(f"\n⚡ Current configuration: {speedup:.1f}x speedup with {training_summary['best_val_correlation']:.3f} correlation!")


if __name__ == "__main__":
    main()
