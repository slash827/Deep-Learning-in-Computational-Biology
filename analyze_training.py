#!/usr/bin/env python3
"""
Utility script to enhance and analyze training summaries.
This script can:
1. Display detailed training analysis from existing summaries
2. Generate enhanced training reports
3. Compare multiple training runs
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def load_training_summary(summary_path: str) -> Dict:
    """Load a training summary JSON file."""
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {summary_path}: {e}")
        return None


def analyze_training_summary(summary: Dict, run_name: str = "Training Run") -> None:
    """Analyze and display detailed information about a training run."""
    print(f"\n{'='*60}")
    print(f"📊 TRAINING ANALYSIS: {run_name}")
    print(f"{'='*60}")
    
    # Basic information
    print(f"🎯 RESULTS SUMMARY:")
    print(f"   Best Validation Correlation: {summary.get('best_val_correlation', 'N/A'):.4f}")
    print(f"   Best Epoch: {summary.get('best_epoch', 'N/A') + 1}")
    print(f"   Total Epochs: {summary.get('total_epochs', summary.get('total_epochs_trained', 'N/A'))}")
    print(f"   Training Time: {summary.get('total_training_time', 0)/60:.1f} minutes")
    
    if summary.get('early_stopped'):
        print(f"   Status: Early stopped")
    elif summary.get('interrupted'):
        print(f"   Status: Interrupted by user")
    else:
        print(f"   Status: Completed all epochs")
    
    # Final metrics
    print(f"\n📈 FINAL METRICS:")
    print(f"   Final Train Loss: {summary.get('final_train_loss', 'N/A'):.6f}")
    print(f"   Final Val Loss: {summary.get('final_val_loss', 'N/A'):.6f}")
    print(f"   Final Train Correlation: {summary.get('final_train_correlation', 'N/A'):.4f}")
    print(f"   Final Val Correlation: {summary.get('final_val_correlation', 'N/A'):.4f}")
    
    # Model information
    if 'model_info' in summary:
        model_info = summary['model_info']
        print(f"\n🏗️  MODEL INFORMATION:")
        print(f"   Model Name: {model_info.get('model_name', 'N/A')}")
        print(f"   Total Parameters: {model_info.get('total_parameters', 'N/A'):,}")
        if 'rna_hidden_size' in model_info:
            print(f"   RNA Hidden Size: {model_info.get('rna_hidden_size')}")
        if 'protein_embedding_dim' in model_info:
            print(f"   Protein Embedding Dim: {model_info.get('protein_embedding_dim')}")
        if 'num_layers' in model_info:
            print(f"   Number of Layers: {model_info.get('num_layers')}")
        if 'dropout' in model_info:
            print(f"   Dropout: {model_info.get('dropout')}")
    
    # Training configuration
    print(f"\n⚙️  TRAINING CONFIG:")
    print(f"   Mixed Precision: {summary.get('mixed_precision', 'N/A')}")
    print(f"   Model Compiled: {summary.get('model_compiled', 'N/A')}")
    print(f"   Optimizations: {summary.get('optimizations_enabled', 'N/A')}")
    
    # Epoch-by-epoch analysis if available
    if 'epoch_history' in summary:
        analyze_epoch_history(summary['epoch_history'])
    
    # Summary statistics if available
    if 'training_metrics_summary' in summary:
        analyze_metrics_summary(summary['training_metrics_summary'])


def analyze_epoch_history(epoch_history: List[Dict]) -> None:
    """Analyze epoch-by-epoch training history."""
    print(f"\n📊 EPOCH-BY-EPOCH ANALYSIS:")
    
    if not epoch_history:
        print("   No detailed epoch history available")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(epoch_history)
    
    print(f"   Total Epochs: {len(epoch_history)}")
    
    # Display first few and last few epochs
    print(f"\n   📋 TRAINING PROGRESS TABLE:")
    print(f"   {'Epoch':<6} {'Train Loss':<12} {'Val Loss':<10} {'Train Corr':<12} {'Val Corr':<10}")
    print(f"   {'-'*60}")
    
    # Show first 3 epochs
    for i, epoch_data in enumerate(epoch_history[:3]):
        epoch_num = epoch_data['epoch'] + 1
        train_loss = f"{epoch_data['train_loss']:.6f}" if epoch_data['train_loss'] is not None else "N/A"
        val_loss = f"{epoch_data['val_loss']:.6f}" if epoch_data['val_loss'] is not None else "N/A"
        train_corr = f"{epoch_data['train_correlation']:.4f}" if epoch_data['train_correlation'] is not None else "N/A"
        val_corr = f"{epoch_data['val_correlation']:.4f}" if epoch_data['val_correlation'] is not None else "N/A"
        print(f"   {epoch_num:<6} {train_loss:<12} {val_loss:<10} {train_corr:<12} {val_corr:<10}")
    
    # Show ellipsis if more than 6 epochs
    if len(epoch_history) > 6:
        print(f"   {'...':<6} {'...':<12} {'...':<10} {'...':<12} {'...':<10}")
        
        # Show last 3 epochs
        for epoch_data in epoch_history[-3:]:
            epoch_num = epoch_data['epoch'] + 1
            train_loss = f"{epoch_data['train_loss']:.6f}" if epoch_data['train_loss'] is not None else "N/A"
            val_loss = f"{epoch_data['val_loss']:.6f}" if epoch_data['val_loss'] is not None else "N/A"
            train_corr = f"{epoch_data['train_correlation']:.4f}" if epoch_data['train_correlation'] is not None else "N/A"
            val_corr = f"{epoch_data['val_correlation']:.4f}" if epoch_data['val_correlation'] is not None else "N/A"
            print(f"   {epoch_num:<6} {train_loss:<12} {val_loss:<10} {train_corr:<12} {val_corr:<10}")
    elif len(epoch_history) > 3:
        # Show remaining epochs
        for epoch_data in epoch_history[3:]:
            epoch_num = epoch_data['epoch'] + 1
            train_loss = f"{epoch_data['train_loss']:.6f}" if epoch_data['train_loss'] is not None else "N/A"
            val_loss = f"{epoch_data['val_loss']:.6f}" if epoch_data['val_loss'] is not None else "N/A"
            train_corr = f"{epoch_data['train_correlation']:.4f}" if epoch_data['train_correlation'] is not None else "N/A"
            val_corr = f"{epoch_data['val_correlation']:.4f}" if epoch_data['val_correlation'] is not None else "N/A"
            print(f"   {epoch_num:<6} {train_loss:<12} {val_loss:<10} {train_corr:<12} {val_corr:<10}")
    
    # Calculate trends
    val_corrs = [e['val_correlation'] for e in epoch_history if e['val_correlation'] is not None]
    if len(val_corrs) > 1:
        trend = val_corrs[-1] - val_corrs[0]
        print(f"\n   📈 TRENDS:")
        print(f"   Validation Correlation Trend: {trend:+.4f} (from {val_corrs[0]:.4f} to {val_corrs[-1]:.4f})")
        
        # Find best improvement
        best_improvement = 0
        best_improvement_epoch = 0
        for i in range(1, len(val_corrs)):
            improvement = val_corrs[i] - val_corrs[i-1]
            if improvement > best_improvement:
                best_improvement = improvement
                best_improvement_epoch = i
        
        if best_improvement > 0:
            print(f"   Best Single Epoch Improvement: +{best_improvement:.4f} at epoch {best_improvement_epoch + 1}")


def analyze_metrics_summary(metrics_summary: Dict) -> None:
    """Analyze training metrics summary statistics."""
    print(f"\n📊 METRICS STATISTICS:")
    
    print(f"   🔻 LOSS STATISTICS:")
    if metrics_summary.get('min_train_loss') is not None:
        print(f"   Train Loss Range: {metrics_summary['min_train_loss']:.6f} - {metrics_summary['max_train_loss']:.6f}")
    if metrics_summary.get('min_val_loss') is not None:
        print(f"   Val Loss Range: {metrics_summary['min_val_loss']:.6f} - {metrics_summary['max_val_loss']:.6f}")
    
    print(f"   📈 CORRELATION STATISTICS:")
    if metrics_summary.get('min_train_correlation') is not None:
        print(f"   Train Correlation Range: {metrics_summary['min_train_correlation']:.4f} - {metrics_summary['max_train_correlation']:.4f}")
    if metrics_summary.get('min_val_correlation') is not None:
        print(f"   Val Correlation Range: {metrics_summary['min_val_correlation']:.4f} - {metrics_summary['max_val_correlation']:.4f}")


def find_training_runs(runs_dir: str = "runs") -> List[str]:
    """Find all training run directories."""
    if not os.path.exists(runs_dir):
        return []
    
    training_runs = []
    for item in os.listdir(runs_dir):
        item_path = os.path.join(runs_dir, item)
        if os.path.isdir(item_path):
            summary_path = os.path.join(item_path, "training_summary.json")
            if os.path.exists(summary_path):
                training_runs.append(item_path)
    
    return sorted(training_runs)


def compare_training_runs(run_paths: List[str]) -> None:
    """Compare multiple training runs."""
    print(f"\n{'='*80}")
    print(f"🔍 COMPARING TRAINING RUNS")
    print(f"{'='*80}")
    
    summaries = []
    run_names = []
    
    for run_path in run_paths:
        summary_path = os.path.join(run_path, "training_summary.json")
        summary = load_training_summary(summary_path)
        if summary:
            summaries.append(summary)
            run_names.append(os.path.basename(run_path))
    
    if not summaries:
        print("No valid training summaries found for comparison")
        return
    
    # Create comparison table
    print(f"\n📊 COMPARISON TABLE:")
    print(f"{'Run Name':<30} {'Best Val Corr':<15} {'Final Val Corr':<15} {'Epochs':<8} {'Time (min)':<10}")
    print(f"{'-'*85}")
    
    for i, (run_name, summary) in enumerate(zip(run_names, summaries)):
        best_corr = f"{summary.get('best_val_correlation', 0):.4f}"
        final_corr = f"{summary.get('final_val_correlation', 0):.4f}"
        epochs = str(summary.get('total_epochs', summary.get('total_epochs_trained', 'N/A')))
        time_mins = f"{summary.get('total_training_time', 0)/60:.1f}"
        
        print(f"{run_name:<30} {best_corr:<15} {final_corr:<15} {epochs:<8} {time_mins:<10}")
    
    # Find best run
    best_run_idx = max(range(len(summaries)), key=lambda i: summaries[i].get('best_val_correlation', -1))
    print(f"\n🏆 BEST RUN: {run_names[best_run_idx]} (correlation: {summaries[best_run_idx].get('best_val_correlation'):.4f})")


def export_training_data(summary: Dict, output_path: str) -> None:
    """Export training data to CSV for external analysis."""
    if 'epoch_history' not in summary:
        print("No epoch history available for export")
        return
    
    # Convert epoch history to DataFrame
    df = pd.DataFrame(summary['epoch_history'])
    df['epoch'] = df['epoch'] + 1  # Make epochs 1-indexed for readability
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Training data exported to: {output_path}")


def main():
    """Main function for training analysis."""
    parser = argparse.ArgumentParser(description='Analyze RNA-Protein training results')
    parser.add_argument('--run_dir', type=str, help='Specific run directory to analyze')
    parser.add_argument('--runs_dir', type=str, default='runs', help='Directory containing all runs')
    parser.add_argument('--compare', action='store_true', help='Compare all runs')
    parser.add_argument('--export', type=str, help='Export training data to CSV file')
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Analyze specific run
        summary_path = os.path.join(args.run_dir, "training_summary.json")
        if os.path.exists(summary_path):
            summary = load_training_summary(summary_path)
            if summary:
                run_name = os.path.basename(args.run_dir)
                analyze_training_summary(summary, run_name)
                
                if args.export:
                    export_training_data(summary, args.export)
            else:
                print(f"Could not load training summary from {summary_path}")
        else:
            print(f"No training summary found at {summary_path}")
    
    elif args.compare:
        # Compare all runs
        run_paths = find_training_runs(args.runs_dir)
        if run_paths:
            compare_training_runs(run_paths)
        else:
            print(f"No training runs found in {args.runs_dir}")
    
    else:
        # Interactive mode - show all available runs
        run_paths = find_training_runs(args.runs_dir)
        if not run_paths:
            print(f"No training runs found in {args.runs_dir}")
            return
        
        print("📁 AVAILABLE TRAINING RUNS:")
        for i, run_path in enumerate(run_paths):
            run_name = os.path.basename(run_path)
            summary_path = os.path.join(run_path, "training_summary.json")
            summary = load_training_summary(summary_path)
            if summary:
                best_corr = summary.get('best_val_correlation', 0)
                epochs = summary.get('total_epochs', summary.get('total_epochs_trained', 'N/A'))
                print(f"{i+1:2d}. {run_name:<35} (Best: {best_corr:.4f}, Epochs: {epochs})")
        
        print(f"\nOptions:")
        print(f"1. Analyze a specific run (enter number 1-{len(run_paths)})")
        print(f"2. Compare all runs (enter 'compare')")
        print(f"3. Exit (enter 'exit')")
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice.lower() == 'compare':
            compare_training_runs(run_paths)
        elif choice.lower() == 'exit':
            return
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(run_paths):
                    run_path = run_paths[idx]
                    summary_path = os.path.join(run_path, "training_summary.json")
                    summary = load_training_summary(summary_path)
                    if summary:
                        run_name = os.path.basename(run_path)
                        analyze_training_summary(summary, run_name)
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input")


if __name__ == "__main__":
    main()
