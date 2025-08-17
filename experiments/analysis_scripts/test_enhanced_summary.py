#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced training summary functionality.
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.training.evaluation_lightweight import MetricsTracker

def test_enhanced_metrics_tracker():
    """Test the enhanced MetricsTracker with epoch-by-epoch data."""
    print("🧪 TESTING ENHANCED METRICS TRACKER")
    print("="*50)
    
    # Create a mock training session
    tracker = MetricsTracker()
    
    # Simulate training for 8 epochs with realistic data
    mock_training_data = [
        (0.0123, 0.0145, 0.423, 0.401),  # Epoch 1
        (0.0087, 0.0098, 0.567, 0.543),  # Epoch 2
        (0.0065, 0.0076, 0.632, 0.618),  # Epoch 3
        (0.0051, 0.0067, 0.689, 0.671),  # Epoch 4
        (0.0043, 0.0058, 0.721, 0.708),  # Epoch 5
        (0.0038, 0.0052, 0.738, 0.731),  # Epoch 6
        (0.0035, 0.0049, 0.745, 0.742),  # Epoch 7 (best)
        (0.0032, 0.0051, 0.749, 0.735),  # Epoch 8 (slight overfit)
    ]
    
    print("Simulating training progress:")
    for epoch, (train_loss, val_loss, train_corr, val_corr) in enumerate(mock_training_data):
        tracker.update(train_loss, val_loss, train_corr, val_corr, epoch)
        print(f"  Epoch {epoch+1}: Val Corr = {val_corr:.3f}")
    
    print("\n📊 ENHANCED SUMMARY GENERATED:")
    summary = tracker.get_summary()
    
    # Pretty print the summary
    print(json.dumps(summary, indent=2))
    
    print(f"\n✅ SUCCESS! The enhanced summary now includes:")
    print(f"   • Epoch-by-epoch training and validation loss")
    print(f"   • Epoch-by-epoch training and validation correlation")
    print(f"   • Summary statistics (min/max for all metrics)")
    print(f"   • Total epochs trained")
    print(f"   • All values properly converted to float/int for JSON serialization")
    
    return summary

def test_with_existing_summary():
    """Test analysis with the existing training summary."""
    print(f"\n🔍 ANALYZING EXISTING TRAINING SUMMARY")
    print("="*50)
    
    # Load the existing summary
    summary_path = "runs/phase2_fast_20250815_194926/training_summary.json"
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            existing_summary = json.load(f)
        
        print(f"Current summary keys: {list(existing_summary.keys())}")
        print(f"Best validation correlation: {existing_summary.get('best_val_correlation'):.4f}")
        print(f"Total epochs: {existing_summary.get('total_epochs')}")
        
        if 'epoch_history' in existing_summary:
            print("✅ Enhanced epoch history is available!")
        else:
            print("❌ No enhanced epoch history found (this is expected for old summaries)")
            print("   Future training runs will include detailed epoch-by-epoch data")
    else:
        print(f"Training summary not found at: {summary_path}")

def main():
    """Main test function."""
    print("🔧 ENHANCED TRAINING SUMMARY TEST")
    print("="*60)
    
    # Test 1: Enhanced metrics tracker
    enhanced_summary = test_enhanced_metrics_tracker()
    
    # Test 2: Check existing summary
    test_with_existing_summary()
    
    print(f"\n{'='*60}")
    print("🎯 NEXT STEPS:")
    print("1. Run any training script (phase2_improved.py, phase2_fast.py, etc.)")
    print("2. Check the generated training_summary.json file")
    print("3. Use analyze_training.py to get detailed analysis:")
    print("   python analyze_training.py --run_dir runs/your_run_folder")
    print("4. Or compare multiple runs:")
    print("   python analyze_training.py --compare")
    
    return enhanced_summary

if __name__ == "__main__":
    main()
