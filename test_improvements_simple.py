#!/usr/bin/env python3
"""
Quick test script to run the improved training without import issues.
"""

import os
import sys
import subprocess
import time

def test_improved_training():
    """Test the improved training with better Ctrl+C handling."""
    print("🔧 TESTING IMPROVED TRAINING")
    print("="*60)
    print("Running phase2_improved.py with bug fixes:")
    print("✅ Better Ctrl+C handling (graceful and force stop)")
    print("✅ Anti-stalling measures (LR boosts and parameter noise)")
    print("✅ Lightweight imports (no heavy seaborn/matplotlib)")
    print("✅ Enhanced error handling")
    print("")
    
    print("📋 INSTRUCTIONS:")
    print("1. Training will start normally")
    print("2. Press Ctrl+C ONCE to gracefully stop training")
    print("3. Press Ctrl+C TWICE to force stop immediately")
    print("4. Watch for anti-stalling measures if training plateaus")
    print("")
    
    # Ask user for confirmation
    response = input("Start improved training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    print("\n🚀 Starting improved training...")
    print("="*60)
    
    # Run the improved training script with small batch for quick testing
    cmd = [
        sys.executable, "phase2_improved.py",
        "--epochs", "20",
        "--batch_size", "8", 
        "--patience", "5",
        "--hidden_size", "128",
        "--max_seq_length", "500"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("")
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, cwd=os.getcwd(), text=True, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
        elif result.returncode == 1:
            print("\n⚠️  Training stopped (likely by Ctrl+C or early stopping)")
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by Ctrl+C")
    except Exception as e:
        print(f"\n❌ Error running training: {e}")


def test_specific_features():
    """Test specific improved features."""
    print("🔍 TESTING SPECIFIC FEATURES")
    print("="*60)
    
    features_to_test = [
        "1. Ctrl+C handling (graceful stop)",
        "2. Ctrl+C handling (force stop)",
        "3. Anti-stalling detection and recovery",
        "4. Memory usage monitoring",
        "5. Enhanced error reporting"
    ]
    
    print("Available features to test:")
    for feature in features_to_test:
        print(f"  {feature}")
    
    print("")
    print("To test these features:")
    print("1. Run: python phase2_improved.py --epochs 50")
    print("2. Let it train for a few epochs")
    print("3. Press Ctrl+C once for graceful stop")
    print("4. Or press Ctrl+C twice for force stop")
    print("5. Watch for stall detection around epoch 10-15")


def show_improvement_summary():
    """Show summary of improvements made."""
    print("📊 IMPROVEMENT SUMMARY")
    print("="*60)
    
    improvements = {
        "🛑 Ctrl+C Handling": [
            "✅ Graceful stop: Finishes current batch, saves progress",
            "✅ Force stop: Immediate termination on second Ctrl+C",
            "✅ Better signal handling throughout training loop",
            "✅ Progress bar integration respects interruptions"
        ],
        "🔧 Anti-Stalling Measures": [
            "✅ Automatic stall detection (monitors 5-epoch windows)",
            "✅ Learning rate boost when stalling detected",
            "✅ Parameter noise injection to escape local minima",
            "✅ Improved LR scheduler settings (less aggressive)"
        ],
        "⚡ Performance & Stability": [
            "✅ Lightweight imports (removed heavy seaborn dependencies)",
            "✅ Better memory management and monitoring",
            "✅ Learning rate warmup for training stability",
            "✅ Enhanced error handling and reporting"
        ],
        "🔍 Monitoring & Debugging": [
            "✅ Real-time GPU memory usage tracking",
            "✅ Detailed progress reporting",
            "✅ Better epoch timing and statistics",
            "✅ Comprehensive training summaries"
        ]
    }
    
    for category, items in improvements.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")
    
    print(f"\n{'='*60}")
    print("🎯 EXPECTED RESULTS:")
    print("• Ctrl+C should work properly (no more hanging)")
    print("• Training should not stall at 65% correlation")
    print("• Better training stability and convergence")
    print("• Faster startup (no heavy imports)")
    print("• More informative error messages")


def main():
    """Main test function."""
    print("🔧 IMPROVED TRAINING TESTER")
    print("="*60)
    
    options = {
        "1": ("Run improved training test", test_improved_training),
        "2": ("Show specific features", test_specific_features),
        "3": ("Show improvement summary", show_improvement_summary),
        "4": ("All of the above", lambda: [f() for f in [show_improvement_summary, test_specific_features, test_improved_training]])
    }
    
    print("Choose an option:")
    for key, (desc, _) in options.items():
        print(f"{key}. {desc}")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in options:
        print(f"\n{'-'*60}")
        try:
            options[choice][1]()
        except KeyboardInterrupt:
            print(f"\n🛑 Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("Invalid choice. Please run again and select 1-4.")


if __name__ == "__main__":
    main()
