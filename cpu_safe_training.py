#!/usr/bin/env python3
"""
CPU-Safe Training Script - Prevents System Crashes
Optimized for CPU-only training (no external dependencies)
"""

import subprocess
import sys
import time
import os

def get_safe_config():
    """Get ultra-conservative configuration for CPU training"""
    print("🔧 Using ultra-conservative CPU-safe settings")
    print("   (No resource monitoring - playing it very safe)")
    
    # Ultra-conservative settings to prevent any crashes
    return {
        'batch_size': 8,
        'hidden_size': 32,
        'subset_size': 100,
        'epochs': 3
    }

def main():
    print("=" * 60)
    print("🛡️  CPU-SAFE TRAINING MODE")
    print("   Ultra-conservative settings to prevent crashes")
    print("=" * 60)
    
    print("💡 TIP: Close other heavy applications for best performance")
    
    # Get safe configuration
    config = get_safe_config()
    
    print(f"\n🚀 Starting ULTRA-FAST training with safe settings:")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Hidden Size: {config['hidden_size']}")
    print(f"   Subset Size: {config['subset_size']}")
    print(f"   Epochs: {config['epochs']}")
    
    # Build command for ultra-fast training
    cmd = [
        'python', 'phase2_ultra_fast.py',
        '--force_cpu',
        '--batch_size', str(config['batch_size']),
        '--hidden_size', str(config['hidden_size']),
        '--subset_size', str(config['subset_size']),
        '--epochs', str(config['epochs'])
    ]
    
    print(f"\n📝 Command: {' '.join(cmd)}")
    print("\n⏰ Starting training in 3 seconds...")
    time.sleep(3)
    
    try:
        # Run the training
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")

if __name__ == "__main__":
    main()
