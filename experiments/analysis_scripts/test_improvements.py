#!/usr/bin/env python3
"""
Quick test script to demonstrate the improved training with bug fixes.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_ctrl_c_handling():
    """Test the improved Ctrl+C handling."""
    print("🧪 TESTING IMPROVED CTRL+C HANDLING")
    print("="*50)
    print("This script will demonstrate the improved Ctrl+C handling.")
    print("You can:")
    print("1. Press Ctrl+C once to gracefully stop training")
    print("2. Press Ctrl+C twice to force stop immediately")
    print("")
    
    try:
        # Import after adding to path - with error handling
        print("Loading trainer (this may take a moment)...")
        from src.training.trainer_improved import ImprovedRNAProteinTrainer
        import torch
        import torch.nn as nn
        print("✅ Trainer loaded successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Import interrupted by Ctrl+C")
        print("This is expected if you pressed Ctrl+C during import.")
        print("Try running again and wait for imports to complete.")
        return None
    except Exception as e:
        print(f"\n❌ Error importing trainer: {e}")
        print("This might be due to missing dependencies.")
        print("Try running the actual training script instead.")
        return None
    
    # Create a dummy trainer for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x, y):
            return self.linear(x.float().mean(dim=-1, keepdim=True))
        
        def get_model_info(self):
            return {
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'memory_estimate_mb': 1.0
            }
    
    # Create dummy data
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'rna': torch.randint(0, 10, (50,)),
                'protein': torch.randint(0, 20, (30,)),
                'score': torch.rand(1)
            }
    
    # Create dummy data loaders
    dataset = DummyDataset(100)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Create model and trainer
    model = DummyModel()
    device = torch.device('cpu')
    
    trainer = ImprovedRNAProteinTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-3,
        patience=5
    )
    
    print("Starting dummy training...")
    print("Try pressing Ctrl+C to test the improved handling!")
    print("")
    
    try:
        # Train for a few epochs
        summary = trainer.train(num_epochs=20, save_path='dummy_model.pth')
        print("Training completed normally.")
        return summary
    except Exception as e:
        print(f"Error during training: {e}")
        return None


def run_actual_training():
    """Run the actual improved training."""
    print("🚀 RUNNING IMPROVED TRAINING")
    print("="*50)
    print("Running phase2_improved.py with better Ctrl+C handling and anti-stalling...")
    print("")
    
    # Run the improved training script
    cmd = f"python phase2_improved.py --epochs 50 --batch_size 16 --patience 10"
    print(f"Command: {cmd}")
    print("")
    
    os.system(cmd)


def main():
    """Main function to test improvements."""
    print("🔧 IMPROVED TRAINING DEMONSTRATION")
    print("="*60)
    print("This script demonstrates the bug fixes:")
    print("1. Better Ctrl+C handling - graceful and force stop")
    print("2. Anti-stalling measures - learning rate boosts and parameter noise")
    print("3. Improved memory management")
    print("4. Better error reporting")
    print("")
    
    choice = input("Choose test:\n1. Test Ctrl+C handling (dummy)\n2. Run actual improved training\n3. Both\nChoice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        test_ctrl_c_handling()
        print("\n" + "="*60 + "\n")
    
    if choice in ['2', '3']:
        run_actual_training()


if __name__ == "__main__":
    main()
