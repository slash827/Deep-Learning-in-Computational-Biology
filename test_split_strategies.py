#!/usr/bin/env python3
"""
Test different splitting strategies to understand their impact on generalization.

Usage: python test_split_strategies.py
"""

import sys
sys.path.append('src')

from src.data.strategic_split import compare_splitting_strategies
from src.data.dataset import load_training_data

def main():
    print("🧪 TESTING SPLIT STRATEGIES FOR BETTER GENERALIZATION")
    print("=" * 70)
    
    # Load data
    data_dir = "src/data"
    subset_size = 1000  # Use smaller subset for quick testing
    
    print(f"📖 Loading data from: {data_dir}")
    print(f"📊 Using subset size: {subset_size}")
    
    try:
        rna_sequences, protein_sequences, binding_scores = load_training_data(data_dir, subset_size)
        
        print(f"✅ Data loaded:")
        print(f"   📊 {len(rna_sequences)} RNA-protein pairs")
        print(f"   🧬 {len(set(rna_sequences))} unique RNA sequences")
        print(f"   🧬 {len(set(protein_sequences))} unique proteins")
        
        # Test all strategies
        results = compare_splitting_strategies(rna_sequences, protein_sequences, binding_scores)
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 STRATEGY COMPARISON SUMMARY")
        print("=" * 70)
        
        for strategy, data in results.items():
            stats = data['stats']
            print(f"\n🎯 {strategy.upper()} Strategy:")
            print(f"   📈 Novel RNAs in validation: {stats['novel_val_rnas']}/{stats['val_rnas']} ({stats['rna_novelty_ratio']:.1%})")
            print(f"   🧬 Novel proteins in validation: {stats['novel_val_proteins']}/{stats['val_proteins']} ({stats['protein_novelty_ratio']:.1%})")
            print(f"   📊 Train/Val split: {len(data['train_indices'])}/{len(data['val_indices'])}")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATIONS FOR BETTER GENERALIZATION")
        print("=" * 70)
        
        print("1. 🎯 MIXED strategy: Tests both novel RNAs AND novel proteins")
        print("   - Best for overall generalization")
        print("   - Realistic evaluation scenario")
        print("   - Use: --split_strategy mixed")
        
        print("\n2. 🧬 PROTEIN strategy: Tests novel proteins only")
        print("   - Good for protein binding generalization")
        print("   - Use: --split_strategy protein")
        
        print("\n3. 🧬 RNA strategy: Tests novel RNAs only")
        print("   - Good for RNA structure generalization")
        print("   - Use: --split_strategy rna")
        
        print("\n4. 🎲 RANDOM strategy: Current method (not recommended)")
        print("   - May overestimate performance")
        print("   - Data leakage possible")
        
        print("\n" + "=" * 70)
        print("🚀 SUGGESTED TRAINING COMMANDS:")
        print("=" * 70)
        
        print("# BEST: Simulate actual test scenario (43 proteins vs many RNAs):")
        print("python phase2_siamese.py --split_strategy test_simulation --subset_size 2000 --epochs 10")
        
        print("\n# Test generalization to novel proteins & RNAs:")
        print("python phase2_siamese.py --split_strategy mixed --subset_size 2000 --epochs 10")
        
        print("\n# Test generalization to novel proteins:")
        print("python phase2_siamese.py --split_strategy protein --subset_size 2000 --epochs 10")
        
        print("\n# Compare with current method:")
        print("python phase2_siamese.py --split_strategy random --subset_size 2000 --epochs 10")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
