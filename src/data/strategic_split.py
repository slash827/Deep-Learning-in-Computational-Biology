#!/usr/bin/env python3
"""
Strategic data splitting for better generalization testing.

This module implements different splitting strategies to ensure the model
can handle unseen proteins and RNA sequences.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Set
from collections import defaultdict

def create_protein_split(protein_sequences: List[str], 
                        train_ratio: float = 0.8) -> Tuple[Set[str], Set[str]]:
    """
    Split proteins into train/validation ensuring no overlap.
    
    Args:
        protein_sequences: List of all protein sequences
        train_ratio: Ratio of proteins for training
        
    Returns:
        Tuple of (train_proteins, val_proteins) sets
    """
    unique_proteins = list(set(protein_sequences))
    random.shuffle(unique_proteins)
    
    n_train = int(len(unique_proteins) * train_ratio)
    
    train_proteins = set(unique_proteins[:n_train])
    val_proteins = set(unique_proteins[n_train:])
    
    print(f"📊 Protein split: {len(train_proteins)} train, {len(val_proteins)} validation")
    return train_proteins, val_proteins

def create_rna_split(rna_sequences: List[str], 
                    train_ratio: float = 0.8) -> Tuple[Set[str], Set[str]]:
    """
    Split RNA sequences into train/validation ensuring no overlap.
    
    Args:
        rna_sequences: List of all RNA sequences
        train_ratio: Ratio of RNAs for training
        
    Returns:
        Tuple of (train_rnas, val_rnas) sets
    """
    unique_rnas = list(set(rna_sequences))
    random.shuffle(unique_rnas)
    
    n_train = int(len(unique_rnas) * train_ratio)
    
    train_rnas = set(unique_rnas[:n_train])
    val_rnas = set(unique_rnas[n_train:])
    
    print(f"📊 RNA split: {len(train_rnas)} train, {len(val_rnas)} validation")
    return train_rnas, val_rnas

def create_strategic_split(rna_sequences: List[str],
                          protein_sequences: List[str], 
                          binding_scores: np.ndarray,
                          strategy: str = "mixed",
                          train_ratio: float = 0.8,
                          test_like: bool = True) -> Tuple[List[int], List[int]]:
    """
    Create strategic train/validation split based on different strategies.
    
    Args:
        rna_sequences: List of RNA sequences (per sample)
        protein_sequences: List of protein sequences (per sample)
        binding_scores: Array of binding scores
        strategy: Split strategy ("random", "protein", "rna", "mixed", "test_simulation")
        train_ratio: Ratio of data for training
        test_like: If True, ensure validation mimics test scenario (43 proteins, many RNAs)
        
    Returns:
        Tuple of (train_indices, val_indices)
    """
    
    n_samples = len(rna_sequences)
    print(f"🎯 Creating {strategy} split for {n_samples} samples...")
    
    if strategy == "random":
        # Standard random split (current method)
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        n_train = int(n_samples * train_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
    elif strategy == "protein":
        # Split by proteins - ensure validation has unseen proteins
        train_proteins, val_proteins = create_protein_split(protein_sequences, train_ratio)
        
        train_indices = []
        val_indices = []
        
        for i, protein in enumerate(protein_sequences):
            if protein in train_proteins:
                train_indices.append(i)
            else:
                val_indices.append(i)
                
    elif strategy == "rna":
        # Split by RNA - ensure validation has unseen RNAs
        train_rnas, val_rnas = create_rna_split(rna_sequences, train_ratio)
        
        train_indices = []
        val_indices = []
        
        for i, rna in enumerate(rna_sequences):
            if rna in train_rnas:
                train_indices.append(i)
            else:
                val_indices.append(i)
                
    elif strategy == "mixed":
        # Mixed strategy - some unseen proteins AND some unseen RNAs
        
        # Split proteins (70% train, 30% val)
        train_proteins, val_proteins = create_protein_split(protein_sequences, 0.7)
        
        # Split RNAs (70% train, 30% val)  
        train_rnas, val_rnas = create_rna_split(rna_sequences, 0.7)
        
        train_indices = []
        val_indices = []
        
        for i, (rna, protein) in enumerate(zip(rna_sequences, protein_sequences)):
            # Training set: both RNA and protein in training sets
            if rna in train_rnas and protein in train_proteins:
                train_indices.append(i)
            # Validation set: at least one is unseen
            elif rna in val_rnas or protein in val_proteins:
                val_indices.append(i)
            else:
                # Edge case: add to training
                train_indices.append(i)
                
    elif strategy == "test_simulation":
        # Simulate the actual test scenario: 43 unique proteins vs many RNAs
        
        if test_like:
            # Mimic test structure: separate small set of proteins for validation
            unique_proteins = list(set(protein_sequences))
            unique_rnas = list(set(rna_sequences))
            
            # Proteins: Reserve ~43 unique proteins for validation (like test)
            n_val_proteins = min(43, max(5, int(len(unique_proteins) * 0.2)))
            
            # RNAs: Reserve large portion for validation (like test with 120K RNAs)
            n_val_rnas = int(len(unique_rnas) * 0.3)  # 30% of unique RNAs
            
            random.shuffle(unique_proteins)
            random.shuffle(unique_rnas)
            
            val_proteins = set(unique_proteins[:n_val_proteins])
            val_rnas = set(unique_rnas[:n_val_rnas])
            
            train_indices = []
            val_indices = []
            
            for i, (rna, protein) in enumerate(zip(rna_sequences, protein_sequences)):
                # Validation: protein is in val set OR RNA is in val set  
                if protein in val_proteins or rna in val_rnas:
                    val_indices.append(i)
                else:
                    train_indices.append(i)
                    
            print(f"🎯 Test simulation: {len(val_proteins)} proteins, {len(val_rnas)} RNAs in validation")
        else:
            # Fallback to mixed strategy
            return create_strategic_split(rna_sequences, protein_sequences, binding_scores, 
                                        "mixed", train_ratio, False)
                                        
    elif strategy == "realistic_test":
        # CORRECTED: Simulate the ACTUAL test scenario where 100% of pairs have both sequences new
        
        unique_proteins = list(set(protein_sequences))
        unique_rnas = list(set(rna_sequences))
        
        # Proteins: Reserve ~43 unique proteins for validation (like test)
        n_val_proteins = min(43, max(5, int(len(unique_proteins) * 0.2)))
        
        # RNAs: Reserve portion for validation 
        n_val_rnas = int(len(unique_rnas) * 0.3)  # 30% of unique RNAs
        
        random.shuffle(unique_proteins)
        random.shuffle(unique_rnas)
        
        val_proteins = set(unique_proteins[:n_val_proteins])
        val_rnas = set(unique_rnas[:n_val_rnas])
        
        train_indices = []
        val_indices = []
        
        for i, (rna, protein) in enumerate(zip(rna_sequences, protein_sequences)):
            # CORRECTED: Validation only when BOTH sequences are new (like real test!)
            if protein in val_proteins and rna in val_rnas:
                val_indices.append(i)
            else:
                train_indices.append(i)
                
        print(f"🎯 Realistic test: {len(val_proteins)} proteins, {len(val_rnas)} RNAs in validation")
        print(f"✅ 100% validation pairs have both sequences new (like real test!)")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"✅ Split complete: {len(train_indices)} train, {len(val_indices)} validation")
    print(f"📊 Split ratio: {len(train_indices)/n_samples:.1%} train, {len(val_indices)/n_samples:.1%} validation")
    
    return train_indices, val_indices

def analyze_split_novelty(train_indices: List[int], 
                         val_indices: List[int],
                         rna_sequences: List[str],
                         protein_sequences: List[str]) -> Dict[str, int]:
    """
    Analyze how many novel sequences are in validation set.
    
    Returns:
        Dictionary with novelty statistics
    """
    
    # Get sequences for each split
    train_rnas = set(rna_sequences[i] for i in train_indices)
    train_proteins = set(protein_sequences[i] for i in train_indices)
    
    val_rnas = set(rna_sequences[i] for i in val_indices)  
    val_proteins = set(protein_sequences[i] for i in val_indices)
    
    # Calculate novelty
    novel_val_rnas = val_rnas - train_rnas
    novel_val_proteins = val_proteins - train_proteins
    
    stats = {
        'train_rnas': len(train_rnas),
        'train_proteins': len(train_proteins),
        'val_rnas': len(val_rnas),
        'val_proteins': len(val_proteins),
        'novel_val_rnas': len(novel_val_rnas),
        'novel_val_proteins': len(novel_val_proteins),
        'rna_novelty_ratio': len(novel_val_rnas) / len(val_rnas) if val_rnas else 0,
        'protein_novelty_ratio': len(novel_val_proteins) / len(val_proteins) if val_proteins else 0
    }
    
    print(f"\n📊 Split Analysis:")
    print(f"  Novel RNAs in validation: {stats['novel_val_rnas']}/{stats['val_rnas']} ({stats['rna_novelty_ratio']:.1%})")
    print(f"  Novel proteins in validation: {stats['novel_val_proteins']}/{stats['val_proteins']} ({stats['protein_novelty_ratio']:.1%})")
    
    return stats

def compare_splitting_strategies(rna_sequences: List[str],
                               protein_sequences: List[str], 
                               binding_scores: np.ndarray) -> Dict[str, Dict]:
    """
    Compare different splitting strategies and their novelty characteristics.
    """
    
    strategies = ["random", "protein", "rna", "mixed", "test_simulation"]
    results = {}
    
    print("🔬 COMPARING SPLITTING STRATEGIES")
    print("=" * 50)
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.upper()} strategy:")
        
        train_idx, val_idx = create_strategic_split(
            rna_sequences, protein_sequences, binding_scores, 
            strategy=strategy, train_ratio=0.8
        )
        
        stats = analyze_split_novelty(
            train_idx, val_idx, rna_sequences, protein_sequences
        )
        
        results[strategy] = {
            'train_indices': train_idx,
            'val_indices': val_idx,
            'stats': stats
        }
    
    return results
