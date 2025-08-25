import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from .preprocessing import (
    encode_rna_sequence, 
    encode_protein_sequence, 
    normalize_binding_scores,
    find_optimal_sequence_lengths
)


class RNAProteinDataset(Dataset):
    """Dataset for RNA-Protein binding prediction."""
    
    def __init__(self, 
                 rna_sequences: List[str],
                 protein_sequences: List[str], 
                 binding_scores: np.ndarray,
                 rna_max_length: int = None,
                 protein_max_length: int = None,
                 normalize_scores: bool = True,
                 protein_embedding_lookup: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize dataset.
        
        Args:
            rna_sequences: List of RNA sequences
            protein_sequences: List of protein sequences (one per RNA sequence)
            binding_scores: Array of binding scores
            rna_max_length: Maximum RNA sequence length
            protein_max_length: Maximum protein sequence length  
            normalize_scores: Whether to normalize binding scores
        """
        self.rna_sequences = rna_sequences
        self.protein_sequences = protein_sequences
        self.binding_scores = binding_scores.copy()
        self.rna_max_length = rna_max_length
        self.protein_max_length = protein_max_length
        self.protein_embedding_lookup = protein_embedding_lookup or {}
        
        # Normalize binding scores if requested
        if normalize_scores:
            self.binding_scores = normalize_binding_scores(self.binding_scores)
        
        # Pre-encode sequences for efficiency
        self._encode_sequences()
    
    def _encode_sequences(self):
        """Pre-encode all sequences."""
        self.rna_encoded = []
        self.protein_encoded = []
        
        print("Encoding sequences...")
        for i, (rna_seq, protein_seq) in enumerate(zip(self.rna_sequences, self.protein_sequences)):
            if i % 10000 == 0:
                print(f"Encoded {i}/{len(self.rna_sequences)} sequences")
            
            rna_enc = encode_rna_sequence(rna_seq, self.rna_max_length)
            # If embedding lookup provided, use vector; else one-hot encode sequence
            if self.protein_embedding_lookup:
                if protein_seq in self.protein_embedding_lookup:
                    protein_enc = self.protein_embedding_lookup[protein_seq]
                else:
                    # Fallback to one-hot if embedding not found
                    protein_enc = encode_protein_sequence(protein_seq, self.protein_max_length)
            else:
                protein_enc = encode_protein_sequence(protein_seq, self.protein_max_length)
            
            self.rna_encoded.append(rna_enc)
            self.protein_encoded.append(protein_enc)
        
        print(f"Encoding complete. Total sequences: {len(self.rna_encoded)}")
    
    def __len__(self):
        return len(self.rna_sequences)
    
    def __getitem__(self, idx):
        rna_tensor = torch.FloatTensor(self.rna_encoded[idx])
        protein_arr = self.protein_encoded[idx]
        protein_tensor = torch.FloatTensor(protein_arr)
        score_tensor = torch.FloatTensor([self.binding_scores[idx]])
        
        return {
            'rna': rna_tensor,
            'protein': protein_tensor,
            'score': score_tensor
        }


def load_training_data(data_dir: str, subset_size: int = None) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Load training data from files.
    
    Args:
        data_dir: Directory containing data files
        subset_size: If specified, only load this many RNA sequences for testing
        
    Returns:
        Tuple of (rna_sequences, protein_sequences, binding_scores)
    """
    print("Loading training data...")
    
    # Load RNA sequences
    rna_file = os.path.join(data_dir, 'training_seqs.txt')
    with open(rna_file, 'r') as f:
        rna_sequences = [line.strip() for line in f.readlines()]
    
    # Load protein sequences  
    protein_file = os.path.join(data_dir, 'training_RBPs2.txt')
    with open(protein_file, 'r') as f:
        protein_sequences = [line.strip() for line in f.readlines()]
    
    # Load binding scores
    scores_file = os.path.join(data_dir, 'training_data2.txt')
    binding_scores = []
    
    print("Loading binding scores...")
    with open(scores_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 50000 == 0:
                print(f"Loaded {line_num} score lines")
            
            # Parse the line - each line contains scores for one RNA sequence across all proteins
            scores_str = line.strip()
            if scores_str:
                # Split by whitespace and convert to float
                scores = [float(x) for x in scores_str.split()]
                binding_scores.extend(scores)
    
    print(f"Loaded {len(binding_scores)} total binding scores")
    print(f"Loaded {len(rna_sequences)} RNA sequences")
    print(f"Loaded {len(protein_sequences)} protein sequences")
    
    # Create protein-RNA pairs
    # Each RNA sequence is paired with each protein
    rna_protein_pairs = []
    protein_rna_pairs = []
    final_scores = []
    
    scores_per_rna = len(protein_sequences)
    
    for rna_idx, rna_seq in enumerate(rna_sequences):
        if subset_size and rna_idx >= subset_size:
            break
            
        start_idx = rna_idx * scores_per_rna
        end_idx = start_idx + scores_per_rna
        
        if end_idx <= len(binding_scores):
            rna_scores = binding_scores[start_idx:end_idx]
            
            for protein_idx, protein_seq in enumerate(protein_sequences):
                rna_protein_pairs.append(rna_seq)
                protein_rna_pairs.append(protein_seq)
                final_scores.append(rna_scores[protein_idx])
    
    print(f"Created {len(rna_protein_pairs)} RNA-protein pairs")
    
    return rna_protein_pairs, protein_rna_pairs, np.array(final_scores)


def load_test_data(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load test data from files.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (rna_sequences, protein_sequences)
    """
    print("Loading test data...")
    
    # Load test RNA sequences
    rna_file = os.path.join(data_dir, 'test_seqs.txt')
    with open(rna_file, 'r') as f:
        rna_sequences = [line.strip() for line in f.readlines()]
    
    # Load test protein sequences
    protein_file = os.path.join(data_dir, 'test_RBPs2.txt')
    with open(protein_file, 'r') as f:
        protein_sequences = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(rna_sequences)} test RNA sequences")
    print(f"Loaded {len(protein_sequences)} test protein sequences")
    
    return rna_sequences, protein_sequences


def create_data_loaders(rna_sequences: List[str],
                       protein_sequences: List[str], 
                       binding_scores: np.ndarray,
                       batch_size: int = 32,
                       validation_split: float = 0.2,
                       rna_max_length: int = None,
                       protein_max_length: int = None,
                       num_workers: int = 0,
                       pin_memory: bool = False,
                       protein_embedding_lookup: Optional[Dict[str, np.ndarray]] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        rna_sequences: List of RNA sequences
        protein_sequences: List of protein sequences
        binding_scores: Array of binding scores
        batch_size: Batch size for data loaders
        validation_split: Fraction of data to use for validation
        rna_max_length: Maximum RNA sequence length
        protein_max_length: Maximum protein sequence length
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    # Determine sequence lengths if not provided
    if rna_max_length is None or protein_max_length is None:
        rna_max_length, protein_max_length = find_optimal_sequence_lengths(
            rna_sequences, protein_sequences, percentile=95
        )
    
    # Create full dataset
    dataset = RNAProteinDataset(
        rna_sequences=rna_sequences,
        protein_sequences=protein_sequences,
        binding_scores=binding_scores,
        rna_max_length=rna_max_length,
        protein_max_length=protein_max_length,
        normalize_scores=True,
        protein_embedding_lookup=protein_embedding_lookup
    )
    
    # Split into train and validation
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

    # Removed the extra code that was causing syntax error
    
    return train_loader, val_loader, rna_max_length, protein_max_length
