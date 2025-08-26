import numpy as np
import torch
from typing import Dict, List, Tuple


def encode_rna_sequence(sequence: str, max_length: int = None) -> np.ndarray:
    """
    One-hot encode RNA sequence.
    
    Args:
        sequence: RNA sequence string (must contain only A, C, G, U)
        max_length: Maximum sequence length for padding
        
    Returns:
        One-hot encoded sequence as numpy array
    """
    # RNA nucleotides mapping (5 dimensions: A, C, G, U + padding)
    nucleotide_mapping = {
        'A': 0, 'C': 1, 'G': 2, 'U': 3
    }
    
    # Convert sequence to indices
    sequence_indices = []
    for nucleotide in sequence.upper():
        if nucleotide in nucleotide_mapping:
            sequence_indices.append(nucleotide_mapping[nucleotide])
        else:
            # This should not happen if validation is working correctly
            raise ValueError(f"Invalid RNA nucleotide: {nucleotide}. Only A, C, G, U are allowed.")
    
    # Pad or truncate to max_length if specified
    if max_length:
        if len(sequence_indices) > max_length:
            sequence_indices = sequence_indices[:max_length]
        else:
            sequence_indices.extend([4] * (max_length - len(sequence_indices)))  # Pad with index 4
    
    # One-hot encode
    num_nucleotides = 5  # A, C, G, U + padding dimension
    one_hot = np.zeros((len(sequence_indices), num_nucleotides))
    for i, idx in enumerate(sequence_indices):
        one_hot[i, idx] = 1
    
    return one_hot


def encode_protein_sequence(sequence: str, max_length: int = None) -> np.ndarray:
    """
    One-hot encode protein sequence.
    
    Args:
        sequence: Protein sequence string
        max_length: Maximum sequence length for padding
        
    Returns:
        One-hot encoded sequence as numpy array
    """
    # Standard amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_mapping = {aa: i for i, aa in enumerate(amino_acids)}
    aa_mapping['X'] = 20  # Unknown amino acid
    
    # Convert sequence to indices
    sequence_indices = []
    for aa in sequence.upper():
        if aa in aa_mapping:
            sequence_indices.append(aa_mapping[aa])
        else:
            sequence_indices.append(aa_mapping['X'])
    
    # Pad or truncate to max_length if specified
    if max_length:
        if len(sequence_indices) > max_length:
            sequence_indices = sequence_indices[:max_length]
        else:
            sequence_indices.extend([20] * (max_length - len(sequence_indices)))  # Pad with 'X'
    
    # One-hot encode
    num_amino_acids = 21  # 20 standard + unknown
    one_hot = np.zeros((len(sequence_indices), num_amino_acids))
    for i, idx in enumerate(sequence_indices):
        one_hot[i, idx] = 1
    
    return one_hot


def find_optimal_sequence_lengths(rna_sequences: List[str], 
                                protein_sequences: List[str],
                                percentile: float = 95) -> Tuple[int, int]:
    """
    Find optimal sequence lengths based on data distribution.
    
    Args:
        rna_sequences: List of RNA sequences
        protein_sequences: List of protein sequences
        percentile: Percentile to use for determining max length
        
    Returns:
        Tuple of (rna_max_length, protein_max_length)
    """
    rna_lengths = [len(seq) for seq in rna_sequences]
    protein_lengths = [len(seq) for seq in protein_sequences]
    
    rna_max_length = int(np.percentile(rna_lengths, percentile))
    protein_max_length = int(np.percentile(protein_lengths, percentile))
    
    print(f"RNA sequences - Min: {min(rna_lengths)}, Max: {max(rna_lengths)}, "
          f"{percentile}th percentile: {rna_max_length}")
    print(f"Protein sequences - Min: {min(protein_lengths)}, Max: {max(protein_lengths)}, "
          f"{percentile}th percentile: {protein_max_length}")
    
    return rna_max_length, protein_max_length


def normalize_binding_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize binding scores to [0, 1] range.
    
    Args:
        scores: Raw binding scores
        
    Returns:
        Normalized scores
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score - min_score == 0:
        return np.zeros_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


def prepare_sequences_for_training(rna_sequences: List[str], 
                                 protein_sequences: List[str],
                                 rna_max_length: int = None,
                                 protein_max_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare sequences for training by encoding and converting to tensors.
    
    Args:
        rna_sequences: List of RNA sequences
        protein_sequences: List of protein sequences  
        rna_max_length: Maximum RNA sequence length
        protein_max_length: Maximum protein sequence length
        
    Returns:
        Tuple of (rna_tensors, protein_tensors)
    """
    # Encode RNA sequences
    rna_encoded = []
    for seq in rna_sequences:
        encoded = encode_rna_sequence(seq, rna_max_length)
        rna_encoded.append(encoded)
    
    # Encode protein sequences
    protein_encoded = []
    for seq in protein_sequences:
        encoded = encode_protein_sequence(seq, protein_max_length)
        protein_encoded.append(encoded)
    
    # Convert to tensors
    rna_tensor = torch.FloatTensor(np.array(rna_encoded))
    protein_tensor = torch.FloatTensor(np.array(protein_encoded))
    
    return rna_tensor, protein_tensor
