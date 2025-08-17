import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
from .dataset import RNAProteinDataset, load_training_data


class SiameseRNAProteinDataset(Dataset):
    """
    Dataset for Siamese contrastive learning on RNA-protein binding prediction.
    
    Generates pairs of samples:
    - Positive pairs: Both samples have similar binding affinities
    - Negative pairs: Samples with different binding affinities
    
    This approach helps the model learn to distinguish between binding and non-binding pairs.
    """
    
    def __init__(self,
                 rna_sequences: List[str],
                 protein_sequences: List[str],
                 binding_affinities: np.ndarray,
                 protein_embeddings: Dict[str, torch.Tensor],
                 max_rna_length: int = 60,
                 max_protein_length: int = 300,
                 positive_threshold: float = 0.7,
                 negative_threshold: float = 0.3,
                 pair_sampling_ratio: float = 1.0,
                 hard_negative_ratio: float = 0.3):
        """
        Args:
            rna_sequences: List of RNA sequences
            protein_sequences: List of protein sequences  
            binding_affinities: Array of binding affinity values (normalized 0-1)
            protein_embeddings: Dict mapping protein sequences to embeddings
            max_rna_length: Maximum RNA sequence length
            max_protein_length: Maximum protein sequence length
            positive_threshold: Threshold for considering pairs as positive (similar binding)
            negative_threshold: Threshold for considering pairs as negative (different binding)
            pair_sampling_ratio: Ratio of pairs to generate relative to dataset size
            hard_negative_ratio: Ratio of hard negatives (close but different) to include
        """
        
        self.rna_sequences = rna_sequences
        self.protein_sequences = protein_sequences
        self.binding_affinities = binding_affinities
        self.protein_embeddings = protein_embeddings
        self.max_rna_length = max_rna_length
        self.max_protein_length = max_protein_length
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.pair_sampling_ratio = pair_sampling_ratio
        self.hard_negative_ratio = hard_negative_ratio
        
        # RNA encoding
        self.rna_vocab = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        
        # Pre-encode RNA sequences
        self.encoded_rnas = [self._encode_rna(seq) for seq in rna_sequences]
        
        # Generate pairs
        self.pairs = self._generate_pairs()
        
        print(f"Generated {len(self.pairs)} pairs for Siamese training")
        print(f"Positive pairs: {sum(1 for _, _, label in self.pairs if label == 1)}")
        print(f"Negative pairs: {sum(1 for _, _, label in self.pairs if label == 0)}")
    
    def _encode_rna(self, sequence: str) -> torch.Tensor:
        """Encode RNA sequence to tensor."""
        encoded = [self.rna_vocab.get(nucleotide, 4) for nucleotide in sequence]
        
        # Pad or truncate
        if len(encoded) > self.max_rna_length:
            encoded = encoded[:self.max_rna_length]
        else:
            encoded.extend([4] * (self.max_rna_length - len(encoded)))
        
        # One-hot encode
        tensor = torch.zeros(self.max_rna_length, 5)
        for i, nuc_idx in enumerate(encoded):
            tensor[i, nuc_idx] = 1.0
        
        return tensor
    
    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Generate positive and negative pairs based on binding affinity similarity.
        
        Returns:
            List of (idx1, idx2, label) tuples where label=1 for positive, 0 for negative
        """
        pairs = []
        n_samples = len(self.binding_affinities)
        n_pairs_target = min(int(n_samples * self.pair_sampling_ratio), n_samples * 2)  # Cap pairs
        
        print(f"Generating {n_pairs_target} pairs from {n_samples} samples...")
        
        # Pre-compute similarity groups for efficiency
        sorted_indices = np.argsort(self.binding_affinities)
        sorted_affinities = self.binding_affinities[sorted_indices]
        
        # Create affinity bins for faster lookup
        n_bins = min(100, n_samples // 10)  # Adaptive binning
        affinity_bins = np.digitize(self.binding_affinities, 
                                   np.linspace(0, 1, n_bins))
        
        # Generate positive pairs (similar binding affinities)
        n_positive_target = n_pairs_target // 2
        positive_pairs = []
        
        attempts = 0
        max_attempts = n_positive_target * 3  # Prevent infinite loops
        
        while len(positive_pairs) < n_positive_target and attempts < max_attempts:
            idx1 = random.randint(0, n_samples - 1)
            target_bin = affinity_bins[idx1]
            
            # Find samples in same or adjacent bins
            candidates = []
            for bin_offset in [0, -1, 1]:
                target_bin_search = target_bin + bin_offset
                if 0 <= target_bin_search < n_bins:
                    bin_candidates = np.where(affinity_bins == target_bin_search)[0]
                    for cand in bin_candidates:
                        if cand != idx1:
                            diff = abs(self.binding_affinities[cand] - self.binding_affinities[idx1])
                            if diff <= (1 - self.positive_threshold):
                                candidates.append(cand)
            
            if candidates:
                idx2 = random.choice(candidates)
                positive_pairs.append((idx1, idx2, 1))
            
            attempts += 1
        
        # Generate negative pairs more efficiently
        n_negative_target = n_pairs_target - len(positive_pairs)
        negative_pairs = []
        
        attempts = 0
        max_attempts = n_negative_target * 3
        
        while len(negative_pairs) < n_negative_target and attempts < max_attempts:
            idx1 = random.randint(0, n_samples - 1)
            idx2 = random.randint(0, n_samples - 1)
            
            if idx1 != idx2:
                diff = abs(self.binding_affinities[idx1] - self.binding_affinities[idx2])
                if diff >= self.negative_threshold:
                    negative_pairs.append((idx1, idx2, 0))
            
            attempts += 1
        
        # Combine and shuffle
        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)
        
        print(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a pair of samples for contrastive learning.
        
        Returns:
            Dict containing:
                - rna_seq1, rna_seq2: RNA sequences (max_rna_length, 5)
                - protein_emb1, protein_emb2: Protein embeddings (protein_embedding_dim,)
                - label: Contrastive label (1 for similar, 0 for dissimilar)
                - affinity1, affinity2: Original binding affinities
        """
        idx1, idx2, label = self.pairs[idx]
        
        # Get RNA sequences
        rna_seq1 = self.encoded_rnas[idx1]
        rna_seq2 = self.encoded_rnas[idx2]
        
        # Get protein embeddings
        protein_key1 = self.protein_sequences[idx1]
        protein_key2 = self.protein_sequences[idx2]
        
        protein_emb1 = self.protein_embeddings[protein_key1]
        protein_emb2 = self.protein_embeddings[protein_key2]
        
        # Ensure embeddings are 1D
        if protein_emb1.dim() > 1:
            protein_emb1 = protein_emb1.squeeze()
        if protein_emb2.dim() > 1:
            protein_emb2 = protein_emb2.squeeze()
        
        return {
            'rna_seq1': rna_seq1,
            'rna_seq2': rna_seq2,
            'protein_emb1': protein_emb1,
            'protein_emb2': protein_emb2,
            'label': torch.tensor(label, dtype=torch.float32),
            'affinity1': torch.tensor(self.binding_affinities[idx1], dtype=torch.float32),
            'affinity2': torch.tensor(self.binding_affinities[idx2], dtype=torch.float32)
        }


class HybridSiameseDataset(Dataset):
    """
    Hybrid dataset that combines regular regression training with contrastive learning.
    
    This allows the model to learn both:
    1. Direct binding affinity prediction 
    2. Relative similarity between RNA-protein pairs
    """
    
    def __init__(self,
                 rna_sequences: List[str],
                 protein_sequences: List[str], 
                 binding_affinities: np.ndarray,
                 protein_embeddings: Dict[str, torch.Tensor],
                 max_rna_length: int = 60,
                 contrastive_ratio: float = 0.5):
        """
        Args:
            contrastive_ratio: Ratio of contrastive samples vs regression samples
        """
        
        self.base_dataset = RNAProteinDataset(
            rna_sequences, protein_sequences, binding_affinities,
            protein_embeddings, max_rna_length
        )
        
        self.siamese_dataset = SiameseRNAProteinDataset(
            rna_sequences, protein_sequences, binding_affinities,
            protein_embeddings, max_rna_length
        )
        
        self.contrastive_ratio = contrastive_ratio
        self.total_length = len(self.base_dataset) + len(self.siamese_dataset)
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns either a regression sample or contrastive sample based on ratio.
        """
        if random.random() < self.contrastive_ratio:
            # Return contrastive sample
            siamese_idx = idx % len(self.siamese_dataset)
            sample = self.siamese_dataset[siamese_idx]
            sample['mode'] = torch.tensor(0, dtype=torch.long)  # 0 = contrastive
            return sample
        else:
            # Return regression sample  
            base_idx = idx % len(self.base_dataset)
            sample = self.base_dataset[base_idx]
            sample['mode'] = torch.tensor(1, dtype=torch.long)  # 1 = regression
            # Add dummy second samples for consistency
            sample['rna_seq2'] = sample['rna_seq']
            sample['protein_emb2'] = sample['protein_emb']
            sample['label'] = torch.tensor(0.0, dtype=torch.float32)  # Dummy
            return sample


def create_siamese_dataloaders(data_dir: str,
                             protein_embedding_path: str,
                             subset_size: Optional[int] = None,
                             batch_size: int = 32,
                             max_rna_length: int = 60,
                             max_protein_length: int = 300,
                             train_ratio: float = 0.8,
                             positive_threshold: float = 0.7,
                             negative_threshold: float = 0.3,
                             num_workers: int = 0) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for Siamese learning.
    
    Args:
        data_dir: Directory containing the data
        protein_embedding_path: Path to protein embeddings file
        subset_size: Optional subset size for testing
        batch_size: Batch size
        max_rna_length: Maximum RNA sequence length
        max_protein_length: Maximum protein sequence length  
        train_ratio: Ratio of data to use for training
        positive_threshold: Threshold for positive pairs
        negative_threshold: Threshold for negative pairs
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader
    """
    
    # Load data
    rna_seqs, protein_seqs, affinities = load_training_data(data_dir, subset_size)
    
    # Load protein embeddings
    protein_embeddings = torch.load(protein_embedding_path)
    
    # Split data
    n_total = len(rna_seqs)
    n_train = int(n_total * train_ratio)
    
    indices = list(range(n_total))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create train dataset
    train_rnas = [rna_seqs[i] for i in train_indices]
    train_proteins = [protein_seqs[i] for i in train_indices]
    train_affinities = affinities[train_indices]
    
    train_dataset = SiameseRNAProteinDataset(
        train_rnas, train_proteins, train_affinities, protein_embeddings,
        max_rna_length, max_protein_length, positive_threshold, negative_threshold
    )
    
    # Create validation dataset  
    val_rnas = [rna_seqs[i] for i in val_indices]
    val_proteins = [protein_seqs[i] for i in val_indices]
    val_affinities = affinities[val_indices]
    
    val_dataset = SiameseRNAProteinDataset(
        val_rnas, val_proteins, val_affinities, protein_embeddings,
        max_rna_length, max_protein_length, positive_threshold, negative_threshold
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader
