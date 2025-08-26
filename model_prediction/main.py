#!/usr/bin/env python3
"""
Main prediction interface for RNA-Protein binding prediction.

Usage: python main.py <output_file> <rbp_file> <rna_file>

This script takes:
- output_file: Path where results will be saved
- rbp_file: File containing RBP protein sequences (one per line)
- rna_file: File containing RNA sequences (one per line)

For each RBP protein, creates a file named after the protein index (RBP1.txt, RBP2.txt, etc.)
containing binding scores for all RNA sequences in the same order as the RNA file.
"""

import argparse
import os
import sys
import torch
import numpy as np
import time
import pickle
import hashlib
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.siamese_protein_bert import SiameseProteinBERT
from data.preprocessing import encode_rna_sequence
from transformers import AutoTokenizer, AutoModel

class ProteinBertEmbedder:
    """Generate ProteinBERT embeddings for protein sequences with caching support."""
    
    def __init__(self, device='cpu', cache_file: Optional[str] = None):
        self.device = device
        self.cache = {}
        
        # Load cache if provided
        if cache_file and os.path.exists(cache_file):
            print(f"Loading protein embedding cache from: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            print(f"Loaded {len(self.cache)} cached protein embeddings")
        elif cache_file:
            print(f"Cache file not found: {cache_file} - will compute embeddings on demand")
        
        # Initialize ProteinBERT model
        print("Loading ProteinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)
        self.model.eval()
        print("ProteinBERT model loaded!")
        
    def embed_protein(self, sequence: str, max_length: int = 512) -> np.ndarray:
        """Generate embedding for a single protein sequence with caching."""
        # Check cache first
        if sequence in self.cache:
            return self.cache[sequence]
        
        # Cache miss - compute embedding using ProteinBERT
        # Add spaces between amino acids for ProteinBERT
        spaced_sequence = ' '.join(list(sequence))
        
        # Tokenize and encode
        inputs = self.tokenizer(
            spaced_sequence,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        embedding = embeddings.cpu().numpy().squeeze()
        
        # Cache the computed embedding for future use
        self.cache[sequence] = embedding
        
        return embedding

class RNAProteinPredictor:
    """RNA-Protein binding prediction using trained Siamese model."""
    
    def __init__(self, model_path: str, device='cpu', cache_file: Optional[str] = None):
        self.device = torch.device(device)
        
        # Load the trained Siamese model
        self.model = SiameseProteinBERT(
            rna_input_size=5,     # Keep 5: compressed model was trained with 5 dimensions
            rna_hidden_size=140,  # Updated to match compressed model
            num_layers=1,
            dropout=0.2,
            protein_embedding_dim=1024,
            embedding_dim=280,    # Updated to match compressed model
            temperature=0.1
        ).to(self.device)
        
        # Load model weights (weights_only=False for compatibility with older PyTorch versions)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        # Initialize protein embedder with cache support
        self.protein_embedder = ProteinBertEmbedder(device, cache_file)
        
    def encode_rna(self, sequence: str, max_length: int = 60) -> torch.Tensor:
        """Encode RNA sequence to one-hot tensor."""
        encoded = encode_rna_sequence(sequence, max_length)
        return torch.FloatTensor(encoded).unsqueeze(0).to(self.device)
    
    def predict_binding_scores(self, rna_sequences: List[str], protein_sequence: str) -> List[float]:
        """Predict binding scores between RNA sequences and a protein."""
        
        # Generate protein embedding (with caching)
        if protein_sequence in self.protein_embedder.cache:
            print(f"Using cached embedding for protein sequence... ⚡")
            protein_embedding = self.protein_embedder.cache[protein_sequence]
        else:
            print(f"Computing new embedding for protein sequence...")
            protein_embedding = self.protein_embedder.embed_protein(protein_sequence)
        protein_tensor = torch.FloatTensor(protein_embedding).unsqueeze(0).to(self.device)
        
        print(f"Predicting binding scores for {len(rna_sequences)} RNA sequences...")
        start_time = time.time()
        
        # BATCH PROCESSING for massive speedup
        batch_size = 1000
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(rna_sequences), batch_size):
                batch_end = min(i + batch_size, len(rna_sequences))
                batch_rnas = rna_sequences[i:batch_end]
                
                # Process batch of RNAs
                batch_rna_tensors = []
                for rna_seq in batch_rnas:
                    rna_tensor = self.encode_rna(rna_seq)
                    batch_rna_tensors.append(rna_tensor)
                
                # Stack into single batch tensor
                batch_rna_tensor = torch.cat(batch_rna_tensors, dim=0)
                
                # Repeat protein tensor to match batch size
                batch_protein_tensor = protein_tensor.repeat(len(batch_rnas), 1)
                
                # Predict scores for entire batch
                batch_scores = self.model(batch_rna_tensor, batch_protein_tensor, mode='inference')
                
                # Extract individual scores
                batch_scores_list = batch_scores.cpu().numpy().flatten()
                scores.extend(batch_scores_list)
                
                if (i + batch_size) % 10000 == 0 or batch_end == len(rna_sequences):
                    print(f"Processed {batch_end}/{len(rna_sequences)} sequences...")
        
        prediction_time = time.time() - start_time
        print(f"Prediction completed in {prediction_time:.2f} seconds")
        print(f"Average time per prediction: {prediction_time/len(rna_sequences)*1000:.2f} ms")
        
        return scores

def load_sequences(file_path: str) -> List[str]:
    """Load sequences from a text file (one per line)."""
    sequences = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            seq = line.strip()
            if seq:
                sequences.append((line_num, seq))
    return sequences

def validate_rna_sequence(sequence: str, line_num: int) -> bool:
    """Validate RNA sequence contains only valid nucleotides."""
    valid_rna_chars = set('ACGUacgu')  # A, C, G, U only
    invalid_chars = set(sequence) - valid_rna_chars
    
    if invalid_chars:
        print(f"Warning: Line {line_num} contains invalid RNA characters: {invalid_chars}")
        print(f"  Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        print(f"  Valid characters: A, C, G, U (case insensitive)")
        return False
    return True

def validate_protein_sequence(sequence: str, line_num: int) -> bool:
    """Validate protein sequence contains only valid amino acids."""
    valid_protein_chars = set('ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwyXx')  # 20 standard + X (unknown)
    invalid_chars = set(sequence) - valid_protein_chars
    
    if invalid_chars:
        print(f"Warning: Line {line_num} contains invalid protein characters: {invalid_chars}")
        print(f"  Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        print(f"  Valid characters: ACDEFGHIKLMNPQRSTVWY, X (case insensitive)")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='RNA-Protein binding prediction')
    parser.add_argument('output_file', help='Output file base name (creates <ofile> for 1 protein, <ofile>1, <ofile>2, etc. for multiple)')
    parser.add_argument('rbp_file', help='File containing RBP protein sequences')
    parser.add_argument('rna_file', help='File containing RNA sequences')
    parser.add_argument('--model_path', default='models/best_model_enhanced.pt', 
                       help='Path to trained model')

    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run predictions on')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.rbp_file):
        print(f"Error: RBP file '{args.rbp_file}' not found")
        sys.exit(1)
        
    if not os.path.exists(args.rna_file):
        print(f"Error: RNA file '{args.rna_file}' not found")
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found")
        sys.exit(1)
    
    # Load and validate RNA sequences
    print("Loading RNA sequences...")
    rna_data = load_sequences(args.rna_file)
    print(f"Loaded {len(rna_data)} RNA sequences")
    
    # Validate that we have RNA sequences
    if len(rna_data) == 0:
        print(f"Error: No RNA sequences found in '{args.rna_file}'")
        print("Please ensure the file contains RNA sequences (one per line)")
        sys.exit(1)
    
    # Validate RNA sequences and extract valid ones
    print("Validating RNA sequences...")
    rna_sequences = []
    invalid_rna_count = 0
    for line_num, sequence in rna_data:
        if validate_rna_sequence(sequence, line_num):
            rna_sequences.append(sequence)
        else:
            invalid_rna_count += 1
    
    if invalid_rna_count > 0:
        print(f"Warning: {invalid_rna_count} RNA sequences contain invalid characters")
        print("Only valid sequences will be processed")
    
    if len(rna_sequences) == 0:
        print(f"Error: No valid RNA sequences found in '{args.rna_file}'")
        sys.exit(1)
    
    print(f"Processing {len(rna_sequences)} valid RNA sequences")
    
    # Load and validate RBP protein sequences
    print("Loading RBP protein sequences...")
    rbp_data = load_sequences(args.rbp_file)
    print(f"Loaded {len(rbp_data)} RBP sequences")
    
    # Validate that we have RBP sequences
    if len(rbp_data) == 0:
        print(f"Error: No RBP sequences found in '{args.rbp_file}'")
        print("Please ensure the file contains protein sequences (one per line)")
        sys.exit(1)
    
    # Validate protein sequences and extract valid ones
    print("Validating protein sequences...")
    rbp_sequences = []
    invalid_protein_count = 0
    for line_num, sequence in rbp_data:
        if validate_protein_sequence(sequence, line_num):
            rbp_sequences.append(sequence)
        else:
            invalid_protein_count += 1
    
    if invalid_protein_count > 0:
        print(f"Warning: {invalid_protein_count} protein sequences contain invalid characters")
        print("Only valid sequences will be processed")
    
    if len(rbp_sequences) == 0:
        print(f"Error: No valid protein sequences found in '{args.rbp_file}'")
        sys.exit(1)
    
    print(f"Processing {len(rbp_sequences)} valid protein sequences")
    
    # Initialize predictor with automatic cache detection
    print(f"Initializing predictor on {args.device}...")
    cache_file = 'protein_embeddings_cache.pkl'  # Always look for this file
    predictor = RNAProteinPredictor(args.model_path, args.device, cache_file)
    
    # Determine output file naming pattern
    base_name = Path(args.output_file).stem  # Get filename without extension
    base_dir = Path(args.output_file).parent if Path(args.output_file).parent != Path('.') else Path('.')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    total_start_time = time.time()
    
    # Process each RBP protein
    for rbp_idx, rbp_seq in enumerate(rbp_sequences):
        print(f"\n{'='*60}")
        print(f"Processing RBP {rbp_idx + 1}/{len(rbp_sequences)}")
        print(f"{'='*60}")
        
        # Predict binding scores for this RBP against all RNAs
        scores = predictor.predict_binding_scores(rna_sequences, rbp_seq)
        
        # Determine output filename based on number of proteins
        if len(rbp_sequences) == 1:
            # Single protein: use the exact output file name given
            output_file = Path(args.output_file)
        else:
            # Multiple proteins: use base name with index (ofile1, ofile2, etc.)
            output_file = base_dir / f"{base_name}{rbp_idx + 1}.txt"
        
        with open(output_file, 'w') as f:
            for score in scores:
                f.write(f"{score:.6f}\n")
        
        print(f"Saved {len(scores)} scores to {output_file}")
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"PREDICTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processed {len(rbp_sequences)} RBPs × {len(rna_sequences)} RNAs = {len(rbp_sequences) * len(rna_sequences):,} predictions")
    
    # Safe division for average time
    if len(rbp_sequences) > 0:
        print(f"Average time per RBP: {total_time/len(rbp_sequences):.2f} seconds")
    else:
        print("No RBPs processed")
    
    # Print file creation summary
    if len(rbp_sequences) == 1:
        print(f"File created: {args.output_file}")
    elif len(rbp_sequences) > 1:
        print(f"Files created: {base_name}1.txt through {base_name}{len(rbp_sequences)}.txt")
    else:
        print("No output files created")

if __name__ == "__main__":
    main()
