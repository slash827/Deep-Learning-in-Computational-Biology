import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BasicLSTM(nn.Module):
    """
    Basic bidirectional LSTM model for RNA-protein binding prediction.
    
    This model processes RNA and protein sequences separately using bidirectional LSTMs
    and then combines their representations to predict binding scores.
    """
    
    def __init__(self,
                 rna_input_size: int = 5,      # RNA nucleotides (A, U, G, C, N)
                 protein_input_size: int = 21,  # Amino acids (20 + unknown)
                 rna_hidden_size: int = 128,
                 protein_hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 fusion_hidden_size: int = 256):
        """
        Initialize the BasicLSTM model.
        
        Args:
            rna_input_size: Input size for RNA sequences (number of nucleotides)
            protein_input_size: Input size for protein sequences (number of amino acids)
            rna_hidden_size: Hidden size for RNA LSTM
            protein_hidden_size: Hidden size for protein LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            fusion_hidden_size: Hidden size for fusion layers
        """
        super(BasicLSTM, self).__init__()
        
        self.rna_hidden_size = rna_hidden_size
        self.protein_hidden_size = protein_hidden_size
        self.num_layers = num_layers
        
        # RNA sequence processing
        self.rna_lstm = nn.LSTM(
            input_size=rna_input_size,
            hidden_size=rna_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Protein sequence processing  
        self.protein_lstm = nn.LSTM(
            input_size=protein_input_size,
            hidden_size=protein_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Fusion layers
        # *2 because of bidirectional LSTM
        fusion_input_size = rna_hidden_size * 2 + protein_hidden_size * 2
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    def forward(self, rna: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rna: RNA sequences tensor of shape (batch_size, rna_seq_len, rna_input_size)
            protein: Protein sequences tensor of shape (batch_size, protein_seq_len, protein_input_size)
            
        Returns:
            Binding score predictions of shape (batch_size, 1)
        """
        batch_size = rna.size(0)
        
        # Process RNA sequences
        rna_output, (rna_hidden, _) = self.rna_lstm(rna)
        # Take the last hidden state from both directions
        rna_hidden = rna_hidden.view(self.num_layers, 2, batch_size, self.rna_hidden_size)
        rna_representation = torch.cat([rna_hidden[-1, 0], rna_hidden[-1, 1]], dim=1)
        
        # Process protein sequences
        protein_output, (protein_hidden, _) = self.protein_lstm(protein)
        # Take the last hidden state from both directions
        protein_hidden = protein_hidden.view(self.num_layers, 2, batch_size, self.protein_hidden_size)
        protein_representation = torch.cat([protein_hidden[-1, 0], protein_hidden[-1, 1]], dim=1)
        
        # Apply dropout
        rna_representation = self.dropout(rna_representation)
        protein_representation = self.dropout(protein_representation)
        
        # Combine representations
        combined = torch.cat([rna_representation, protein_representation], dim=1)
        
        # Predict binding score
        binding_score = self.fusion_layers(combined)
        
        return binding_score
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'BasicLSTM',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'rna_hidden_size': self.rna_hidden_size,
            'protein_hidden_size': self.protein_hidden_size,
            'num_layers': self.num_layers
        }
