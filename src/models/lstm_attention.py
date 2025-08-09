"""
Enhanced BiLSTM with Self-Attention for RNA-Protein Binding Prediction

This module implements a bidirectional LSTM with multi-head self-attention mechanism
for improved modeling of RNA-protein interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for sequence modeling"""
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional mask tensor of shape (batch_size, seq_len)
        Returns:
            attended: Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Generate queries, keys, values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided (use smaller value for FP16 compatibility)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            # Use -1e4 instead of -1e9 to avoid FP16 overflow
            mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, mask_value)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Output projection and residual connection
        attended = self.out(attended)
        attended = self.layer_norm(attended + residual)
        
        return attended


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    
    def __init__(self, hidden_size, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, hidden_size)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class AttentionLSTM(nn.Module):
    """
    Enhanced BiLSTM with Self-Attention for RNA-Protein Binding Prediction
    
    Architecture:
    1. Separate BiLSTM encoders for RNA and protein sequences
    2. Multi-head self-attention on both encoded sequences
    3. Cross-attention between RNA and protein representations
    4. Fusion network for final binding prediction
    """
    
    def __init__(self, 
                 rna_input_size=5,
                 protein_input_size=21,
                 rna_hidden_size=128,
                 protein_hidden_size=128,
                 num_layers=2,
                 dropout=0.3,
                 attention_heads=8,
                 attention_dropout=0.1,
                 use_positional_encoding=False):
        super(AttentionLSTM, self).__init__()
        
        self.rna_input_size = rna_input_size
        self.protein_input_size = protein_input_size
        self.rna_hidden_size = rna_hidden_size
        self.protein_hidden_size = protein_hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.attention_heads = attention_heads
        self.use_positional_encoding = use_positional_encoding
        
        # RNA encoder with BiLSTM
        self.rna_lstm = nn.LSTM(
            input_size=rna_input_size,
            hidden_size=rna_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Protein encoder with BiLSTM
        self.protein_lstm = nn.LSTM(
            input_size=protein_input_size,
            hidden_size=protein_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Hidden size after bidirectional LSTM
        rna_lstm_output_size = rna_hidden_size * 2
        protein_lstm_output_size = protein_hidden_size * 2
        
        # Positional encoding (optional)
        if use_positional_encoding:
            self.rna_pos_encoding = PositionalEncoding(rna_lstm_output_size)
            self.protein_pos_encoding = PositionalEncoding(protein_lstm_output_size)
        
        # Self-attention layers
        self.rna_attention = MultiHeadSelfAttention(
            hidden_size=rna_lstm_output_size,
            num_heads=attention_heads,
            dropout=attention_dropout
        )
        
        self.protein_attention = MultiHeadSelfAttention(
            hidden_size=protein_lstm_output_size,
            num_heads=attention_heads,
            dropout=attention_dropout
        )
        
        # Cross-attention between RNA and protein
        # We need to project to the same dimension for cross-attention
        cross_dim = min(rna_lstm_output_size, protein_lstm_output_size)
        self.rna_projection = nn.Linear(rna_lstm_output_size, cross_dim)
        self.protein_projection = nn.Linear(protein_lstm_output_size, cross_dim)
        
        self.cross_attention = MultiHeadSelfAttention(
            hidden_size=cross_dim,
            num_heads=attention_heads,
            dropout=attention_dropout
        )
        
        # Global pooling layers
        self.rna_pooling = nn.AdaptiveMaxPool1d(1)
        self.protein_pooling = nn.AdaptiveMaxPool1d(1)
        
        # Feature fusion network
        fusion_input_size = rna_lstm_output_size + protein_lstm_output_size
        fusion_hidden_size = max(fusion_input_size, 256)
        
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size // 2, fusion_hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias to 1
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
    
    def create_mask(self, sequences):
        """Create attention mask for padded sequences"""
        # Assume padding token is 0
        return (sequences.sum(dim=-1) != 0).float()
    
    def forward(self, rna_seq, protein_seq):
        """
        Forward pass
        
        Args:
            rna_seq: RNA sequence tensor (batch_size, rna_seq_len, rna_input_size)
            protein_seq: Protein sequence tensor (batch_size, protein_seq_len, protein_input_size)
            
        Returns:
            binding_score: Predicted binding score (batch_size, 1)
        """
        batch_size = rna_seq.size(0)
        
        # Create attention masks
        rna_mask = self.create_mask(rna_seq)
        protein_mask = self.create_mask(protein_seq)
        
        # Encode sequences with BiLSTM
        rna_lstm_out, _ = self.rna_lstm(rna_seq)
        protein_lstm_out, _ = self.protein_lstm(protein_seq)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            rna_lstm_out = rna_lstm_out.transpose(0, 1)  # (seq_len, batch, hidden)
            rna_lstm_out = self.rna_pos_encoding(rna_lstm_out)
            rna_lstm_out = rna_lstm_out.transpose(0, 1)  # (batch, seq_len, hidden)
            
            protein_lstm_out = protein_lstm_out.transpose(0, 1)
            protein_lstm_out = self.protein_pos_encoding(protein_lstm_out)
            protein_lstm_out = protein_lstm_out.transpose(0, 1)
        
        # Apply self-attention
        rna_attended = self.rna_attention(rna_lstm_out, rna_mask)
        protein_attended = self.protein_attention(protein_lstm_out, protein_mask)
        
        # Project to common dimension for cross-attention
        rna_projected = self.rna_projection(rna_attended)
        protein_projected = self.protein_projection(protein_attended)
        
        # Cross-attention: concatenate projected features and attend
        combined_features = torch.cat([rna_projected, protein_projected], dim=1)
        combined_mask = torch.cat([rna_mask, protein_mask], dim=1)
        cross_attended = self.cross_attention(combined_features, combined_mask)
        
        # Split back to RNA and protein parts
        rna_seq_len = rna_projected.size(1)
        rna_cross = cross_attended[:, :rna_seq_len, :]
        protein_cross = cross_attended[:, rna_seq_len:, :]
        
        # Global pooling to get fixed-size representations
        rna_global = rna_cross.transpose(1, 2)
        protein_global = protein_cross.transpose(1, 2)
        
        rna_pooled = self.rna_pooling(rna_global).squeeze(-1)
        protein_pooled = self.protein_pooling(protein_global).squeeze(-1)
        
        # Fusion and prediction
        fused_features = torch.cat([rna_pooled, protein_pooled], dim=1)
        binding_score = self.fusion_network(fused_features)
        
        return binding_score
    
    def get_model_info(self):
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'AttentionLSTM',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'rna_hidden_size': self.rna_hidden_size,
            'protein_hidden_size': self.protein_hidden_size,
            'num_layers': self.num_layers,
            'attention_heads': self.attention_heads,
            'dropout': self.dropout_rate,
            'use_positional_encoding': self.use_positional_encoding
        }
    
    def get_attention_weights(self, rna_seq, protein_seq):
        """
        Get attention weights for visualization
        
        Returns:
            dict: Attention weights for RNA, protein, and cross-attention
        """
        self.eval()
        with torch.no_grad():
            batch_size = rna_seq.size(0)
            
            # Create attention masks
            rna_mask = self.create_mask(rna_seq)
            protein_mask = self.create_mask(protein_seq)
            
            # Encode sequences
            rna_lstm_out, _ = self.rna_lstm(rna_seq)
            protein_lstm_out, _ = self.protein_lstm(protein_seq)
            
            # Get attention weights (would need to modify attention layers to return weights)
            # This is a simplified version for demonstration
            return {
                'rna_self_attention': None,  # Would contain RNA self-attention weights
                'protein_self_attention': None,  # Would contain protein self-attention weights
                'cross_attention': None  # Would contain cross-attention weights
            }


# Test function for model verification
def test_attention_lstm():
    """Test the AttentionLSTM model"""
    print("Testing AttentionLSTM model...")
    
    # Model parameters
    batch_size = 4
    rna_seq_len = 50
    protein_seq_len = 200
    
    # Create model
    model = AttentionLSTM(
        rna_input_size=5,
        protein_input_size=21,
        rna_hidden_size=64,
        protein_hidden_size=64,
        num_layers=2,
        dropout=0.1,
        attention_heads=8,
        attention_dropout=0.1,
        use_positional_encoding=True
    )
    
    # Create dummy data
    rna_seq = torch.randn(batch_size, rna_seq_len, 5)
    protein_seq = torch.randn(batch_size, protein_seq_len, 21)
    
    # Forward pass
    output = model(rna_seq, protein_seq)
    
    print(f"Input shapes: RNA {rna_seq.shape}, Protein {protein_seq.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model info: {model.get_model_info()}")
    print("✅ AttentionLSTM test passed!")


if __name__ == "__main__":
    test_attention_lstm()
