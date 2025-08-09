"""
Fast BiLSTM with Simplified Attention for RNA-Protein Binding Prediction

This module implements an optimized version of the bidirectional LSTM with 
simplified attention mechanism for faster training while maintaining accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_mask_value(tensor_dtype):
    """Get appropriate mask value based on tensor dtype to avoid overflow."""
    if tensor_dtype == torch.float16:
        return -1e4  # Safe value for FP16
    else:
        return -1e9  # Standard value for FP32


class SimplifiedAttention(nn.Module):
    """Simplified attention mechanism for faster computation"""
    
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(SimplifiedAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Single linear layer for all Q, K, V projections (more efficient)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
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
        
        # Generate Q, K, V in one go (more efficient)
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention (simplified)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided (use appropriate value for tensor dtype)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            mask_value = get_mask_value(scores.dtype)
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


class FastAttentionLSTM(nn.Module):
    """
    Fast BiLSTM with Simplified Attention for RNA-Protein Binding Prediction
    
    Optimizations:
    1. Simplified attention mechanism
    2. Reduced cross-attention complexity
    3. More efficient pooling
    4. Streamlined fusion network
    """
    
    def __init__(self, 
                 rna_input_size=5,
                 protein_input_size=21,
                 rna_hidden_size=128,
                 protein_hidden_size=128,
                 num_layers=2,
                 dropout=0.3,
                 attention_heads=4,
                 attention_dropout=0.1,
                 use_positional_encoding=False):
        super(FastAttentionLSTM, self).__init__()
        
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
        
        # Simplified self-attention layers (faster than multi-head)
        self.rna_attention = SimplifiedAttention(
            hidden_size=rna_lstm_output_size,
            num_heads=attention_heads,
            dropout=attention_dropout
        )
        
        self.protein_attention = SimplifiedAttention(
            hidden_size=protein_lstm_output_size,
            num_heads=attention_heads,
            dropout=attention_dropout
        )
        
        # Simplified cross-attention (single layer instead of full cross-attention)
        cross_dim = min(rna_lstm_output_size, protein_lstm_output_size)
        self.rna_projection = nn.Linear(rna_lstm_output_size, cross_dim)
        self.protein_projection = nn.Linear(protein_lstm_output_size, cross_dim)
        
        # Simple cross-attention using dot product
        self.cross_attention_scale = math.sqrt(cross_dim)
        
        # More efficient pooling (adaptive + average for robustness)
        self.rna_pooling = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        self.protein_pooling = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        # Streamlined fusion network (fewer layers for speed)
        fusion_input_size = cross_dim * 2  # RNA + protein representations
        
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_input_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_size // 2, fusion_input_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_size // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights efficiently"""
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
        Fast forward pass
        
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
        
        # Apply simplified self-attention
        rna_attended = self.rna_attention(rna_lstm_out, rna_mask)
        protein_attended = self.protein_attention(protein_lstm_out, protein_mask)
        
        # Project to common dimension
        rna_projected = self.rna_projection(rna_attended)
        protein_projected = self.protein_projection(protein_attended)
        
        # Simplified cross-attention using mean pooling and dot product
        # This is much faster than full cross-attention
        rna_mean = torch.mean(rna_projected, dim=1)  # (batch_size, cross_dim)
        protein_mean = torch.mean(protein_projected, dim=1)  # (batch_size, cross_dim)
        
        # Cross-attention weights
        cross_score = torch.sum(rna_mean * protein_mean, dim=1, keepdim=True) / self.cross_attention_scale
        cross_weight = torch.sigmoid(cross_score)
        
        # Weighted combination
        rna_weighted = rna_mean * cross_weight
        protein_weighted = protein_mean * cross_weight
        
        # Alternative: Use max pooling for more robust representation
        rna_max = torch.max(rna_projected, dim=1)[0]
        protein_max = torch.max(protein_projected, dim=1)[0]
        
        # Combine mean and max pooling
        rna_final = (rna_weighted + rna_max) / 2
        protein_final = (protein_weighted + protein_max) / 2
        
        # Fusion and prediction
        fused_features = torch.cat([rna_final, protein_final], dim=1)
        binding_score = self.fusion_network(fused_features)
        
        return binding_score
    
    def get_model_info(self):
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'FastAttentionLSTM',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'rna_hidden_size': self.rna_hidden_size,
            'protein_hidden_size': self.protein_hidden_size,
            'num_layers': self.num_layers,
            'attention_heads': self.attention_heads,
            'dropout': self.dropout_rate,
            'use_positional_encoding': self.use_positional_encoding,
            'optimizations': 'simplified_attention_enabled'
        }


# Alternative: Even simpler model for maximum speed
class UltraFastLSTM(nn.Module):
    """Ultra-fast LSTM model with minimal attention for maximum speed"""
    
    def __init__(self, 
                 rna_input_size=5,
                 protein_input_size=21,
                 hidden_size=64,
                 num_layers=1,
                 dropout=0.2):
        super(UltraFastLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Single LSTM layers (no bidirectional for speed)
        self.rna_lstm = nn.LSTM(
            input_size=rna_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  # Faster than bidirectional
            batch_first=True
        )
        
        self.protein_lstm = nn.LSTM(
            input_size=protein_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        # Simple attention weights (learnable)
        self.rna_attention_weight = nn.Parameter(torch.randn(hidden_size))
        self.protein_attention_weight = nn.Parameter(torch.randn(hidden_size))
        
        # Minimal fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, rna_seq, protein_seq):
        """Ultra-fast forward pass"""
        # LSTM encoding
        rna_out, _ = self.rna_lstm(rna_seq)
        protein_out, _ = self.protein_lstm(protein_seq)
        
        # Simple attention pooling
        rna_weights = torch.softmax(torch.matmul(rna_out, self.rna_attention_weight), dim=1)
        protein_weights = torch.softmax(torch.matmul(protein_out, self.protein_attention_weight), dim=1)
        
        rna_pooled = torch.sum(rna_out * rna_weights.unsqueeze(-1), dim=1)
        protein_pooled = torch.sum(protein_out * protein_weights.unsqueeze(-1), dim=1)
        
        # Fusion and prediction
        fused = torch.cat([rna_pooled, protein_pooled], dim=1)
        binding_score = self.fusion_network(fused)
        
        return binding_score
    
    def get_model_info(self):
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'UltraFastLSTM',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'optimizations': 'ultra_fast_mode'
        }


# Test function for model verification
def test_fast_models():
    """Test the fast attention models"""
    print("Testing Fast Attention Models...")
    
    # Model parameters
    batch_size = 8
    rna_seq_len = 50
    protein_seq_len = 200
    
    # Create models
    fast_model = FastAttentionLSTM(
        rna_input_size=5,
        protein_input_size=21,
        rna_hidden_size=64,
        protein_hidden_size=64,
        num_layers=1,
        dropout=0.1,
        attention_heads=4,
        attention_dropout=0.05
    )
    
    ultra_fast_model = UltraFastLSTM(
        rna_input_size=5,
        protein_input_size=21,
        hidden_size=64,
        num_layers=1,
        dropout=0.1
    )
    
    # Create dummy data
    rna_seq = torch.randn(batch_size, rna_seq_len, 5)
    protein_seq = torch.randn(batch_size, protein_seq_len, 21)
    
    # Test FastAttentionLSTM
    print("\n1. FastAttentionLSTM:")
    output1 = fast_model(rna_seq, protein_seq)
    print(f"   Input shapes: RNA {rna_seq.shape}, Protein {protein_seq.shape}")
    print(f"   Output shape: {output1.shape}")
    print(f"   Model info: {fast_model.get_model_info()}")
    
    # Test UltraFastLSTM
    print("\n2. UltraFastLSTM:")
    output2 = ultra_fast_model(rna_seq, protein_seq)
    print(f"   Input shapes: RNA {rna_seq.shape}, Protein {protein_seq.shape}")
    print(f"   Output shape: {output2.shape}")
    print(f"   Model info: {ultra_fast_model.get_model_info()}")
    
    print("\n✅ All fast models test passed!")
    
    # Speed comparison
    import time
    num_runs = 100
    
    # FastAttentionLSTM speed test
    fast_model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = fast_model(rna_seq, protein_seq)
    fast_time = time.time() - start_time
    
    # UltraFastLSTM speed test
    ultra_fast_model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = ultra_fast_model(rna_seq, protein_seq)
    ultra_fast_time = time.time() - start_time
    
    print(f"\n⚡ Speed Comparison ({num_runs} runs):")
    print(f"   FastAttentionLSTM: {fast_time:.3f}s ({fast_time/num_runs*1000:.2f}ms per batch)")
    print(f"   UltraFastLSTM: {ultra_fast_time:.3f}s ({ultra_fast_time/num_runs*1000:.2f}ms per batch)")
    print(f"   Speedup: {fast_time/ultra_fast_time:.1f}x faster")


if __name__ == "__main__":
    test_fast_models()
