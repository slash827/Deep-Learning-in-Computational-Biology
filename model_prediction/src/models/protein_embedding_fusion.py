import torch
import torch.nn as nn
from typing import Optional


class ProteinEmbeddingFusion(nn.Module):
    """
    Model that fuses an RNA BiLSTM representation with a precomputed protein embedding vector.
    - RNA branch: BiLSTM + optional self-attention-like pooling (mean + max)
    - Protein branch: MLP over fixed-length embedding vector
    - Fusion: concatenate and predict a scalar binding score
    """

    def __init__(self,
                 rna_input_size: int = 5,
                 rna_hidden_size: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 protein_embedding_dim: int = 1024,
                 protein_mlp_hidden: Optional[int] = None):
        super().__init__()

        self.rna_input_size = rna_input_size
        self.rna_hidden_size = rna_hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.protein_embedding_dim = protein_embedding_dim

        # RNA encoder
        self.rna_lstm = nn.LSTM(
            input_size=rna_input_size,
            hidden_size=rna_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )

        # Simple pooling: mean + max over sequence
        self.dropout = nn.Dropout(dropout)

        # Protein embedding MLP
        pe_hidden = protein_mlp_hidden or max(128, protein_embedding_dim // 2)
        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_embedding_dim, pe_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(pe_hidden, pe_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Fusion head
        fusion_in = (rna_hidden_size * 2) * 2 + (pe_hidden // 2)  # mean+max for RNA
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_in // 2, fusion_in // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_in // 4, 1)
        )

    def forward(self, rna_seq: torch.Tensor, protein_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna_seq: (batch, rna_seq_len, 5)
            protein_embedding: (batch, protein_embedding_dim)
        Returns:
            (batch, 1)
        """
        # RNA encoding
        rna_out, _ = self.rna_lstm(rna_seq)
        rna_out = self.dropout(rna_out)
        rna_mean = torch.mean(rna_out, dim=1)
        rna_max, _ = torch.max(rna_out, dim=1)
        rna_feat = torch.cat([rna_mean, rna_max], dim=1)

        # Protein embedding branch
        if protein_embedding.dim() == 3 and protein_embedding.size(1) == 1:
            protein_embedding = protein_embedding.squeeze(1)
        protein_feat = self.protein_mlp(protein_embedding)

        # Fuse
        fused = torch.cat([rna_feat, protein_feat], dim=1)
        out = self.fusion_head(fused)
        return out

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model_name': 'ProteinEmbeddingFusion',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'rna_hidden_size': self.rna_hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'protein_embedding_dim': self.protein_embedding_dim
        }


