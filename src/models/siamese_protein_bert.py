import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .protein_embedding_fusion import ProteinEmbeddingFusion


class SiameseProteinBERT(nn.Module):
    """
    Siamese Neural Network with Contrastive Learning for RNA-Protein Binding Prediction.
    
    Architecture:
    - Uses ProteinEmbeddingFusion as the backbone encoder
    - Processes pairs of RNA-protein samples through shared weights
    - Learns to distinguish binding vs non-binding pairs through contrastive loss
    - Supports both contrastive training and standard regression inference
    """
    
    def __init__(self,
                 rna_input_size: int = 5,
                 rna_hidden_size: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 protein_embedding_dim: int = 1024,
                 protein_mlp_hidden: Optional[int] = None,
                 embedding_dim: int = 256,
                 temperature: float = 0.1):
        """
        Args:
            rna_input_size: Size of RNA input features
            rna_hidden_size: Hidden size for RNA LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            protein_embedding_dim: Dimension of protein embeddings (from ProteinBERT)
            protein_mlp_hidden: Hidden size for protein MLP
            embedding_dim: Final embedding dimension for contrastive learning
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        
        # Backbone encoder (shared weights)
        self.backbone = ProteinEmbeddingFusion(
            rna_input_size=rna_input_size,
            rna_hidden_size=rna_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            protein_embedding_dim=protein_embedding_dim,
            protein_mlp_hidden=protein_mlp_hidden
        )
        
        # Calculate the correct fusion input size after backbone processing
        # RNA: bidirectional LSTM output (mean + max pooling) = rna_hidden_size * 2 * 2
        # Protein: MLP output = (protein_mlp_hidden or max(128, protein_embedding_dim // 2)) // 2
        pe_hidden = protein_mlp_hidden or max(128, protein_embedding_dim // 2)
        rna_feat_size = (rna_hidden_size * 2) * 2  # bidirectional * (mean + max)
        protein_feat_size = pe_hidden // 2
        fusion_in = rna_feat_size + protein_feat_size
        
        # We'll use the fused features directly (before backbone's fusion_head)
        # So the input to projection_head is fusion_in
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_in // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)  # Normalize embeddings
        )
        
        # Regression head for binding score prediction (for inference)
        self.regression_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        
        # Modify backbone to remove its final layer
        self._modify_backbone()
    
    def _modify_backbone(self):
        """Remove the final prediction layer from backbone to access intermediate features."""
        # We'll bypass the fusion_head entirely and access fused features directly
        # The original fusion_head will be replaced by our projection_head
        pass
    
    def encode(self, rna_seq: torch.Tensor, protein_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encode RNA-protein pair into embedding vector.
        
        Args:
            rna_seq: (batch, rna_seq_len, 5)
            protein_embedding: (batch, protein_embedding_dim)
            
        Returns:
            embeddings: (batch, embedding_dim)
        """
        # Get features from backbone (without final prediction)
        rna_out, _ = self.backbone.rna_lstm(rna_seq)
        rna_out = self.backbone.dropout(rna_out)
        rna_mean = torch.mean(rna_out, dim=1)
        rna_max, _ = torch.max(rna_out, dim=1)
        rna_feat = torch.cat([rna_mean, rna_max], dim=1)
        
        # Protein features
        if protein_embedding.dim() == 3 and protein_embedding.size(1) == 1:
            protein_embedding = protein_embedding.squeeze(1)
        protein_feat = self.backbone.protein_mlp(protein_embedding)
        
        # Fuse features
        fused = torch.cat([rna_feat, protein_feat], dim=1)
        
        # Project to embedding space directly (bypass backbone's fusion_head)
        embeddings = self.projection_head(fused)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, 
                rna_seq1: torch.Tensor, 
                protein_emb1: torch.Tensor,
                rna_seq2: Optional[torch.Tensor] = None,
                protein_emb2: Optional[torch.Tensor] = None,
                mode: str = 'contrastive') -> torch.Tensor:
        """
        Forward pass supporting both contrastive learning and inference.
        
        Args:
            rna_seq1: First RNA sequence (batch, rna_seq_len, 5)
            protein_emb1: First protein embedding (batch, protein_embedding_dim)
            rna_seq2: Second RNA sequence (optional, for contrastive learning)
            protein_emb2: Second protein embedding (optional, for contrastive learning)
            mode: 'contrastive' for training, 'inference' for prediction
            
        Returns:
            For contrastive mode: (embedding1, embedding2)
            For inference mode: binding_score
        """
        if mode == 'contrastive':
            if rna_seq2 is None or protein_emb2 is None:
                raise ValueError("rna_seq2 and protein_emb2 required for contrastive mode")
            
            # Encode both pairs
            emb1 = self.encode(rna_seq1, protein_emb1)
            emb2 = self.encode(rna_seq2, protein_emb2)
            
            return emb1, emb2
        
        elif mode == 'inference':
            # Encode single pair and predict binding score
            embedding = self.encode(rna_seq1, protein_emb1)
            binding_score = self.regression_head(embedding)
            return binding_score
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'contrastive' or 'inference'")
    
    def compute_contrastive_loss(self, 
                                embeddings1: torch.Tensor, 
                                embeddings2: torch.Tensor, 
                                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss for pairs.
        
        Args:
            embeddings1: First set of embeddings (batch, embedding_dim)
            embeddings2: Second set of embeddings (batch, embedding_dim)
            labels: Binary labels (1 for similar/binding, 0 for dissimilar/non-binding)
            
        Returns:
            contrastive_loss: Scalar loss value
        """
        # Compute cosine similarity
        similarity = F.cosine_similarity(embeddings1, embeddings2, dim=1)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        # Contrastive loss
        positive_loss = labels * (1 - similarity) ** 2
        negative_loss = (1 - labels) * torch.clamp(similarity - 0.1, min=0) ** 2
        
        loss = (positive_loss + negative_loss).mean()
        
        return loss
    
    def compute_triplet_loss(self, 
                           anchor: torch.Tensor,
                           positive: torch.Tensor, 
                           negative: torch.Tensor,
                           margin: float = 0.3) -> torch.Tensor:
        """
        Alternative: Compute triplet loss for better contrastive learning.
        
        Args:
            anchor: Anchor embeddings (batch, embedding_dim)
            positive: Positive embeddings (batch, embedding_dim)
            negative: Negative embeddings (batch, embedding_dim)
            margin: Margin for triplet loss
            
        Returns:
            triplet_loss: Scalar loss value
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + margin).mean()
        
        return loss
    
    def get_model_info(self):
        """Get model information including parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_info = self.backbone.get_model_info()
        
        return {
            'model_name': 'SiameseProteinBERT',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_info['total_parameters'],
            'embedding_dim': self.embedding_dim,
            'temperature': self.temperature,
            **backbone_info
        }


class ContrastiveLoss(nn.Module):
    """
    Standalone contrastive loss implementation with additional features.
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss with margin.
        
        Args:
            embeddings1: (batch, dim)
            embeddings2: (batch, dim)
            labels: (batch,) - 1 for positive pairs, 0 for negative pairs
        """
        # Euclidean distance
        distance = F.pairwise_distance(embeddings1, embeddings2, p=2)
        
        # Contrastive loss with margin
        positive_loss = labels * distance ** 2
        negative_loss = (1 - labels) * torch.clamp(self.margin - distance, min=0) ** 2
        
        loss = (positive_loss + negative_loss).mean()
        
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning (used in SimCLR, MoCo).
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss assuming positive pairs are aligned in batch.
        
        Args:
            embeddings1: (batch, dim)
            embeddings2: (batch, dim)
        """
        batch_size = embeddings1.size(0)
        
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        # Labels for positive pairs (diagonal)
        labels = torch.arange(batch_size, device=embeddings1.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
