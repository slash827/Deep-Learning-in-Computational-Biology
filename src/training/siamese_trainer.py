import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from ..training.evaluation import calculate_metrics, evaluate_model
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..models.siamese_protein_bert import SiameseProteinBERT, ContrastiveLoss, InfoNCELoss
from ..utils.helpers import create_run_directory, save_training_config


class SiameseTrainer:
    """
    Trainer for Siamese contrastive learning on RNA-protein binding prediction.
    
    Supports multiple training strategies:
    1. Pure contrastive learning
    2. Hybrid contrastive + regression learning
    3. Two-stage training (contrastive -> fine-tune regression)
    """
    
    def __init__(self,
                 model: SiameseProteinBERT,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device = None):
        """
        Args:
            model: SiameseProteinBERT model
            train_loader: Training data loader
            val_loader: Validation data loader  
            config: Training configuration
            device: Device to train on
        """
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training parameters
        self.epochs = config.get('epochs', 10)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.patience = config.get('patience', 5)
        self.min_delta = config.get('min_delta', 1e-4)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Loss configuration
        self.loss_type = config.get('loss_type', 'contrastive')  # 'contrastive', 'infonce', 'hybrid'
        self.contrastive_weight = config.get('contrastive_weight', 1.0)
        self.regression_weight = config.get('regression_weight', 1.0)
        self.temperature = config.get('temperature', 0.1)
        
        # Initialize losses
        self.contrastive_loss = ContrastiveLoss(temperature=self.temperature)
        self.infonce_loss = InfoNCELoss(temperature=self.temperature)
        self.regression_loss = nn.MSELoss()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=3
        )
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.val_correlations = []
        self.best_val_loss = float('inf')
        self.best_val_correlation = -1.0
        self.early_stop_counter = 0
        self.best_epoch = 0
        
        # Create output directory
        self.output_dir = create_run_directory(config.get('run_name', 'siamese_training'))
        save_training_config(config, self.output_dir)
        
        print(f"Trainer initialized. Output directory: {self.output_dir}")
        print(f"Training on device: {self.device}")
        print(f"Loss type: {self.loss_type}")
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], mode: str = 'train') -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for a batch.
        
        Args:
            batch: Batch of data
            mode: 'train' or 'val'
            
        Returns:
            total_loss, loss_components
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Extract data
        rna_seq1 = batch['rna_seq1']
        protein_emb1 = batch['protein_emb1']
        rna_seq2 = batch['rna_seq2'] 
        protein_emb2 = batch['protein_emb2']
        labels = batch['label']
        
        losses = {}
        total_loss = 0.0
        
        if self.loss_type in ['contrastive', 'hybrid']:
            # Contrastive learning
            emb1, emb2 = self.model(rna_seq1, protein_emb1, rna_seq2, protein_emb2, mode='contrastive')
            
            if self.loss_type == 'contrastive':
                contrastive_loss = self.model.compute_contrastive_loss(emb1, emb2, labels)
            else:
                contrastive_loss = self.contrastive_loss(emb1, emb2, labels)
            
            losses['contrastive'] = contrastive_loss.item()
            total_loss += self.contrastive_weight * contrastive_loss
        
        elif self.loss_type == 'infonce':
            # InfoNCE loss (assumes positive pairs are aligned in batch)
            emb1, emb2 = self.model(rna_seq1, protein_emb1, rna_seq2, protein_emb2, mode='contrastive')
            infonce_loss = self.infonce_loss(emb1, emb2)
            
            losses['infonce'] = infonce_loss.item()
            total_loss += infonce_loss
        
        # Add regression loss for hybrid training
        if self.loss_type == 'hybrid':
            # Predict binding scores for both samples
            pred1 = self.model(rna_seq1, protein_emb1, mode='inference')
            pred2 = self.model(rna_seq2, protein_emb2, mode='inference')
            
            # Use original affinities as targets
            target1 = batch['affinity1']
            target2 = batch['affinity2']
            
            reg_loss1 = self.regression_loss(pred1.squeeze(), target1)
            reg_loss2 = self.regression_loss(pred2.squeeze(), target2)
            reg_loss = (reg_loss1 + reg_loss2) / 2
            
            losses['regression'] = reg_loss.item()
            total_loss += self.regression_weight * reg_loss
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {'total': 0.0, 'contrastive': 0.0, 'regression': 0.0, 'infonce': 0.0}
        num_batches = 0
        
        # Use progress bar like existing trainers
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False, 
                           dynamic_ncols=True, ascii=True)
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Compute loss
            total_loss, batch_losses = self.compute_loss(batch, mode='train')
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in batch_losses.items():
                epoch_losses[key] += value
            
            num_batches += 1
            
            # Update progress bar less frequently for speed
            if num_batches % 10 == 0:
                progress_bar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Tuple[Dict[str, float], float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_losses = {'total': 0.0, 'contrastive': 0.0, 'regression': 0.0, 'infonce': 0.0}
        num_batches = 0
        
        # For correlation calculation (if doing regression)
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Compute loss
                total_loss, batch_losses = self.compute_loss(batch, mode='val')
                
                # Accumulate losses
                for key, value in batch_losses.items():
                    epoch_losses[key] += value
                
                # Collect predictions for correlation (if hybrid training)
                if self.loss_type == 'hybrid':
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    pred1 = self.model(batch['rna_seq1'], batch['protein_emb1'], mode='inference')
                    pred2 = self.model(batch['rna_seq2'], batch['protein_emb2'], mode='inference')
                    
                    all_predictions.extend(pred1.cpu().numpy().flatten())
                    all_predictions.extend(pred2.cpu().numpy().flatten())
                    all_targets.extend(batch['affinity1'].cpu().numpy().flatten())
                    all_targets.extend(batch['affinity2'].cpu().numpy().flatten())
                
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Calculate correlation if applicable using standard evaluation
        correlation = 0.0
        if self.loss_type == 'hybrid' and len(all_predictions) > 0:
            # Use the same evaluation method as the standard trainers
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            metrics = calculate_metrics(all_targets, all_predictions)
            correlation = metrics['pearson_correlation']
        
        return epoch_losses, correlation
    
    def evaluate_on_standard_task(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model on standard binding prediction task using existing evaluation functions.
        
        Args:
            test_loader: Optional test loader. If None, uses validation loader.
            
        Returns:
            Evaluation metrics
        """
        # Create a wrapper for the model to work with existing evaluation function
        class SiameseModelWrapper(torch.nn.Module):
            def __init__(self, siamese_model, device):
                super().__init__()
                self.siamese_model = siamese_model
                self.device = device
            
            def forward(self, rna, protein):
                # Convert to the format expected by Siamese model
                return self.siamese_model(rna, protein, mode='inference')
        
        # Create wrapper
        wrapper_model = SiameseModelWrapper(self.model, self.device)
        
        # Create a compatible data loader
        loader = test_loader or self.val_loader
        
        # Convert Siamese data format to standard format for evaluation
        standard_batches = []
        for batch in loader:
            # Use first sample from each Siamese pair
            standard_batch = {
                'rna': batch['rna_seq1'],
                'protein': batch['protein_emb1'], 
                'score': batch['affinity1']
            }
            standard_batches.append(standard_batch)
        
        # Create a simple data loader wrapper
        class StandardDataLoader:
            def __init__(self, batches):
                self.batches = batches
            
            def __iter__(self):
                return iter(self.batches)
        
        standard_loader = StandardDataLoader(standard_batches)
        
        # Use the existing evaluate_model function
        metrics, predictions, targets = evaluate_model(wrapper_model, standard_loader, self.device)
        
        # Add spearman correlation
        spearman_corr = spearmanr(predictions.flatten(), targets.flatten())[0]
        spearman_corr = 0.0 if np.isnan(spearman_corr) else spearman_corr
        
        metrics.update({
            'spearman_correlation': spearman_corr,
            'num_samples': len(predictions)
        })
        
        return metrics
    
    def train(self) -> Dict[str, any]:
        """Main training loop."""
        print(f"Starting Siamese training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: SiameseProteinBERT")
        print(f"Loss type: {self.loss_type}")
        print(f"Early stopping patience: {self.patience}")
        print(f"Gradient clipping: {self.max_grad_norm}")
        
        # Get model info
        model_info = self.model.get_model_info()
        print(f"Model parameters: {model_info.get('total_parameters', 'Unknown')}")
        print("="*50)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate epoch
            val_losses, val_correlation = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step(val_losses['total'])
            
            # Save losses
            self.train_losses.append(train_losses['total'])
            self.val_losses.append(val_losses['total'])
            self.val_correlations.append(val_correlation)
            
            # Check for best model
            is_best = False
            if self.loss_type == 'hybrid':
                # Use correlation for hybrid training
                if val_correlation > self.best_val_correlation:
                    self.best_val_correlation = val_correlation
                    self.best_val_loss = val_losses['total']
                    self.best_epoch = epoch
                    is_best = True
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            else:
                # Use loss for pure contrastive training
                if val_losses['total'] < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_losses['total']
                    self.best_val_correlation = val_correlation
                    self.best_epoch = epoch
                    is_best = True
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            
            # Save best model
            if is_best:
                self.save_checkpoint('best_model.pt')
            
            # Print epoch summary using the same format as existing trainers
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.epochs} ({epoch_time:.1f}s) - LR: {lr:.2e}")
            print(f"  Train Loss: {train_losses['total']:.4f}, Val Loss: {val_losses['total']:.4f}")
            if self.loss_type == 'hybrid':
                print(f"  Val Corr: {val_correlation:.4f}")
                print(f"  Best Val Corr: {self.best_val_correlation:.4f}")
            else:
                print(f"  Best Val Loss: {self.best_val_loss:.4f}")
            print(f"  No improvement: {self.early_stop_counter}/{self.patience}")
            
            # Early stopping
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        # Final evaluation on standard task
        final_metrics = self.evaluate_on_standard_task()
        
        # Training summary
        summary = {
            'best_val_loss': self.best_val_loss,
            'best_val_correlation': self.best_val_correlation,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch + 1,
            'total_training_time': total_time,
            'final_metrics': final_metrics,
            'model_info': self.model.get_model_info(),
            'early_stopped': self.early_stop_counter >= self.patience,
            'loss_type': self.loss_type
        }
        
        # Save summary
        self.save_training_summary(summary)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Print final summary in the same format as existing trainers
        print("="*50)
        print(f"Siamese training completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
        if self.early_stop_counter >= self.patience:
            print(f"[STOP] Training stopped early due to no improvement")
        else:
            print(f"[SUCCESS] Training completed all epochs")
        print(f"[STATS] Best validation correlation: {self.best_val_correlation:.4f} "
              f"at epoch {self.best_epoch+1}")
        print(f"[SPEED] Average time per epoch: {total_time/summary['total_epochs']:.1f}s")
        print("="*50)
        
        print(f"Final evaluation metrics:")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        return summary
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'epoch': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'best_val_correlation': self.best_val_correlation
        }
        
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
    
    def save_training_summary(self, summary: Dict):
        """Save training summary to JSON."""
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                # Handle numpy types
                if hasattr(obj, 'item'):  # numpy scalars
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy arrays
                    return obj.tolist()
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return [convert_types(item) for item in obj]
                # Handle PyTorch tensors
                elif hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
                    return obj.cpu().numpy().tolist()
                return obj
            
            json_summary = convert_types(summary)
            json.dump(json_summary, f, indent=2)
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Correlation curve (if applicable)
        if self.loss_type == 'hybrid':
            plt.subplot(1, 3, 2)
            plt.plot(self.val_correlations, label='Val Correlation')
            plt.xlabel('Epoch')
            plt.ylabel('Correlation')
            plt.title('Validation Correlation')
            plt.legend()
            plt.grid(True)
        
        # Learning rate
        plt.subplot(1, 3, 3)
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        plt.plot([lrs[0]] * len(self.train_losses))  # Simplified LR plot
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/training_curves.png")
