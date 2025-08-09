import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import signal
import sys
import os
from typing import Dict, Optional, Tuple
from tqdm import tqdm

try:
    from .evaluation import evaluate_model, MetricsTracker
except ImportError:
    from src.training.evaluation import evaluate_model, MetricsTracker


class FastRNAProteinTrainer:
    """Optimized trainer class for RNA-protein binding prediction models with speed optimizations."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 max_grad_norm: float = 1.0,
                 lr_scheduler_patience: int = 5,
                 lr_scheduler_factor: float = 0.5,
                 output_dir: str = None,
                 warmup_epochs: int = 0,
                 use_mixed_precision: bool = True):
        """
        Initialize optimized trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping
            max_grad_norm: Maximum gradient norm for clipping
            lr_scheduler_patience: Patience for learning rate scheduler
            lr_scheduler_factor: Factor for learning rate reduction
            output_dir: Directory to save outputs
            warmup_epochs: Number of epochs for learning rate warmup
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.warmup_epochs = warmup_epochs
        self.initial_lr = learning_rate
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'
        
        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer with optimized settings
        self.optimizer = optim.AdamW(  # AdamW often works better than Adam
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with more aggressive reduction
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize validation correlation
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            verbose=True,
            min_lr=1e-6,
            threshold=1e-4
        )
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("🚀 Mixed precision training enabled")
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Early stopping
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        # Interruption handling
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Optimization flags
        self.compile_model = hasattr(torch, 'compile') and device.type == 'cuda'
        if self.compile_model:
            try:
                self.model = torch.compile(self.model, mode='default')
                print("🚀 Model compilation enabled for speed boost")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")
                self.compile_model = False
        
        # Print memory info if using CUDA
        if self.device.type == 'cuda':
            self._print_memory_info("Initial")
    
    def _print_memory_info(self, stage: str = "Current"):
        """Print current GPU memory usage."""
        if self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                cached = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f"[GPU] {stage} GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
            except Exception:
                pass
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C interruption gracefully."""
        print("\n" + "="*50)
        print("[STOP] Training interrupted by user (Ctrl+C)")
        print("Finishing current epoch and moving to evaluation...")
        print("="*50)
        self.interrupted = True
        
    def train_epoch(self) -> float:
        """Train for one epoch with optimizations."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Use faster progress bar settings
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False, 
                           dynamic_ncols=True, ascii=True)
        
        for batch in progress_bar:
            # Move data to device (non-blocking for speed)
            rna = batch['rna'].to(self.device, non_blocking=True)
            protein = batch['protein'].to(self.device, non_blocking=True)
            targets = batch['score'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Forward pass with mixed precision
            try:
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(rna, protein)
                        loss = self.criterion(predictions, targets)
                else:
                    predictions = self.model(rna, protein)
                    loss = self.criterion(predictions, targets)
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDNN_STATUS_ALLOC_FAILED" in str(e):
                    print(f"\n[ERROR] GPU Memory Error: {e}")
                    self._print_memory_info("Before Error")
                    print("[TIP] Try reducing --batch_size, --hidden_size, or use --force_cpu")
                    raise
                else:
                    raise
            
            # Backward pass with mixed precision
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                # Update weights
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar less frequently for speed
            if num_batches % 10 == 0:
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        metrics, predictions, targets = evaluate_model(
            self.model, self.val_loader, self.device
        )
        
        return metrics['mse'], metrics['pearson_correlation']
    
    def train(self, num_epochs: int, save_path: str = None) -> Dict:
        """
        Train the model with optimizations.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save the best model
            
        Returns:
            Training summary dictionary
        """
        print(f"Starting FAST training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Mixed precision: {self.use_mixed_precision}")
        print(f"Model compilation: {self.compile_model}")
        print(f"Early stopping patience: {self.patience}")
        
        # Get model info
        model_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        print(f"Model parameters: {model_info.get('total_parameters', 'Unknown')}")
        
        start_time = time.time()
        
        # Warmup iterations for GPU
        if self.device.type == 'cuda':
            print("🔥 GPU warmup...")
            self.model.train()
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                for i, batch in enumerate(self.train_loader):
                    if i >= 3:  # Just a few warmup iterations
                        break
                    rna = batch['rna'].to(self.device, non_blocking=True)
                    protein = batch['protein'].to(self.device, non_blocking=True)
                    _ = self.model(rna, protein)
            torch.cuda.synchronize()
            print("✅ GPU warmup completed")
        
        for epoch in range(num_epochs):
            if self.interrupted:
                print(f"Training stopped at epoch {epoch+1} due to interruption")
                break
                
            epoch_start_time = time.time()
            
            # Apply learning rate warmup
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                warmup_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"Warmup LR: {warmup_lr:.6f}")
            
            # Train for one epoch
            try:
                train_loss = self.train_epoch()
                if self.interrupted:
                    print(f"Epoch {epoch+1} interrupted during training")
                    break
            except KeyboardInterrupt:
                print(f"Training interrupted during epoch {epoch+1}")
                self.interrupted = True
                break
            
            # Validate
            try:
                val_loss, val_correlation = self.validate()
                if self.interrupted:
                    print(f"Epoch {epoch+1} interrupted during validation")
                    break
            except KeyboardInterrupt:
                print(f"Validation interrupted during epoch {epoch+1}")
                self.interrupted = True
                break
            
            # Calculate training correlation (less frequently for speed)
            train_correlation = 0.0
            if epoch % 2 == 0 or epoch == num_epochs - 1:  # Every other epoch
                try:
                    train_metrics, _, _ = evaluate_model(
                        self.model, self.train_loader, self.device
                    )
                    train_correlation = train_metrics['pearson_correlation']
                except KeyboardInterrupt:
                    print(f"Training evaluation interrupted during epoch {epoch+1}")
                    self.interrupted = True
                    break
            else:
                # Use previous train correlation to speed up training
                train_correlation = self.metrics_tracker.train_correlations[-1] if self.metrics_tracker.train_correlations else 0.0
            
            # Update metrics tracker
            self.metrics_tracker.update(
                train_loss, val_loss, train_correlation, val_correlation, epoch
            )
            
            # Update learning rate scheduler
            self.scheduler.step(val_correlation)
            
            # Check for improvement
            improvement = val_correlation - self.metrics_tracker.best_val_correlation
            if improvement > self.min_delta:
                self.epochs_without_improvement = 0
                # Save best model
                if save_path:
                    # Save to output directory if specified
                    if self.output_dir:
                        models_dir = os.path.join(self.output_dir, 'models')
                        os.makedirs(models_dir, exist_ok=True)
                        save_path = os.path.join(models_dir, os.path.basename(save_path))
                    
                    self.best_model_state = self.model.state_dict().copy()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_correlation': val_correlation,
                        'model_info': model_info,
                        'training_interrupted': self.interrupted,
                        'mixed_precision': self.use_mixed_precision
                    }, save_path)
                    print(f"[SAVE] Model saved to {save_path} (correlation improved by {improvement:.4f})")
            else:
                self.epochs_without_improvement += 1
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - LR: {lr:.2e}")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Train Corr: {train_correlation:.4f}, Val Corr: {val_correlation:.4f}")
            print(f"  Best Val Corr: {self.metrics_tracker.best_val_correlation:.4f}")
            print(f"  No improvement: {self.epochs_without_improvement}/{self.patience}")
            
            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                print(f"[STOP] Early stopping triggered after {epoch+1} epochs")
                break
                
            # Check for overfitting (less frequently)
            if len(self.metrics_tracker.train_correlations) > 3 and epoch % 2 == 0:
                recent_train = np.mean(self.metrics_tracker.train_correlations[-2:])
                recent_val = np.mean(self.metrics_tracker.val_correlations[-2:])
                overfitting_gap = recent_train - recent_val
                
                if overfitting_gap > 0.2:  # Overfitting threshold
                    print(f"[WARNING] Potential overfitting detected (gap: {overfitting_gap:.3f})")
        
        total_time = time.time() - start_time
        
        # Load best model if available
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"[SUCCESS] Loaded best model from epoch {self.metrics_tracker.best_epoch + 1}")
        
        # Create training summary
        summary = self.metrics_tracker.get_summary()
        summary.update({
            'total_epochs': len(self.metrics_tracker.train_losses),
            'total_training_time': total_time,
            'model_info': model_info,
            'interrupted': self.interrupted,
            'early_stopped': self.epochs_without_improvement >= self.patience,
            'output_dir': self.output_dir,
            'mixed_precision': self.use_mixed_precision,
            'model_compiled': self.compile_model,
            'optimizations_enabled': True
        })
        
        print(f"\n{'='*50}")
        print(f"FAST Training completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
        if self.interrupted:
            print(f"[STOP] Training was interrupted by user")
        elif summary['early_stopped']:
            print(f"[STOP] Training stopped early due to no improvement")
        else:
            print(f"[SUCCESS] Training completed all epochs")
        print(f"[STATS] Best validation correlation: {summary['best_val_correlation']:.4f} "
              f"at epoch {summary['best_epoch']+1}")
        print(f"[SPEED] Average time per epoch: {total_time/summary['total_epochs']:.1f}s")
        print("="*50)
        
        return summary
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions on a dataset with optimizations.
        
        Args:
            data_loader: DataLoader for prediction data
            
        Returns:
            Array of predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Predicting', leave=False):
                rna = batch['rna'].to(self.device, non_blocking=True)
                protein = batch['protein'].to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        batch_predictions = self.model(rna, protein)
                else:
                    batch_predictions = self.model(rna, protein)
                    
                predictions.append(batch_predictions.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)


# Enhanced evaluation function with mixed precision support
def evaluate_model_fast(model, data_loader, device, use_mixed_precision=False):
    """Fast evaluation with mixed precision support."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in data_loader:
            rna = batch['rna'].to(device, non_blocking=True)
            protein = batch['protein'].to(device, non_blocking=True)
            targets = batch['score'].to(device, non_blocking=True)
            
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = model(rna, protein)
                    loss = criterion(predictions, targets)
            else:
                predictions = model(rna, protein)
                loss = criterion(predictions, targets)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            total_loss += loss.item()
    
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Pearson correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'pearson_correlation': correlation
    }
    
    return metrics, predictions, targets
