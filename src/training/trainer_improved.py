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

from .evaluation_lightweight import evaluate_model, MetricsTracker


class ImprovedRNAProteinTrainer:
    """Improved trainer with better Ctrl+C handling and anti-stalling measures."""
    
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
                 warmup_epochs: int = 0):
        """
        Initialize improved trainer with better interruption handling.
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
        
        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with less aggressive settings
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize validation correlation
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            verbose=True,
            min_lr=1e-7,  # Lower minimum LR
            threshold=1e-5,  # Smaller threshold for more sensitivity
            cooldown=2  # Cooldown period to prevent rapid changes
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Early stopping
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        # Improved interruption handling
        self.interrupted = False
        self.force_stop = False
        self._setup_signal_handlers()
        
        # Anti-stalling measures
        self.stall_detection_window = 5
        self.stall_threshold = 1e-4
        self.lr_boost_factor = 2.0
        self.last_boost_epoch = -10
        
        # Print memory info if using CUDA
        if self.device.type == 'cuda':
            self._print_memory_info("Initial")
    
    def _setup_signal_handlers(self):
        """Setup improved signal handlers for better Ctrl+C handling."""
        def signal_handler(signum, frame):
            if not self.interrupted:
                print("\n" + "="*60)
                print("🛑 INTERRUPTION DETECTED (Ctrl+C)")
                print("Finishing current batch and saving progress...")
                print("Press Ctrl+C again to force immediate stop.")
                print("="*60)
                self.interrupted = True
            else:
                print("\n" + "="*60)
                print("🚨 FORCE STOP REQUESTED")
                print("Stopping immediately...")
                print("="*60)
                self.force_stop = True
                # Force exit after a short delay
                import threading
                def force_exit():
                    time.sleep(2)
                    print("Force stopping training...")
                    os._exit(1)
                threading.Thread(target=force_exit, daemon=True).start()
        
        signal.signal(signal.SIGINT, signal_handler)
        
    def _print_memory_info(self, stage: str = "Current"):
        """Print current GPU memory usage."""
        if self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                cached = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f"[GPU] {stage} GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
            except Exception:
                pass
    
    def _detect_stalling(self) -> bool:
        """Detect if training is stalling and apply countermeasures."""
        if len(self.metrics_tracker.val_correlations) < self.stall_detection_window:
            return False
        
        recent_correlations = self.metrics_tracker.val_correlations[-self.stall_detection_window:]
        correlation_std = np.std(recent_correlations)
        correlation_trend = recent_correlations[-1] - recent_correlations[0]
        
        # Check if correlations are stagnant
        if correlation_std < self.stall_threshold and abs(correlation_trend) < self.stall_threshold:
            current_epoch = len(self.metrics_tracker.val_correlations) - 1
            if current_epoch - self.last_boost_epoch > 5:  # Wait at least 5 epochs between boosts
                print(f"🔧 STALL DETECTED: Correlation stagnant at {recent_correlations[-1]:.4f}")
                print(f"   Standard deviation: {correlation_std:.6f}")
                print(f"   Trend: {correlation_trend:.6f}")
                return True
        
        return False
    
    def _apply_anti_stall_measures(self, epoch: int):
        """Apply measures to break out of training stalls."""
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = min(current_lr * self.lr_boost_factor, self.initial_lr)
        
        print(f"🚀 APPLYING ANTI-STALL MEASURES:")
        print(f"   Boosting learning rate: {current_lr:.2e} → {new_lr:.2e}")
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Add small amount of noise to model parameters
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * 0.001 * param.std()
                    param.add_(noise)
        
        print(f"   Added parameter noise for exploration")
        self.last_boost_epoch = epoch
    
    def train_epoch(self) -> float:
        """Train for one epoch with improved interruption handling."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Check for interruption more frequently
            if self.interrupted or self.force_stop:
                print(f"\n🛑 Training interrupted at batch {batch_idx+1}")
                break
            
            # Move data to device
            rna = batch['rna'].to(self.device)
            protein = batch['protein'].to(self.device)
            targets = batch['score'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                predictions = self.model(rna, protein)
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDNN_STATUS_ALLOC_FAILED" in str(e):
                    print(f"\n[ERROR] GPU Memory Error: {e}")
                    self._print_memory_info("Before Error")
                    print("[TIP] Try reducing --batch_size, --hidden_size, or use --force_cpu")
                    raise
                else:
                    raise
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Check for force stop even during batch processing
            if self.force_stop:
                print(f"\n🚨 Force stop during batch {batch_idx+1}")
                break
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model with interruption checking."""
        if self.interrupted or self.force_stop:
            return 0.0, 0.0
        
        metrics, predictions, targets = evaluate_model(
            self.model, self.val_loader, self.device
        )
        
        return metrics['mse'], metrics['pearson_correlation']
    
    def train(self, num_epochs: int, save_path: str = None) -> Dict:
        """
        Train the model with improved interruption handling and anti-stalling.
        """
        print(f"Starting IMPROVED training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Early stopping patience: {self.patience}")
        print(f"🔧 Improved features: Better Ctrl+C handling, anti-stalling measures")
        
        # Get model info
        model_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        print(f"Model parameters: {model_info.get('total_parameters', 'Unknown')}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            if self.interrupted and not self.force_stop:
                print(f"\n🛑 Training gracefully stopped at epoch {epoch+1}")
                break
            elif self.force_stop:
                print(f"\n🚨 Training force stopped at epoch {epoch+1}")
                break
                
            epoch_start_time = time.time()
            
            # Apply learning rate warmup
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                warmup_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"Warmup LR: {warmup_lr:.6f}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            if self.interrupted or self.force_stop:
                break
            
            # Validate
            val_loss, val_correlation = self.validate()
            
            if self.interrupted or self.force_stop:
                break
            
            # Calculate training correlation (less frequently to save time)
            train_correlation = 0.0
            if epoch % 2 == 0 or epoch == num_epochs - 1:  # Every other epoch
                train_metrics, _, _ = evaluate_model(
                    self.model, self.train_loader, self.device
                )
                train_correlation = train_metrics['pearson_correlation']
            else:
                # Use previous value for speed
                train_correlation = self.metrics_tracker.train_correlations[-1] if self.metrics_tracker.train_correlations else 0.0
            
            # Update metrics tracker
            self.metrics_tracker.update(
                train_loss, val_loss, train_correlation, val_correlation, epoch
            )
            
            # Detect and handle stalling
            if self._detect_stalling():
                self._apply_anti_stall_measures(epoch)
            
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
                        'training_interrupted': self.interrupted
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
            print(f"  Epochs w/o improvement: {self.epochs_without_improvement}/{self.patience}")
            
            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                print(f"[STOP] Early stopping triggered after {epoch+1} epochs")
                break
            
            # Check for force stop between epochs
            if self.force_stop:
                break
        
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
            'force_stopped': self.force_stop,
            'early_stopped': self.epochs_without_improvement >= self.patience,
            'output_dir': self.output_dir,
            'improvements_applied': 'better_ctrl_c_handling_and_anti_stalling'
        })
        
        print(f"\n{'='*50}")
        print(f"IMPROVED Training completed in {total_time:.1f}s")
        if self.force_stop:
            print(f"[STOP] Training was force stopped by user")
        elif self.interrupted:
            print(f"[STOP] Training was gracefully interrupted by user")
        elif summary['early_stopped']:
            print(f"[STOP] Training stopped early due to no improvement")
        else:
            print(f"[SUCCESS] Training completed all epochs")
        print(f"[STATS] Best validation correlation: {summary['best_val_correlation']:.4f} "
              f"at epoch {summary['best_epoch']+1}")
        print("="*50)
        
        return summary
