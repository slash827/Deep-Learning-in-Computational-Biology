import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime


def calculate_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Pearson correlation coefficient
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return 0.0
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return 0.0
    
    # Check for constant arrays
    if np.std(y_true_clean) == 0 or np.std(y_pred_clean) == 0:
        return 0.0
    
    try:
        correlation, _ = pearsonr(y_true_clean, y_pred_clean)
        return correlation if not np.isnan(correlation) else 0.0
    except Exception:
        return 0.0


def calculate_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Spearman correlation coefficient."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return 0.0
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return 0.0
    
    try:
        correlation, _ = spearmanr(y_true_clean, y_pred_clean)
        return correlation if not np.isnan(correlation) else 0.0
    except Exception:
        return 0.0


def evaluate_model(model: torch.nn.Module, 
                  data_loader: torch.utils.data.DataLoader, 
                  device: torch.device,
                  criterion: torch.nn.Module = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a model on a dataset with lightweight dependencies.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run evaluation on
        criterion: Loss function (optional)
        
    Returns:
        Tuple of (metrics_dict, predictions, targets)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    if criterion is None:
        criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in data_loader:
            # Move data to device
            rna = batch['rna'].to(device)
            protein = batch['protein'].to(device)
            targets = batch['score'].to(device)
            
            # Forward pass
            predictions = model(rna, protein)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and targets
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    predictions_array = np.array(all_predictions)
    targets_array = np.array(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(targets_array, predictions_array)
    mae = mean_absolute_error(targets_array, predictions_array)
    pearson_corr = calculate_pearson_correlation(targets_array, predictions_array)
    spearman_corr = calculate_spearman_correlation(targets_array, predictions_array)
    r2 = r2_score(targets_array, predictions_array)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'r2_score': r2,
        'avg_loss': total_loss / num_batches if num_batches > 0 else 0.0
    }
    
    return metrics, predictions_array, targets_array


def save_predictions_plot(predictions: np.ndarray, 
                         targets: np.ndarray, 
                         save_path: str,
                         title: str = "Predictions vs Targets") -> bool:
    """
    Save a scatter plot of predictions vs targets with lazy matplotlib import.
    
    Args:
        predictions: Predicted values
        targets: True values  
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        True if plot was saved successfully, False otherwise
    """
    try:
        # Lazy import to avoid heavy dependencies during regular imports
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        plt.scatter(targets, predictions, alpha=0.6, s=20)
        
        # Calculate correlation for display
        correlation = calculate_pearson_correlation(targets, predictions)
        
        # Plot perfect prediction line
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{title}\nPearson Correlation: {correlation:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Warning: Could not save plot to {save_path}: {e}")
        return False


class MetricsTracker:
    """
    Simple metrics tracker without heavy plotting dependencies.
    """
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_correlations = []
        self.val_correlations = []
        self.best_val_correlation = -1.0
        self.best_epoch = 0
        self.epochs = []
    
    def update(self, train_loss: float, val_loss: float, 
               train_correlation: float, val_correlation: float, epoch: int):
        """Update metrics for current epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_correlations.append(train_correlation)
        self.val_correlations.append(val_correlation)
        self.epochs.append(epoch)
        
        if val_correlation > self.best_val_correlation:
            self.best_val_correlation = val_correlation
            self.best_epoch = epoch
    
    def get_summary(self) -> Dict:
        """Get summary of training metrics."""
        return {
            'best_val_correlation': self.best_val_correlation,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0.0,
            'final_train_correlation': self.train_correlations[-1] if self.train_correlations else 0.0,
            'final_val_correlation': self.val_correlations[-1] if self.val_correlations else 0.0,
            'total_epochs': len(self.train_losses)
        }
    
    def save_training_plots(self, output_dir: str) -> bool:
        """
        Save training history plots with lazy import.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            True if plots were saved successfully, False otherwise
        """
        try:
            # Lazy import to avoid heavy dependencies
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Create plots directory
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            epochs = list(range(1, len(self.train_losses) + 1))
            
            # Plot losses
            ax1.plot(epochs, self.train_losses, label='Train Loss', marker='o', markersize=3)
            ax1.plot(epochs, self.val_losses, label='Val Loss', marker='s', markersize=3)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot correlations
            ax2.plot(epochs, self.train_correlations, label='Train Correlation', marker='o', markersize=3)
            ax2.plot(epochs, self.val_correlations, label='Val Correlation', marker='s', markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Pearson Correlation')
            ax2.set_title('Training and Validation Correlation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot loss vs correlation
            ax3.scatter(self.train_losses, self.train_correlations, alpha=0.6, label='Train', s=20)
            ax3.scatter(self.val_losses, self.val_correlations, alpha=0.6, label='Val', s=20)
            ax3.set_xlabel('Loss')
            ax3.set_ylabel('Correlation')
            ax3.set_title('Loss vs Correlation')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot improvement over time
            best_so_far = []
            current_best = -1.0
            for val_corr in self.val_correlations:
                if val_corr > current_best:
                    current_best = val_corr
                best_so_far.append(current_best)
            
            ax4.plot(epochs, best_so_far, label='Best Val Correlation', marker='o', markersize=3)
            ax4.axhline(y=self.best_val_correlation, color='r', linestyle='--', alpha=0.7, 
                       label=f'Overall Best: {self.best_val_correlation:.4f}')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Best Correlation So Far')
            ax4.set_title('Best Validation Correlation Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plots_dir, 'training_history.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Training plots saved to {plot_path}")
            return True
            
        except Exception as e:
            print(f"Warning: Could not save training plots: {e}")
            return False


def calculate_detailed_metrics(predictions: np.ndarray, 
                             targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate detailed evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Correlation metrics
    pearson_corr = calculate_pearson_correlation(targets, predictions)
    spearman_corr = calculate_spearman_correlation(targets, predictions)
    
    # Additional metrics
    rmse = np.sqrt(mse)
    
    # Calculate relative errors
    relative_errors = np.abs(targets - predictions) / (np.abs(targets) + 1e-8)
    mean_relative_error = np.mean(relative_errors)
    
    # Calculate prediction statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    target_mean = np.mean(targets)
    target_std = np.std(targets)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'mean_relative_error': mean_relative_error,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'pred_target_mean_diff': abs(pred_mean - target_mean),
        'pred_target_std_ratio': pred_std / (target_std + 1e-8)
    }
