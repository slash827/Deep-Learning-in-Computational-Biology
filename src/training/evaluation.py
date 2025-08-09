import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plot display
import matplotlib.pyplot as plt
import seaborn as sns
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
    correlation, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return correlation if not np.isnan(correlation) else 0.0


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    metrics = {
        'pearson_correlation': calculate_pearson_correlation(y_true, y_pred),
        'mse': mean_squared_error(y_true_flat, y_pred_flat),
        'mae': mean_absolute_error(y_true_flat, y_pred_flat),
        'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    }
    
    return metrics


def evaluate_model(model: torch.nn.Module, 
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            rna = batch['rna'].to(device)
            protein = batch['protein'].to(device)
            targets = batch['score'].to(device)
            
            predictions = model(rna, protein)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(targets, predictions)
    
    return metrics, predictions, targets


def plot_predictions_vs_targets(targets: np.ndarray, 
                               predictions: np.ndarray,
                               title: str = "Predictions vs Targets",
                               save_path: str = None,
                               output_dir: str = None) -> None:
    """
    Plot predictions vs targets.
    
    Args:
        targets: True values
        predictions: Predicted values
        title: Plot title
        save_path: Path to save the plot
        output_dir: Directory to save the plot (alternative to save_path)
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.5, s=1)
    
    # Add diagonal line (perfect predictions)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Calculate correlation
    correlation = calculate_pearson_correlation(targets, predictions)
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title}\nPearson Correlation: {correlation:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Determine save path
    if output_dir and not save_path:
        save_path = os.path.join(output_dir, 'predictions_vs_targets.png')
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close()  # Close figure to free memory instead of showing


def plot_training_history(train_losses: List[float], 
                         val_losses: List[float],
                         train_correlations: List[float] = None,
                         val_correlations: List[float] = None,
                         save_path: str = None,
                         output_dir: str = None) -> None:
    """
    Plot training history.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_correlations: Training correlations (optional)
        val_correlations: Validation correlations (optional)
        save_path: Path to save the plot
        output_dir: Directory to save the plot (alternative to save_path)
    """
    fig, axes = plt.subplots(1, 2 if train_correlations is not None else 1, figsize=(15, 5))
    
    if train_correlations is not None:
        # Plot losses
        axes[0].plot(train_losses, label='Training Loss', color='blue')
        axes[0].plot(val_losses, label='Validation Loss', color='orange')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot correlations
        axes[1].plot(train_correlations, label='Training Correlation', color='blue')
        axes[1].plot(val_correlations, label='Validation Correlation', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Pearson Correlation')
        axes[1].set_title('Training and Validation Correlation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Only plot losses
        if isinstance(axes, np.ndarray):
            ax = axes[0]
        else:
            ax = axes
        ax.plot(train_losses, label='Training Loss', color='blue')
        ax.plot(val_losses, label='Validation Loss', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Determine save path
    if output_dir and not save_path:
        save_path = os.path.join(output_dir, 'training_history.png')
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.close()  # Close figure to free memory instead of showing


class MetricsTracker:
    """Track and store training metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_correlations = []
        self.val_correlations = []
        self.best_val_correlation = -np.inf
        self.best_epoch = 0
    
    def update(self, train_loss: float, val_loss: float, 
               train_correlation: float, val_correlation: float, epoch: int):
        """Update metrics."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_correlations.append(train_correlation)
        self.val_correlations.append(val_correlation)
        
        if val_correlation > self.best_val_correlation:
            self.best_val_correlation = val_correlation
            self.best_epoch = epoch
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        return {
            'best_val_correlation': self.best_val_correlation,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'final_train_correlation': self.train_correlations[-1] if self.train_correlations else None,
            'final_val_correlation': self.val_correlations[-1] if self.val_correlations else None
        }
    
    def plot_history(self, save_path: str = None, output_dir: str = None):
        """Plot training history."""
        plot_training_history(
            self.train_losses, 
            self.val_losses,
            self.train_correlations,
            self.val_correlations,
            save_path,
            output_dir
        )
