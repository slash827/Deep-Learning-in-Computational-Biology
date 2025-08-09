import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def create_run_directory(base_dir: str = "runs", run_name: str = None) -> str:
    """
    Create a unique directory for this training run.
    
    Args:
        base_dir: Base directory for all runs
        run_name: Optional custom name for the run
        
    Returns:
        Path to the created run directory
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"phase1_run_{timestamp}"
    
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories including configs
    subdirs = ['models', 'plots', 'logs', 'metrics', 'configs']
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    return run_dir


def save_training_config(config: dict, output_dir: str):
    """Save training configuration to a file."""
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")


def save_training_summary(summary: dict, output_dir: str):
    """Save training summary to a file."""
    # Convert any numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj
    
    json_summary = {}
    for key, value in summary.items():
        json_summary[key] = convert_numpy(value)
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"Training summary saved to: {summary_path}")


def save_comprehensive_run_report(config: Dict[str, Any], 
                                 model_info: Dict[str, Any],
                                 training_summary: Dict[str, Any],
                                 validation_metrics: Dict[str, Any],
                                 run_dir: str,
                                 timestamp: Optional[str] = None,
                                 phase: str = "Phase 1 - Basic BiLSTM") -> str:
    """
    Save a comprehensive run report with all configuration, model, and result information.
    
    Args:
        config: Training configuration
        model_info: Model architecture information
        training_summary: Training results summary
        validation_metrics: Final validation metrics
        run_dir: Directory to save the report
        timestamp: Optional custom timestamp, otherwise uses current time
        phase: Phase description for the experiment
        
    Returns:
        Path to the saved report file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    # Create comprehensive report
    report = {
        "experiment_info": {
            "timestamp": timestamp,
            "run_name": config.get('run_name', 'unknown'),
            "phase": phase,
            "description": "RNA-Protein Binding Prediction using Enhanced BiLSTM Architecture"
        },
        "system_info": {
            "device": config.get('device', 'unknown'),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_gb": None
        },
        "data_configuration": {
            "subset_size": config.get('subset_size'),
            "batch_size": config.get('batch_size'),
            "validation_split": 0.2,  # Fixed in current implementation
            "max_rna_length": config.get('max_rna_length'),
            "max_protein_length": config.get('max_protein_length'),
            "rna_input_size": 5,  # A, U, G, C, N
            "protein_input_size": 21  # 20 amino acids + unknown
        },
        "model_architecture": {
            "model_type": model_info.get('model_name', 'BasicLSTM'),
            "total_parameters": model_info.get('total_parameters'),
            "trainable_parameters": model_info.get('trainable_parameters'),
            "rna_hidden_size": config.get('hidden_size'),
            "protein_hidden_size": config.get('hidden_size'),
            "num_layers": config.get('num_layers'),
            "dropout": config.get('dropout'),
            "bidirectional": True,
            "fusion_hidden_size": max(config.get('hidden_size', 64) * 2, 128),
            # Attention parameters (if present)
            **({
                "attention_heads": config.get('attention_heads'),
                "attention_dropout": config.get('attention_dropout'),
                "use_positional_encoding": config.get('use_positional_encoding', False)
            } if config.get('attention_heads') else {})
        },
        "training_hyperparameters": {
            "epochs": config.get('epochs'),
            "learning_rate": config.get('learning_rate'),
            "weight_decay": 1e-4,  # Fixed in current implementation
            "optimizer": "Adam",
            "loss_function": "MSE",
            "early_stopping": {
                "patience": config.get('patience'),
                "min_delta": config.get('min_delta')
            },
            "gradient_clipping": {
                "max_grad_norm": config.get('max_grad_norm')
            },
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "patience": config.get('lr_scheduler_patience'),
                "factor": config.get('lr_scheduler_factor')
            },
            # Warmup parameters (if present)
            **({
                "warmup_epochs": config.get('warmup_epochs')
            } if config.get('warmup_epochs', 0) > 0 else {})
        },
        "training_results": convert_numpy({
            "best_epoch": training_summary.get('best_epoch'),
            "total_epochs_trained": training_summary.get('total_epochs', 'unknown'),
            "early_stopped": training_summary.get('early_stopped', False),
            "interrupted": training_summary.get('interrupted', False),
            "training_time_seconds": training_summary.get('total_training_time'),
            "best_validation_correlation": training_summary.get('best_val_correlation'),
            "final_training_correlation": training_summary.get('final_train_correlation'),
            "final_validation_correlation": training_summary.get('final_val_correlation'),
            "best_validation_loss": training_summary.get('best_val_loss'),
            "final_training_loss": training_summary.get('final_train_loss'),
            "final_validation_loss": training_summary.get('final_val_loss')
        }),
        "validation_metrics": convert_numpy(validation_metrics),
        "performance_analysis": {
            "overfitting_assessment": _assess_overfitting(training_summary),
            "convergence_status": _assess_convergence(training_summary),
            "performance_tier": _assess_performance_tier(training_summary.get('best_val_correlation', 0))
        },
        "file_paths": {
            "model_checkpoint": f"models/{config.get('save_model', 'model.pth')}",
            "training_plots": "plots/training_history.png",
            "prediction_plot": "plots/predictions_vs_targets.png",
            "config_file": "config.json",
            "training_summary": "training_summary.json"
        }
    }
    
    # Add GPU memory info if available
    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            report["system_info"]["gpu_memory_gb"] = round(gpu_memory, 1)
        except:
            pass
    
    # Save timestamped report
    report_filename = f"experiment_report_{timestamp}.json"
    report_path = os.path.join(run_dir, 'configs', report_filename)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save a latest report (without timestamp) for easy access
    latest_path = os.path.join(run_dir, 'configs', 'latest_experiment_report.json')
    with open(latest_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Comprehensive experiment report saved to: {report_filename}")
    print(f"Latest report also saved to: latest_experiment_report.json")
    
    return report_path


def _assess_overfitting(training_summary: Dict[str, Any]) -> str:
    """Assess overfitting based on training vs validation performance."""
    train_corr = training_summary.get('final_train_correlation', 0)
    val_corr = training_summary.get('final_val_correlation', 0)
    
    if train_corr == 0 or val_corr == 0:
        return "unknown"
    
    gap = train_corr - val_corr
    
    if gap < 0.02:
        return "no_overfitting"
    elif gap < 0.05:
        return "minimal_overfitting"
    elif gap < 0.1:
        return "moderate_overfitting"
    else:
        return "significant_overfitting"


def _assess_convergence(training_summary: Dict[str, Any]) -> str:
    """Assess whether training converged properly."""
    early_stopped = training_summary.get('early_stopped', False)
    interrupted = training_summary.get('interrupted', False)
    total_epochs = training_summary.get('total_epochs', 0)
    best_epoch = training_summary.get('best_epoch', 0)
    
    if interrupted:
        return "interrupted"
    elif early_stopped:
        return "converged_early"
    elif best_epoch == total_epochs - 1:  # Best epoch was the last one
        return "may_need_more_epochs"
    else:
        return "converged_normally"


def _assess_performance_tier(correlation: float) -> str:
    """Assess performance tier based on correlation."""
    if correlation >= 0.7:
        return "excellent"
    elif correlation >= 0.6:
        return "very_good"
    elif correlation >= 0.5:
        return "good"
    elif correlation >= 0.4:
        return "moderate"
    elif correlation >= 0.3:
        return "fair"
    else:
        return "poor"


def get_latest_run_dir(base_dir: str = "runs") -> str:
    """Get the path to the most recent run directory."""
    if not os.path.exists(base_dir):
        return None
    
    run_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    if not run_dirs:
        return None
    
    # Sort by creation time
    run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(base_dir, x)), reverse=True)
    
    return os.path.join(base_dir, run_dirs[0])
