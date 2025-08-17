#!/usr/bin/env python3
"""
Comprehensive Experiment Suite for Siamese Model Optimization
"""

import subprocess
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any
import itertools

class ExperimentRunner:
    def __init__(self, base_output_dir="experiments"):
        self.base_output_dir = base_output_dir
        self.results = []
        
    def run_experiment(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with given parameters"""
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT: {name}")
        print(f"{'='*60}")
        
        # Build command
        cmd = ["python", "phase2_siamese.py", "--data_dir", "src/data/"]
        
        for key, value in params.items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])
        
        print(f"🔧 Command: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            end_time = time.time()
            
            success = result.returncode == 0
            runtime = end_time - start_time
            
            # Try to extract correlation from output
            correlation = None
            if success and "Best validation correlation:" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Best validation correlation:" in line:
                        try:
                            correlation = float(line.split(':')[-1].strip())
                        except:
                            pass
            
            experiment_result = {
                'name': name,
                'params': params,
                'success': success,
                'runtime': runtime,
                'correlation': correlation,
                'timestamp': datetime.now().isoformat(),
                'stdout': result.stdout[-2000:] if result.stdout else "",  # Last 2000 chars
                'stderr': result.stderr[-1000:] if result.stderr else ""   # Last 1000 chars
            }
            
            if success and correlation:
                print(f"✅ SUCCESS: Correlation = {correlation:.4f}, Time = {runtime:.1f}s")
            else:
                print(f"❌ FAILED: {result.stderr[:200] if result.stderr else 'Unknown error'}")
                
            return experiment_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: Experiment exceeded 1 hour")
            return {
                'name': name,
                'params': params,
                'success': False,
                'runtime': 3600,
                'correlation': None,
                'error': 'timeout',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"💥 ERROR: {str(e)}")
            return {
                'name': name,
                'params': params,
                'success': False,
                'runtime': time.time() - start_time,
                'correlation': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# =============================================================================
# HYPERPARAMETER EXPERIMENTS
# =============================================================================

HYPERPARAMETER_EXPERIMENTS = [
    
    # 1. LEARNING RATE OPTIMIZATION
    {
        'name': 'lr_0001',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5, 'learning_rate': 0.001}
    },
    {
        'name': 'lr_0002',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5, 'learning_rate': 0.002}
    },
    {
        'name': 'lr_0005',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5, 'learning_rate': 0.005}
    },
    {
        'name': 'lr_0003',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5, 'learning_rate': 0.003}
    },
    
    # 2. LOSS WEIGHT OPTIMIZATION
    {
        'name': 'loss_weights_1_0.3',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5, 
                  'contrastive_weight': 1.0, 'regression_weight': 0.3}
    },
    {
        'name': 'loss_weights_1.5_0.3',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'contrastive_weight': 1.5, 'regression_weight': 0.3}
    },
    {
        'name': 'loss_weights_2_0.2',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'contrastive_weight': 2.0, 'regression_weight': 0.2}
    },
    {
        'name': 'loss_weights_0.8_0.7',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'contrastive_weight': 0.8, 'regression_weight': 0.7}
    },
    
    # 3. PAIR SAMPLING OPTIMIZATION
    {
        'name': 'pairs_0.3',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.3}
    },
    {
        'name': 'pairs_0.7',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.7}
    },
    {
        'name': 'pairs_1.0',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 1.0}
    },
    
    # 4. THRESHOLD OPTIMIZATION
    {
        'name': 'thresh_0.8_0.2',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'positive_threshold': 0.8, 'negative_threshold': 0.2}
    },
    {
        'name': 'thresh_0.6_0.4',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'positive_threshold': 0.6, 'negative_threshold': 0.4}
    },
    {
        'name': 'thresh_0.75_0.25',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'positive_threshold': 0.75, 'negative_threshold': 0.25}
    },
    
    # 5. ARCHITECTURE OPTIMIZATION
    {
        'name': 'arch_hidden_96',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'rna_hidden_size': 96}
    },
    {
        'name': 'arch_hidden_160',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'rna_hidden_size': 160}
    },
    {
        'name': 'arch_emb_128',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'embedding_dim': 128}
    },
    {
        'name': 'arch_emb_384',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'embedding_dim': 384}
    },
    
    # 6. TEMPERATURE OPTIMIZATION  
    {
        'name': 'temp_0.05',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'temperature': 0.05}
    },
    {
        'name': 'temp_0.2',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'temperature': 0.2}
    },
    {
        'name': 'temp_0.07',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'temperature': 0.07}
    },
    
    # 7. EXTENDED TRAINING
    {
        'name': 'epochs_7',
        'params': {'subset_size': 500, 'epochs': 7, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'patience': 5}
    },
    {
        'name': 'epochs_10',
        'params': {'subset_size': 500, 'epochs': 10, 'loss_type': 'hybrid', 'pair_sampling_ratio': 0.5,
                  'patience': 6}
    },
    
    # 8. PURE CONTRASTIVE vs HYBRID
    {
        'name': 'pure_contrastive',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'contrastive', 'pair_sampling_ratio': 0.5}
    },
    {
        'name': 'infonce_loss',
        'params': {'subset_size': 500, 'epochs': 5, 'loss_type': 'infonce', 'pair_sampling_ratio': 0.5}
    },
]

def run_hyperparameter_experiments():
    """Run all hyperparameter experiments"""
    runner = ExperimentRunner("hyperparameter_experiments")
    results = []
    
    print(f"🚀 Starting {len(HYPERPARAMETER_EXPERIMENTS)} hyperparameter experiments...")
    
    for i, exp in enumerate(HYPERPARAMETER_EXPERIMENTS, 1):
        print(f"\n📊 Progress: {i}/{len(HYPERPARAMETER_EXPERIMENTS)}")
        result = runner.run_experiment(exp['name'], exp['params'])
        results.append(result)
        
        # Save intermediate results
        with open(f"hyperparameter_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary
    successful = [r for r in results if r['success'] and r['correlation']]
    if successful:
        best = max(successful, key=lambda x: x['correlation'])
        print(f"\n🏆 BEST HYPERPARAMETERS:")
        print(f"   Experiment: {best['name']}")
        print(f"   Correlation: {best['correlation']:.4f}")
        print(f"   Params: {best['params']}")
    
    return results

if __name__ == "__main__":
    results = run_hyperparameter_experiments()
    print(f"\n✅ Completed {len(results)} experiments")
