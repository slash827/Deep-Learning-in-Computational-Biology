#!/usr/bin/env python3
"""
Multiple Small Experiments Runner
Systematically test different configurations to find optimal settings
"""

import subprocess
import time
import json
import os
from datetime import datetime

# Base safe settings that worked
BASE_SETTINGS = {
    'subset_size': 50,
    'batch_size': 2,
    'epochs': 1,
    'force_cpu': True,
    'max_protein_length': 50,
    'max_rna_length': 20
}

# Define experiment configurations
EXPERIMENTS = [
    # Experiment 1: Baseline (what we know works)
    {
        'name': 'baseline',
        'description': 'Minimal working configuration',
        'params': {
            'hidden_size': 16,
            'attention_heads': 2,
            'dropout': 0.3,
            'attention_dropout': 0.1
        }
    },
    
    # Experiment 2: Larger model
    {
        'name': 'larger_model',
        'description': 'Increase model size',
        'params': {
            'hidden_size': 32,
            'attention_heads': 4,
            'dropout': 0.3,
            'attention_dropout': 0.1
        }
    },
    
    # Experiment 3: More data
    {
        'name': 'more_data',
        'description': 'Increase dataset size',
        'params': {
            'hidden_size': 16,
            'attention_heads': 2,
            'subset_size': 100,
            'dropout': 0.3,
            'attention_dropout': 0.1
        }
    },
    
    # Experiment 4: Larger batch
    {
        'name': 'larger_batch',
        'description': 'Increase batch size',
        'params': {
            'hidden_size': 16,
            'attention_heads': 2,
            'batch_size': 4,
            'dropout': 0.3,
            'attention_dropout': 0.1
        }
    },
    
    # Experiment 5: Longer sequences
    {
        'name': 'longer_sequences',
        'description': 'Increase sequence lengths',
        'params': {
            'hidden_size': 16,
            'attention_heads': 2,
            'max_protein_length': 100,
            'max_rna_length': 40,
            'dropout': 0.3,
            'attention_dropout': 0.1
        }
    },
    
    # Experiment 6: Multiple epochs
    {
        'name': 'multiple_epochs',
        'description': 'Train for multiple epochs',
        'params': {
            'hidden_size': 16,
            'attention_heads': 2,
            'epochs': 3,
            'dropout': 0.3,
            'attention_dropout': 0.1
        }
    },
    
    # Experiment 7: High dropout for regularization
    {
        'name': 'high_dropout',
        'description': 'Test higher dropout rates',
        'params': {
            'hidden_size': 32,
            'attention_heads': 4,
            'dropout': 0.5,
            'attention_dropout': 0.3,
            'subset_size': 100
        }
    },
    
    # Experiment 8: Maximum safe settings
    {
        'name': 'max_safe',
        'description': 'Push limits while staying safe',
        'params': {
            'hidden_size': 48,
            'attention_heads': 6,
            'subset_size': 150,
            'batch_size': 4,
            'epochs': 2,
            'max_protein_length': 75,
            'max_rna_length': 30,
            'dropout': 0.4,
            'attention_dropout': 0.2
        }
    }
]

def build_command(experiment):
    """Build the command string for an experiment"""
    cmd = ["python", "phase2_simple.py"]
    
    # Add base settings
    for key, value in BASE_SETTINGS.items():
        if key == 'force_cpu':
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    # Add experiment-specific settings
    for key, value in experiment['params'].items():
        if key in BASE_SETTINGS and key != 'force_cpu':
            # Skip if already added from base settings
            continue
        cmd.extend([f"--{key}", str(value)])
    
    # Add run name
    timestamp = datetime.now().strftime("%H%M%S")
    run_name = f"exp_{experiment['name']}_{timestamp}"
    cmd.extend(["--run_name", run_name])
    
    return cmd, run_name

def run_experiment(experiment, exp_num, total_exps):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT {exp_num}/{total_exps}: {experiment['name'].upper()}")
    print(f"📝 Description: {experiment['description']}")
    print(f"⚙️ Parameters: {experiment['params']}")
    print(f"{'='*60}")
    
    cmd, run_name = build_command(experiment)
    
    print(f"🏃 Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS in {duration:.1f}s")
            
            # Extract validation correlation from output
            output_lines = result.stdout.split('\n')
            val_corr = None
            for line in output_lines:
                if 'Best validation correlation:' in line:
                    try:
                        val_corr = float(line.split(':')[1].strip())
                        break
                    except:
                        pass
            
            return {
                'name': experiment['name'],
                'description': experiment['description'],
                'params': experiment['params'],
                'status': 'SUCCESS',
                'duration': duration,
                'validation_correlation': val_corr,
                'run_name': run_name,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr
            }
        else:
            print(f"❌ FAILED in {duration:.1f}s")
            print(f"Error: {result.stderr[:500]}")
            
            return {
                'name': experiment['name'],
                'description': experiment['description'], 
                'params': experiment['params'],
                'status': 'FAILED',
                'duration': duration,
                'validation_correlation': None,
                'run_name': run_name,
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[:1000]
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after 5 minutes")
        return {
            'name': experiment['name'],
            'description': experiment['description'],
            'params': experiment['params'], 
            'status': 'TIMEOUT',
            'duration': 300,
            'validation_correlation': None,
            'run_name': run_name,
            'stdout': '',
            'stderr': 'Timeout after 5 minutes'
        }
    
    except Exception as e:
        print(f"💥 ERROR: {str(e)}")
        return {
            'name': experiment['name'],
            'description': experiment['description'],
            'params': experiment['params'],
            'status': 'ERROR', 
            'duration': time.time() - start_time,
            'validation_correlation': None,
            'run_name': run_name,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    print("🔬 Starting Multiple Small Experiments")
    print(f"📊 Total experiments: {len(EXPERIMENTS)}")
    print(f"⏱️ Estimated time: {len(EXPERIMENTS) * 2} minutes")
    
    results = []
    
    for i, experiment in enumerate(EXPERIMENTS, 1):
        result = run_experiment(experiment, i, len(EXPERIMENTS))
        results.append(result)
        
        # Short break between experiments
        if i < len(EXPERIMENTS):
            print("⏸️ Waiting 10 seconds before next experiment...")
            time.sleep(10)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiment_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] != 'SUCCESS']
    
    print(f"✅ Successful: {len(successful)}/{len(EXPERIMENTS)}")
    print(f"❌ Failed: {len(failed)}/{len(EXPERIMENTS)}")
    
    if successful:
        print(f"\n🏆 TOP PERFORMERS:")
        successful_with_corr = [r for r in successful if r['validation_correlation'] is not None]
        if successful_with_corr:
            sorted_results = sorted(successful_with_corr, key=lambda x: x['validation_correlation'], reverse=True)
            
            for i, result in enumerate(sorted_results[:3], 1):
                print(f"  {i}. {result['name']}: {result['validation_correlation']:.4f} correlation")
                print(f"     Duration: {result['duration']:.1f}s")
                print(f"     Params: {result['params']}")
                print()
    
    if failed:
        print(f"\n💥 FAILED EXPERIMENTS:")
        for result in failed:
            print(f"  - {result['name']}: {result['status']}")
            if result['stderr']:
                print(f"    Error: {result['stderr'][:100]}...")
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"🔍 Check individual run directories in 'runs/' for detailed outputs")

if __name__ == "__main__":
    main()
