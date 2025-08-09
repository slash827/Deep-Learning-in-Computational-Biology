#!/usr/bin/env python3
"""
Quick Phase 2 Experiments Runner
Simple and fast experiments for parameter testing
"""

import subprocess
import time
import json
from datetime import datetime

def main():
    print("🚀 Starting Quick Phase 2 Experiments")
    print("📊 Total experiments: 4")
    print("⏱️ Estimated time: 8 minutes")
    
    # Quick experiment configurations
    experiments = [
        {
            'name': 'tiny_baseline',
            'description': 'Very small baseline',
            'args': [
                '--subset_size', '25',
                '--batch_size', '4', 
                '--epochs', '1',
                '--hidden_size', '8',
                '--attention_heads', '1',
                '--max_protein_length', '30',
                '--max_rna_length', '15',
                '--force_cpu'
            ]
        },
        {
            'name': 'small_model',
            'description': 'Small model test',
            'args': [
                '--subset_size', '25',
                '--batch_size', '4',
                '--epochs', '1', 
                '--hidden_size', '16',
                '--attention_heads', '2',
                '--max_protein_length', '30',
                '--max_rna_length', '15',
                '--force_cpu'
            ]
        },
        {
            'name': 'more_samples',
            'description': 'More training samples',
            'args': [
                '--subset_size', '50',
                '--batch_size', '4',
                '--epochs', '1',
                '--hidden_size', '8', 
                '--attention_heads', '1',
                '--max_protein_length', '30',
                '--max_rna_length', '15',
                '--force_cpu'
            ]
        },
        {
            'name': 'higher_dropout',
            'description': 'Test regularization',
            'args': [
                '--subset_size', '25',
                '--batch_size', '4',
                '--epochs', '1',
                '--hidden_size', '16',
                '--attention_heads', '2', 
                '--dropout', '0.5',
                '--attention_dropout', '0.3',
                '--max_protein_length', '30',
                '--max_rna_length', '15',
                '--force_cpu'
            ]
        }
    ]
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print("=" * 60)
        print(f"🧪 EXPERIMENT {i}/4: {exp['name'].upper()}")
        print(f"📝 Description: {exp['description']}")
        print("=" * 60)
        
        # Build command
        run_name = f"quick_{exp['name']}_{datetime.now().strftime('%H%M%S')}"
        cmd = [
            'python', 'phase2_simple.py',
            '--run_name', run_name
        ] + exp['args']
        
        cmd_str = ' '.join(cmd)
        print(f"🏃 Running: {cmd_str}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS in {duration:.1f}s")
                
                # Try to extract validation correlation from output
                val_corr = "N/A"
                for line in result.stdout.split('\n'):
                    if 'Best validation correlation:' in line:
                        val_corr = line.split(':')[-1].strip()
                        break
                
                results.append({
                    'name': exp['name'],
                    'description': exp['description'],
                    'status': 'SUCCESS',
                    'duration': duration,
                    'validation_correlation': val_corr,
                    'run_name': run_name
                })
                
                print(f"📊 Validation correlation: {val_corr}")
                
            else:
                print(f"❌ FAILED in {duration:.1f}s")
                error_msg = result.stderr[-500:] if result.stderr else "No error message"
                results.append({
                    'name': exp['name'],
                    'description': exp['description'], 
                    'status': 'FAILED',
                    'duration': duration,
                    'validation_correlation': 'N/A',
                    'error': error_msg
                })
                print(f"Error: {error_msg}")
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ TIMEOUT after {duration:.1f}s")
            results.append({
                'name': exp['name'],
                'description': exp['description'],
                'status': 'TIMEOUT', 
                'duration': duration,
                'validation_correlation': 'N/A',
                'error': 'Timeout after 5 minutes'
            })
        
        if i < len(experiments):
            print("⏸️ Waiting 5 seconds before next experiment...")
            time.sleep(5)
    
    # Summary
    print("=" * 60)
    print("📊 QUICK EXPERIMENTS SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] != 'SUCCESS']
    
    print(f"✅ Successful: {len(successful)}/{len(experiments)}")
    print(f"❌ Failed: {len(failed)}/{len(experiments)}")
    
    if successful:
        print("\n🏆 SUCCESSFUL EXPERIMENTS:")
        for result in successful:
            print(f"  - {result['name']}: {result['validation_correlation']} correlation ({result['duration']:.1f}s)")
    
    if failed:
        print("\n💥 FAILED EXPERIMENTS:")
        for result in failed:
            print(f"  - {result['name']}: {result['status']}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"quick_experiment_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_experiments': len(experiments),
            'successful': len(successful),
            'failed': len(failed),
            'results': results
        }, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
