#!/usr/bin/env python3
"""
Master Experiment Runner - Run All Experiments Systematically
"""

import json
import time
from datetime import datetime
from experiment_suite import run_hyperparameter_experiments
from proteinbert_experiments import run_proteinbert_experiments

def main():
    print("🚀 COMPREHENSIVE SIAMESE MODEL OPTIMIZATION")
    print("=" * 60)
    
    all_results = {
        'start_time': datetime.now().isoformat(),
        'hyperparameter_experiments': [],
        'proteinbert_experiments': [],
        'summary': {}
    }
    
    # Phase 1: Hyperparameter Optimization
    print("\n📊 PHASE 1: HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    
    start_time = time.time()
    hyperparam_results = run_hyperparameter_experiments()
    hyperparam_time = time.time() - start_time
    
    all_results['hyperparameter_experiments'] = hyperparam_results
    
    # Find best hyperparameters
    successful_hyperparam = [r for r in hyperparam_results if r['success'] and r['correlation']]
    if successful_hyperparam:
        best_hyperparam = max(successful_hyperparam, key=lambda x: x['correlation'])
        print(f"\n🏆 BEST HYPERPARAMETERS FOUND:")
        print(f"   Correlation: {best_hyperparam['correlation']:.4f}")
        print(f"   Config: {best_hyperparam['params']}")
        
        all_results['summary']['best_hyperparameters'] = {
            'correlation': best_hyperparam['correlation'],
            'params': best_hyperparam['params'],
            'name': best_hyperparam['name']
        }
    
    # Phase 2: ProteinBERT Optimization
    print(f"\n🧬 PHASE 2: PROTEINBERT OPTIMIZATION")
    print("-" * 40)
    
    start_time = time.time()
    proteinbert_results = run_proteinbert_experiments()
    proteinbert_time = time.time() - start_time
    
    all_results['proteinbert_experiments'] = proteinbert_results
    
    # Find best ProteinBERT config
    successful_proteinbert = [r for r in proteinbert_results if r['success'] and r['correlation']]
    if successful_proteinbert:
        best_proteinbert = max(successful_proteinbert, key=lambda x: x['correlation'])
        print(f"\n🏆 BEST PROTEINBERT CONFIG FOUND:")
        print(f"   Correlation: {best_proteinbert['correlation']:.4f}")
        print(f"   Name: {best_proteinbert['name']}")
        
        all_results['summary']['best_proteinbert'] = {
            'correlation': best_proteinbert['correlation'],
            'name': best_proteinbert['name'],
            'embedding_path': best_proteinbert['embedding_path']
        }
    
    # Final Summary
    all_results['end_time'] = datetime.now().isoformat()
    all_results['summary']['total_experiments'] = len(hyperparam_results) + len(proteinbert_results)
    all_results['summary']['hyperparam_time'] = hyperparam_time
    all_results['summary']['proteinbert_time'] = proteinbert_time
    all_results['summary']['total_time'] = hyperparam_time + proteinbert_time
    
    # Find overall best
    all_successful = successful_hyperparam + successful_proteinbert
    if all_successful:
        overall_best = max(all_successful, key=lambda x: x['correlation'])
        all_results['summary']['overall_best'] = {
            'correlation': overall_best['correlation'],
            'experiment_type': 'hyperparameter' if overall_best in successful_hyperparam else 'proteinbert',
            'name': overall_best['name']
        }
        
        print(f"\n🎯 OVERALL BEST RESULT:")
        print(f"   Correlation: {overall_best['correlation']:.4f}")
        print(f"   Type: {'Hyperparameter' if overall_best in successful_hyperparam else 'ProteinBERT'}")
        print(f"   Name: {overall_best['name']}")
    
    # Save comprehensive results
    results_file = f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📄 All results saved to: {results_file}")
    print(f"⏱️  Total runtime: {(hyperparam_time + proteinbert_time)/3600:.2f} hours")
    
    return all_results

if __name__ == "__main__":
    results = main()
