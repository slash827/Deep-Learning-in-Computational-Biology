@echo off
echo ============================================================
echo ⚡ QUICK SIAMESE MODEL EXPERIMENTS (Top Candidates Only)
echo ============================================================
echo.

REM Create results directory
if not exist "quick_results" mkdir quick_results

REM Set timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"

echo 🚀 Running top 6 most promising experiments...
echo Estimated time: ~6 hours (50 min each)
echo Results will be saved with timestamp: %timestamp%
echo.

REM ============================================
REM QUICK HIGH-IMPACT EXPERIMENTS
REM ============================================

echo [1/6] 🔧 Optimized Learning Rate (0.002)
echo Starting: %time%
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --learning_rate 0.002 > quick_results/quick_lr_opt_%timestamp%.log 2>&1
echo Completed: %time%
if %errorlevel% equ 0 (echo ✅ SUCCESS) else (echo ❌ FAILED)
echo.

echo [2/6] ⚖️ Optimized Loss Weights (1.5/0.3)
echo Starting: %time%
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --contrastive_weight 1.5 --regression_weight 0.3 > quick_results/quick_loss_opt_%timestamp%.log 2>&1
echo Completed: %time%
if %errorlevel% equ 0 (echo ✅ SUCCESS) else (echo ❌ FAILED)
echo.

echo [3/6] 🎯 Stricter Thresholds (0.8/0.2)
echo Starting: %time%
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --positive_threshold 0.8 --negative_threshold 0.2 > quick_results/quick_thresh_%timestamp%.log 2>&1
echo Completed: %time%
if %errorlevel% equ 0 (echo ✅ SUCCESS) else (echo ❌ FAILED)
echo.

echo [4/6] 🌡️ Lower Temperature (0.07)
echo Starting: %time%
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --temperature 0.07 > quick_results/quick_temp_%timestamp%.log 2>&1
echo Completed: %time%
if %errorlevel% equ 0 (echo ✅ SUCCESS) else (echo ❌ FAILED)
echo.

echo [5/6] ⏰ Extended Training (7 epochs)
echo Starting: %time%
python phase2_siamese.py --subset_size 500 --epochs 7 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --patience 5 > quick_results/quick_extended_%timestamp%.log 2>&1
echo Completed: %time%
if %errorlevel% equ 0 (echo ✅ SUCCESS) else (echo ❌ FAILED)
echo.

echo [6/6] 🎖️ Best Combined Settings
echo Starting: %time%
python phase2_siamese.py --subset_size 500 --epochs 7 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --learning_rate 0.002 --contrastive_weight 1.5 --regression_weight 0.3 --positive_threshold 0.8 --negative_threshold 0.2 --temperature 0.07 --patience 5 > quick_results/quick_best_combo_%timestamp%.log 2>&1
echo Completed: %time%
if %errorlevel% equ 0 (echo ✅ SUCCESS) else (echo ❌ FAILED)
echo.

REM ============================================
REM EXTRACT QUICK RESULTS
REM ============================================
echo 📊 Extracting quick experiment results...

python -c "
import os
import re
import json
from datetime import datetime

results = []
log_dir = 'quick_results'

# Define experiment names
exp_names = {
    'quick_lr_opt': 'Optimized Learning Rate (0.002)',
    'quick_loss_opt': 'Optimized Loss Weights (1.5/0.3)',
    'quick_thresh': 'Stricter Thresholds (0.8/0.2)',
    'quick_temp': 'Lower Temperature (0.07)',
    'quick_extended': 'Extended Training (7 epochs)',
    'quick_best_combo': 'Best Combined Settings'
}

for filename in os.listdir(log_dir):
    if '%timestamp%' in filename and filename.endswith('.log'):
        # Extract experiment type
        exp_key = filename.replace('_%timestamp%.log', '')
        exp_name = exp_names.get(exp_key, exp_key)
        
        filepath = os.path.join(log_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract correlation
            correlation = None
            if 'Best validation correlation:' in content:
                match = re.search(r'Best validation correlation:\s*([\d.]+)', content)
                if match:
                    correlation = float(match.group(1))
            
            # Extract training time
            training_time = None
            if 'training completed in' in content.lower():
                match = re.search(r'training completed in ([\d.]+)s', content, re.IGNORECASE)
                if match:
                    training_time = float(match.group(1))
            
            # Check if successful
            success = 'Training completed successfully!' in content
            
            results.append({
                'experiment': exp_name,
                'correlation': correlation,
                'training_time': training_time,
                'success': success,
                'log_file': filename
            })
            
        except Exception as e:
            results.append({
                'experiment': exp_name,
                'correlation': None,
                'training_time': None,
                'success': False,
                'error': str(e),
                'log_file': filename
            })

# Sort by correlation
successful = [r for r in results if r['correlation'] is not None]
successful.sort(key=lambda x: x['correlation'], reverse=True)

# Calculate improvement over baseline (assuming your baseline was 0.8185)
baseline = 0.8185
for r in successful:
    if r['correlation']:
        improvement = ((r['correlation'] - baseline) / baseline) * 100
        r['improvement_percent'] = round(improvement, 2)

# Save results
results_summary = {
    'timestamp': '%timestamp%',
    'baseline_correlation': baseline,
    'total_experiments': len(results),
    'successful_experiments': len(successful),
    'results': results,
    'ranked_results': successful
}

with open('quick_results/quick_summary_%timestamp%.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# Print summary
print('\\n🏆 QUICK EXPERIMENT SUMMARY:')
print(f'Baseline: {baseline:.4f}')
print(f'Total experiments: {len(results)}')
print(f'Successful: {len(successful)}')

if successful:
    print('\\n📈 RESULTS RANKED BY PERFORMANCE:')
    for i, r in enumerate(successful, 1):
        improvement = r.get('improvement_percent', 0)
        time_str = f\" ({r['training_time']:.0f}s)\" if r['training_time'] else \"\"
        if improvement > 0:
            print(f'{i}. {r[\"experiment\"]}: {r[\"correlation\"]:.4f} (+{improvement:.1f}%){time_str}')
        else:
            print(f'{i}. {r[\"experiment\"]}: {r[\"correlation\"]:.4f} ({improvement:.1f}%){time_str}')
    
    best = successful[0]
    best_improvement = best.get('improvement_percent', 0)
    print(f'\\n🥇 BEST RESULT: {best[\"experiment\"]}')
    print(f'   Correlation: {best[\"correlation\"]:.4f}')
    if best_improvement > 0:
        print(f'   Improvement: +{best_improvement:.1f}% over baseline')
    else:
        print(f'   Change: {best_improvement:.1f}% from baseline')
else:
    print('\\n❌ No successful experiments found.')

print(f'\\n📄 Detailed results saved to: quick_results/quick_summary_%timestamp%.json')
"

echo.
echo ✅ Quick experiments completed!
echo 📁 Check quick_results/ folder for detailed logs
echo 📊 Summary saved to quick_results/quick_summary_%timestamp%.json
echo.

REM Show completion time
echo 🏁 All experiments finished at: %time%
echo.
pause
