@echo off
echo ============================================================
echo 🚀 SIAMESE MODEL HYPERPARAMETER EXPERIMENTS
echo ============================================================
echo.

REM Create results directory
if not exist "experiment_results" mkdir experiment_results

REM Set timestamp for results
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"

echo 📊 Starting hyperparameter optimization experiments...
echo Results will be saved with timestamp: %timestamp%
echo.

REM ============================================
REM LEARNING RATE EXPERIMENTS
REM ============================================
echo 🔧 Testing Learning Rates...

echo [1/20] Learning Rate 0.001 (baseline)
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --learning_rate 0.001 > experiment_results/lr_0001_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: lr_0001

echo [2/20] Learning Rate 0.002
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --learning_rate 0.002 > experiment_results/lr_0002_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: lr_0002

echo [3/20] Learning Rate 0.003
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --learning_rate 0.003 > experiment_results/lr_0003_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: lr_0003

echo [4/20] Learning Rate 0.005
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --learning_rate 0.005 > experiment_results/lr_0005_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: lr_0005

REM ============================================
REM LOSS WEIGHT EXPERIMENTS
REM ============================================
echo.
echo ⚖️ Testing Loss Weight Combinations...

echo [5/20] Loss Weights: Contrastive=1.5, Regression=0.3
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --contrastive_weight 1.5 --regression_weight 0.3 > experiment_results/loss_1.5_0.3_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: loss_1.5_0.3

echo [6/20] Loss Weights: Contrastive=2.0, Regression=0.2
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --contrastive_weight 2.0 --regression_weight 0.2 > experiment_results/loss_2.0_0.2_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: loss_2.0_0.2

echo [7/20] Loss Weights: Contrastive=0.8, Regression=0.7
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --contrastive_weight 0.8 --regression_weight 0.7 > experiment_results/loss_0.8_0.7_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: loss_0.8_0.7

REM ============================================
REM PAIR SAMPLING EXPERIMENTS
REM ============================================
echo.
echo 👥 Testing Pair Sampling Ratios...

echo [8/20] Pair Sampling Ratio: 0.3
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.3 > experiment_results/pairs_0.3_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: pairs_0.3

echo [9/20] Pair Sampling Ratio: 0.7
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.7 > experiment_results/pairs_0.7_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: pairs_0.7

echo [10/20] Pair Sampling Ratio: 1.0
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 1.0 > experiment_results/pairs_1.0_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: pairs_1.0

REM ============================================
REM THRESHOLD EXPERIMENTS
REM ============================================
echo.
echo 🎯 Testing Similarity Thresholds...

echo [11/20] Thresholds: Positive=0.8, Negative=0.2
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --positive_threshold 0.8 --negative_threshold 0.2 > experiment_results/thresh_0.8_0.2_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: thresh_0.8_0.2

echo [12/20] Thresholds: Positive=0.6, Negative=0.4
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --positive_threshold 0.6 --negative_threshold 0.4 > experiment_results/thresh_0.6_0.4_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: thresh_0.6_0.4

REM ============================================
REM ARCHITECTURE EXPERIMENTS
REM ============================================
echo.
echo 🏗️ Testing Architecture Variations...

echo [13/20] RNA Hidden Size: 96
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --rna_hidden_size 96 > experiment_results/hidden_96_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: hidden_96

echo [14/20] RNA Hidden Size: 160
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --rna_hidden_size 160 > experiment_results/hidden_160_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: hidden_160

echo [15/20] Embedding Dimension: 128
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --embedding_dim 128 > experiment_results/emb_128_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: emb_128

echo [16/20] Embedding Dimension: 384
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --embedding_dim 384 > experiment_results/emb_384_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: emb_384

REM ============================================
REM TEMPERATURE EXPERIMENTS
REM ============================================
echo.
echo 🌡️ Testing Temperature Values...

echo [17/20] Temperature: 0.05
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --temperature 0.05 > experiment_results/temp_0.05_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: temp_0.05

echo [18/20] Temperature: 0.2
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --temperature 0.2 > experiment_results/temp_0.2_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: temp_0.2

REM ============================================
REM EXTENDED TRAINING
REM ============================================
echo.
echo ⏰ Testing Extended Training...

echo [19/20] Extended Training: 7 epochs
python phase2_siamese.py --subset_size 500 --epochs 7 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --patience 5 > experiment_results/epochs_7_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: epochs_7

echo [20/20] Pure Contrastive Learning
python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type contrastive --data_dir src/data/ --pair_sampling_ratio 0.5 > experiment_results/pure_contrastive_%timestamp%.log 2>&1
if %errorlevel% neq 0 echo ❌ FAILED: pure_contrastive

REM ============================================
REM EXTRACT RESULTS
REM ============================================
echo.
echo 📊 Extracting results...

python -c "
import os
import re
import json
from datetime import datetime

results = []
log_dir = 'experiment_results'

for filename in os.listdir(log_dir):
    if '%timestamp%' in filename and filename.endswith('.log'):
        exp_name = filename.replace('_%timestamp%.log', '')
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
            
            # Check if successful
            success = 'Training completed successfully!' in content
            
            results.append({
                'experiment': exp_name,
                'correlation': correlation,
                'success': success,
                'log_file': filename
            })
            
        except Exception as e:
            results.append({
                'experiment': exp_name,
                'correlation': None,
                'success': False,
                'error': str(e),
                'log_file': filename
            })

# Sort by correlation
successful = [r for r in results if r['correlation'] is not None]
successful.sort(key=lambda x: x['correlation'], reverse=True)

# Save results
with open('experiment_results/summary_%timestamp%.json', 'w') as f:
    json.dump({
        'timestamp': '%timestamp%',
        'total_experiments': len(results),
        'successful_experiments': len(successful),
        'results': results,
        'best_results': successful[:5]
    }, f, indent=2)

# Print summary
print('\\n🏆 EXPERIMENT SUMMARY:')
print(f'Total experiments: {len(results)}')
print(f'Successful: {len(successful)}')

if successful:
    print('\\n📈 TOP 5 RESULTS:')
    for i, r in enumerate(successful[:5], 1):
        print(f'{i}. {r[\"experiment\"]}: {r[\"correlation\"]:.4f}')
    
    print(f'\\n🥇 BEST: {successful[0][\"experiment\"]} = {successful[0][\"correlation\"]:.4f}')
else:
    print('\\n❌ No successful experiments found.')

print(f'\\n📄 Detailed results saved to: experiment_results/summary_%timestamp%.json')
"

echo.
echo ✅ Hyperparameter experiments completed!
echo 📁 Check experiment_results/ folder for detailed logs
echo 📊 Summary saved to experiment_results/summary_%timestamp%.json
echo.
pause
