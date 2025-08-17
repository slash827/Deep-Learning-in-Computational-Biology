@echo off
echo ============================================================
echo 🚀 COMPREHENSIVE SIAMESE MODEL EXPERIMENT SUITE
echo ============================================================
echo.

echo This will run:
echo 1. 20 Hyperparameter experiments (~17 hours)
echo 2. 8 ProteinBERT embedding experiments (~8 hours)
echo 3. Analysis and comparison of all results
echo.
echo Total estimated time: ~25 hours
echo.
set /p choice="Do you want to continue? (y/n): "
if /i not "%choice%"=="y" goto :end

REM Set master timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "master_timestamp=%dt:~0,8%_%dt:~8,6%"

echo.
echo 📅 Master experiment session: %master_timestamp%
echo ⏰ Started at: %time%
echo.

REM Create master results directory
if not exist "master_experiments_%master_timestamp%" mkdir master_experiments_%master_timestamp%

echo ============================================================ > master_experiments_%master_timestamp%/experiment_log.txt
echo COMPREHENSIVE EXPERIMENT LOG >> master_experiments_%master_timestamp%/experiment_log.txt
echo Started: %date% %time% >> master_experiments_%master_timestamp%/experiment_log.txt
echo ============================================================ >> master_experiments_%master_timestamp%/experiment_log.txt
echo. >> master_experiments_%master_timestamp%/experiment_log.txt

REM ============================================
REM PHASE 1: HYPERPARAMETER EXPERIMENTS
REM ============================================
echo 📊 PHASE 1: HYPERPARAMETER OPTIMIZATION
echo ============================================
echo Phase 1 started: %time% >> master_experiments_%master_timestamp%/experiment_log.txt

call run_hyperparameter_experiments.bat

echo Phase 1 completed: %time% >> master_experiments_%master_timestamp%/experiment_log.txt

REM Move hyperparameter results to master folder
if exist "experiment_results" (
    xcopy "experiment_results\*" "master_experiments_%master_timestamp%\hyperparameter_results\" /E /I /Y
    echo Hyperparameter results moved to master folder
)

REM ============================================
REM PHASE 2: PROTEINBERT EXPERIMENTS
REM ============================================
echo.
echo 🧬 PHASE 2: PROTEINBERT OPTIMIZATION
echo ==========================================
echo Phase 2 started: %time% >> master_experiments_%master_timestamp%/experiment_log.txt

call run_proteinbert_experiments.bat

echo Phase 2 completed: %time% >> master_experiments_%master_timestamp%/experiment_log.txt

REM Move ProteinBERT results to master folder
if exist "embedding_experiments" (
    xcopy "embedding_experiments\*" "master_experiments_%master_timestamp%\embedding_results\" /E /I /Y
    echo ProteinBERT results moved to master folder
)

REM ============================================
REM PHASE 3: COMPREHENSIVE ANALYSIS
REM ============================================
echo.
echo 📈 PHASE 3: COMPREHENSIVE ANALYSIS
echo ===================================
echo Phase 3 started: %time% >> master_experiments_%master_timestamp%/experiment_log.txt

python -c "
import os
import re
import json
from datetime import datetime
import glob

def extract_correlation_from_log(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        correlation = None
        if 'Best validation correlation:' in content:
            match = re.search(r'Best validation correlation:\s*([\d.]+)', content)
            if match:
                correlation = float(match.group(1))
        
        success = 'Training completed successfully!' in content
        return correlation, success
    except:
        return None, False

# Collect all results
all_results = {
    'master_timestamp': '%master_timestamp%',
    'analysis_time': datetime.now().isoformat(),
    'baseline_correlation': 0.8185,
    'hyperparameter_experiments': [],
    'embedding_experiments': [],
    'summary': {}
}

# Process hyperparameter results
hp_dir = 'master_experiments_%master_timestamp%/hyperparameter_results'
if os.path.exists(hp_dir):
    for log_file in os.listdir(hp_dir):
        if log_file.endswith('.log') and 'test_' not in log_file:
            correlation, success = extract_correlation_from_log(os.path.join(hp_dir, log_file))
            
            # Extract experiment type from filename
            exp_name = log_file.replace('.log', '').split('_')[0:-1]
            exp_name = '_'.join(exp_name)
            
            all_results['hyperparameter_experiments'].append({
                'experiment': exp_name,
                'correlation': correlation,
                'success': success,
                'log_file': log_file
            })

# Process embedding results  
emb_dir = 'master_experiments_%master_timestamp%/embedding_results'
if os.path.exists(emb_dir):
    for log_file in os.listdir(emb_dir):
        if log_file.startswith('test_') and log_file.endswith('.log'):
            correlation, success = extract_correlation_from_log(os.path.join(emb_dir, log_file))
            
            # Extract embedding type
            if 'test_mean' in log_file:
                emb_type = 'Mean Pooling'
            elif 'test_cls' in log_file:
                emb_type = 'CLS Pooling'
            elif 'test_512' in log_file:
                emb_type = 'MaxLen 512'
            elif 'test_768' in log_file:
                emb_type = 'MaxLen 768'
            elif 'test_batch16' in log_file:
                emb_type = 'Batch 16'
            elif 'test_batch64' in log_file:
                emb_type = 'Batch 64'
            elif 'test_alt' in log_file:
                emb_type = 'Alternative Model'
            elif 'test_optimized' in log_file:
                emb_type = 'Optimized Config'
            else:
                emb_type = 'Unknown'
            
            all_results['embedding_experiments'].append({
                'experiment': emb_type,
                'correlation': correlation,
                'success': success,
                'log_file': log_file
            })

# Find best results
hp_successful = [r for r in all_results['hyperparameter_experiments'] if r['correlation'] is not None]
emb_successful = [r for r in all_results['embedding_experiments'] if r['correlation'] is not None]

hp_successful.sort(key=lambda x: x['correlation'], reverse=True)
emb_successful.sort(key=lambda x: x['correlation'], reverse=True)

# Calculate improvements
baseline = 0.8185
for results_list in [hp_successful, emb_successful]:
    for r in results_list:
        if r['correlation']:
            improvement = ((r['correlation'] - baseline) / baseline) * 100
            r['improvement_percent'] = round(improvement, 2)

# Summary statistics
all_results['summary'] = {
    'total_experiments': len(all_results['hyperparameter_experiments']) + len(all_results['embedding_experiments']),
    'successful_experiments': len(hp_successful) + len(emb_successful),
    'hyperparameter_success_rate': len(hp_successful) / max(1, len(all_results['hyperparameter_experiments'])),
    'embedding_success_rate': len(emb_successful) / max(1, len(all_results['embedding_experiments'])),
    'best_hyperparameter': hp_successful[0] if hp_successful else None,
    'best_embedding': emb_successful[0] if emb_successful else None
}

# Find overall best
all_successful = hp_successful + emb_successful
if all_successful:
    overall_best = max(all_successful, key=lambda x: x['correlation'])
    all_results['summary']['overall_best'] = overall_best

# Save comprehensive results
with open('master_experiments_%master_timestamp%/comprehensive_analysis.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Generate report
report = []
report.append('🏆 COMPREHENSIVE EXPERIMENT RESULTS')
report.append('=' * 50)
report.append(f'Baseline Correlation: {baseline:.4f}')
report.append(f'Total Experiments: {all_results[\"summary\"][\"total_experiments\"]}')
report.append(f'Successful: {all_results[\"summary\"][\"successful_experiments\"]}')
report.append('')

if hp_successful:
    report.append('📊 TOP 5 HYPERPARAMETER RESULTS:')
    for i, r in enumerate(hp_successful[:5], 1):
        improvement = r.get('improvement_percent', 0)
        report.append(f'{i}. {r[\"experiment\"]}: {r[\"correlation\"]:.4f} ({improvement:+.1f}%)')
    report.append('')

if emb_successful:
    report.append('🧬 TOP 3 EMBEDDING RESULTS:')
    for i, r in enumerate(emb_successful[:3], 1):
        improvement = r.get('improvement_percent', 0)
        report.append(f'{i}. {r[\"experiment\"]}: {r[\"correlation\"]:.4f} ({improvement:+.1f}%)')
    report.append('')

if all_successful:
    best = max(all_successful, key=lambda x: x['correlation'])
    improvement = best.get('improvement_percent', 0)
    report.append('🥇 OVERALL BEST RESULT:')
    report.append(f'   {best[\"experiment\"]}: {best[\"correlation\"]:.4f} ({improvement:+.1f}%)')
    
    if improvement > 0:
        report.append(f'   🎉 IMPROVEMENT ACHIEVED: +{improvement:.1f}% over baseline!')
    else:
        report.append(f'   📊 Result: {improvement:.1f}% change from baseline')

# Print and save report
print('\\n'.join(report))

with open('master_experiments_%master_timestamp%/experiment_report.txt', 'w') as f:
    f.write('\\n'.join(report))

print(f'\\n📄 Comprehensive analysis saved to:')
print(f'   - master_experiments_%master_timestamp%/comprehensive_analysis.json')
print(f'   - master_experiments_%master_timestamp%/experiment_report.txt')
"

echo Phase 3 completed: %time% >> master_experiments_%master_timestamp%/experiment_log.txt

REM ============================================
REM COMPLETION
REM ============================================
echo.
echo ============================================================
echo 🎉 ALL EXPERIMENTS COMPLETED!
echo ============================================================
echo.
echo 📅 Session: %master_timestamp%
echo ⏰ Finished at: %time%
echo 📁 Results location: master_experiments_%master_timestamp%/
echo.

echo Experiment session completed: %date% %time% >> master_experiments_%master_timestamp%/experiment_log.txt

echo 📊 You can find:
echo    - Comprehensive analysis: master_experiments_%master_timestamp%/comprehensive_analysis.json
echo    - Summary report: master_experiments_%master_timestamp%/experiment_report.txt
echo    - Hyperparameter logs: master_experiments_%master_timestamp%/hyperparameter_results/
echo    - Embedding logs: master_experiments_%master_timestamp%/embedding_results/
echo.

:end
pause
