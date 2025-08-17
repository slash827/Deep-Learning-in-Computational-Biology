@echo off
echo ============================================================
echo 🧬 PROTEINBERT EMBEDDING EXPERIMENTS
echo ============================================================
echo.

REM Create results directory
if not exist "embedding_experiments" mkdir embedding_experiments
if not exist "runs\emb_cache" mkdir runs\emb_cache

REM Set timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"

echo 🧬 Testing different ProteinBERT embedding strategies...
echo Results will be saved with timestamp: %timestamp%
echo.

REM ============================================
REM POOLING STRATEGY EXPERIMENTS
REM ============================================
echo 🏊 Creating embeddings with different pooling strategies...

echo [1/8] Mean Pooling (baseline)
python scripts/cache_proteinbert_embeddings.py --data_dir src/data --out runs/emb_cache/protein_bert_mean_%timestamp%.pt --pooling mean --max_length 1024 > embedding_experiments/mean_pooling_%timestamp%.log 2>&1
if %errorlevel% equ 0 (
    echo ✅ Mean pooling embeddings created
    echo [1/8] Testing mean pooling with Siamese model...
    python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --protein_embedding_path runs/emb_cache/protein_bert_mean_%timestamp%.pt > embedding_experiments/test_mean_%timestamp%.log 2>&1
    if %errorlevel% equ 0 echo ✅ Mean pooling test completed
) else (
    echo ❌ FAILED: Mean pooling creation
)

echo [2/8] CLS Token Pooling
python scripts/cache_proteinbert_embeddings.py --data_dir src/data --out runs/emb_cache/protein_bert_cls_%timestamp%.pt --pooling cls --max_length 1024 > embedding_experiments/cls_pooling_%timestamp%.log 2>&1
if %errorlevel% equ 0 (
    echo ✅ CLS pooling embeddings created
    echo [2/8] Testing CLS pooling with Siamese model...
    python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --protein_embedding_path runs/emb_cache/protein_bert_cls_%timestamp%.pt > embedding_experiments/test_cls_%timestamp%.log 2>&1
    if %errorlevel% equ 0 echo ✅ CLS pooling test completed
) else (
    echo ❌ FAILED: CLS pooling creation
)

REM ============================================
REM SEQUENCE LENGTH EXPERIMENTS
REM ============================================
echo.
echo 📏 Testing different maximum sequence lengths...

echo [3/8] Max Length 512
python scripts/cache_proteinbert_embeddings.py --data_dir src/data --out runs/emb_cache/protein_bert_512_%timestamp%.pt --pooling mean --max_length 512 > embedding_experiments/maxlen_512_%timestamp%.log 2>&1
if %errorlevel% equ 0 (
    echo ✅ MaxLen 512 embeddings created
    echo [3/8] Testing MaxLen 512 with Siamese model...
    python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --protein_embedding_path runs/emb_cache/protein_bert_512_%timestamp%.pt > embedding_experiments/test_512_%timestamp%.log 2>&1
    if %errorlevel% equ 0 echo ✅ MaxLen 512 test completed
) else (
    echo ❌ FAILED: MaxLen 512 creation
)

echo [4/8] Max Length 768
python scripts/cache_proteinbert_embeddings.py --data_dir src/data --out runs/emb_cache/protein_bert_768_%timestamp%.pt --pooling mean --max_length 768 > embedding_experiments/maxlen_768_%timestamp%.log 2>&1
if %errorlevel% equ 0 (
    echo ✅ MaxLen 768 embeddings created
    echo [4/8] Testing MaxLen 768 with Siamese model...
    python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --protein_embedding_path runs/emb_cache/protein_bert_768_%timestamp%.pt > embedding_experiments/test_768_%timestamp%.log 2>&1
    if %errorlevel% equ 0 echo ✅ MaxLen 768 test completed
) else (
    echo ❌ FAILED: MaxLen 768 creation
)

REM ============================================
REM BATCH SIZE EXPERIMENTS
REM ============================================
echo.
echo 📦 Testing different batch sizes for embedding creation...

echo [5/8] Batch Size 16
python scripts/cache_proteinbert_embeddings.py --data_dir src/data --out runs/emb_cache/protein_bert_batch16_%timestamp%.pt --pooling mean --max_length 1024 --batch_size 16 > embedding_experiments/batch_16_%timestamp%.log 2>&1
if %errorlevel% equ 0 (
    echo ✅ Batch 16 embeddings created
    echo [5/8] Testing Batch 16 with Siamese model...
    python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --protein_embedding_path runs/emb_cache/protein_bert_batch16_%timestamp%.pt > embedding_experiments/test_batch16_%timestamp%.log 2>&1
    if %errorlevel% equ 0 echo ✅ Batch 16 test completed
) else (
    echo ❌ FAILED: Batch 16 creation
)

echo [6/8] Batch Size 64
python scripts/cache_proteinbert_embeddings.py --data_dir src/data --out runs/emb_cache/protein_bert_batch64_%timestamp%.pt --pooling mean --max_length 1024 --batch_size 64 > embedding_experiments/batch_64_%timestamp%.log 2>&1
if %errorlevel% equ 0 (
    echo ✅ Batch 64 embeddings created
    echo [6/8] Testing Batch 64 with Siamese model...
    python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --protein_embedding_path runs/emb_cache/protein_bert_batch64_%timestamp%.pt > embedding_experiments/test_batch64_%timestamp%.log 2>&1
    if %errorlevel% equ 0 echo ✅ Batch 64 test completed
) else (
    echo ❌ FAILED: Batch 64 creation
)

REM ============================================
REM ALTERNATIVE MODEL EXPERIMENTS
REM ============================================
echo.
echo 🤖 Testing alternative ProteinBERT models...

echo [7/8] ProtBERT (alternative model)
python scripts/cache_proteinbert_embeddings.py --data_dir src/data --out runs/emb_cache/protein_bert_alt_%timestamp%.pt --model Rostlab/prot_bert --pooling mean --max_length 1024 > embedding_experiments/alt_model_%timestamp%.log 2>&1
if %errorlevel% equ 0 (
    echo ✅ Alternative model embeddings created
    echo [7/8] Testing alternative model with Siamese...
    python phase2_siamese.py --subset_size 500 --epochs 5 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --protein_embedding_path runs/emb_cache/protein_bert_alt_%timestamp%.pt > embedding_experiments/test_alt_%timestamp%.log 2>&1
    if %errorlevel% equ 0 echo ✅ Alternative model test completed
) else (
    echo ❌ FAILED: Alternative model creation
)

REM ============================================
REM COMBINED BEST EXPERIMENT
REM ============================================
echo.
echo 🎯 Creating optimized embedding (best settings combination)...

echo [8/8] Optimized: CLS pooling + 768 length + batch 32
python scripts/cache_proteinbert_embeddings.py --data_dir src/data --out runs/emb_cache/protein_bert_optimized_%timestamp%.pt --pooling cls --max_length 768 --batch_size 32 > embedding_experiments/optimized_%timestamp%.log 2>&1
if %errorlevel% equ 0 (
    echo ✅ Optimized embeddings created
    echo [8/8] Testing optimized embeddings with best hyperparams...
    python phase2_siamese.py --subset_size 500 --epochs 7 --loss_type hybrid --data_dir src/data/ --pair_sampling_ratio 0.5 --protein_embedding_path runs/emb_cache/protein_bert_optimized_%timestamp%.pt --learning_rate 0.002 --contrastive_weight 1.5 --regression_weight 0.3 --patience 5 > embedding_experiments/test_optimized_%timestamp%.log 2>&1
    if %errorlevel% equ 0 echo ✅ Optimized test completed
) else (
    echo ❌ FAILED: Optimized creation
)

REM ============================================
REM EXTRACT EMBEDDING EXPERIMENT RESULTS
REM ============================================
echo.
echo 📊 Extracting embedding experiment results...

python -c "
import os
import re
import json
from datetime import datetime

results = []
log_dir = 'embedding_experiments'

# Map test log files to embedding configs
test_configs = {
    'test_mean': 'Mean Pooling',
    'test_cls': 'CLS Pooling', 
    'test_512': 'MaxLen 512',
    'test_768': 'MaxLen 768',
    'test_batch16': 'Batch Size 16',
    'test_batch64': 'Batch Size 64',
    'test_alt': 'Alternative Model',
    'test_optimized': 'Optimized Config'
}

for filename in os.listdir(log_dir):
    if 'test_' in filename and '%timestamp%' in filename and filename.endswith('.log'):
        # Extract experiment type
        for prefix, config_name in test_configs.items():
            if filename.startswith(prefix):
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
                        'config': config_name,
                        'correlation': correlation,
                        'success': success,
                        'log_file': filename
                    })
                    
                except Exception as e:
                    results.append({
                        'config': config_name,
                        'correlation': None,
                        'success': False,
                        'error': str(e),
                        'log_file': filename
                    })
                break

# Sort by correlation
successful = [r for r in results if r['correlation'] is not None]
successful.sort(key=lambda x: x['correlation'], reverse=True)

# Save results
with open('embedding_experiments/embedding_summary_%timestamp%.json', 'w') as f:
    json.dump({
        'timestamp': '%timestamp%',
        'total_configurations': len(results),
        'successful_configurations': len(successful),
        'results': results,
        'best_embeddings': successful[:3]
    }, f, indent=2)

# Print summary
print('\\n🏆 EMBEDDING EXPERIMENT SUMMARY:')
print(f'Total configurations tested: {len(results)}')
print(f'Successful: {len(successful)}')

if successful:
    print('\\n📈 TOP 3 EMBEDDING CONFIGS:')
    for i, r in enumerate(successful[:3], 1):
        print(f'{i}. {r[\"config\"]}: {r[\"correlation\"]:.4f}')
    
    print(f'\\n🥇 BEST EMBEDDING: {successful[0][\"config\"]} = {successful[0][\"correlation\"]:.4f}')
else:
    print('\\n❌ No successful embedding configurations found.')

print(f'\\n📄 Detailed results saved to: embedding_experiments/embedding_summary_%timestamp%.json')
"

echo.
echo ✅ ProteinBERT embedding experiments completed!
echo 📁 Check embedding_experiments/ folder for detailed logs
echo 📊 Summary saved to embedding_experiments/embedding_summary_%timestamp%.json
echo.
pause
