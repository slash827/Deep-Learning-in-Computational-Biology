#!/bin/bash
# GCP Training Script for Optimized RNA-Protein Binding Prediction
# This script runs the exact configuration that achieved 0.7733 validation correlation

set -e

echo "🧬 Starting Optimized RNA-Protein Binding Training on GCP"
echo "=========================================================="
echo "⚡ Using the proven configuration with 0.7733 validation correlation"
echo "🚀 Expected training time: ~75 minutes (561.3s per epoch × 8 epochs)"
echo "=========================================================="

# Check if we're in the right directory
if [ ! -f "phase2_fast.py" ]; then
    echo "❌ Error: phase2_fast.py not found in current directory"
    echo "Please ensure you're in the project root directory"
    exit 1
fi

# Check if required data directory exists
if [ ! -d "src/data" ]; then
    echo "❌ Error: src/data directory not found"
    echo "Please ensure the project is properly uploaded"
    exit 1
fi

# Check if cached protein embeddings exist
PROTEIN_EMBEDDING_PATH="runs/emb_cache/protein_bert.pt"
if [ ! -f "$PROTEIN_EMBEDDING_PATH" ]; then
    echo "⚠️ Warning: Cached protein embeddings not found at $PROTEIN_EMBEDDING_PATH"
    echo "The script will run but may need to generate embeddings on-the-fly"
    echo "Consider uploading the cached embeddings for faster training"
fi

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Check GPU availability
echo "🖥️ Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️ No GPU detected - training will be much slower on CPU')
"

# Create tmux session for long-running training
echo "🔄 Starting training in tmux session 'rna_training'..."
echo "💡 You can detach with Ctrl+B, D and reattach with 'tmux attach -t rna_training'"

# The exact command that achieved 0.7733 validation correlation
tmux new-session -d -s rna_training "
python phase2_fast.py \
    --data_dir 'src/data' \
    --subset_size 2000 \
    --batch_size 24 \
    --epochs 8 \
    --learning_rate 0.0005 \
    --hidden_size 128 \
    --num_layers 2 \
    --dropout 0.25 \
    --attention_heads 8 \
    --attention_dropout 0.1 \
    --patience 10 \
    --min_delta 1e-5 \
    --max_grad_norm 0.5 \
    --lr_scheduler_patience 5 \
    --lr_scheduler_factor 0.7 \
    --max_protein_length 400 \
    --max_rna_length 80 \
    --warmup_epochs 2 \
    --simple_attention \
    --num_workers 0 \
    --protein_encoder protbert_cached \
    --protein_embedding_path runs/emb_cache/protein_bert.pt \
    --protein_embedding_dim 1024 \
    --device auto
"

echo ""
echo "🚀 Training started in tmux session!"
echo ""
echo "📋 Training Configuration:"
echo "   • Data subset: 2,000 samples"
echo "   • Batch size: 24"
echo "   • Epochs: 8"
echo "   • Learning rate: 0.0005"
echo "   • Hidden size: 128"
echo "   • LSTM layers: 2"
echo "   • Attention heads: 8"
echo "   • Max protein length: 400"
echo "   • Max RNA length: 80"
echo "   • Using cached ProteinBERT embeddings"
echo ""
echo "📊 Expected Results:"
echo "   • Validation correlation: ~0.77"
echo "   • Training time: ~75 minutes"
echo "   • Model parameters: ~1.5M"
echo ""
echo "🔍 Monitoring commands:"
echo "   • View training: tmux attach -t rna_training"
echo "   • Monitor GPU: watch -n 1 nvidia-smi"
echo "   • Check progress: ls -la runs/"
echo ""
echo "💡 Tips:"
echo "   • Press Ctrl+B, D to detach from tmux"
echo "   • Training will continue even if you disconnect"
echo "   • Results will be saved in runs/ directory"
echo "   • Stop VM when training is complete to save costs"

# Also show the tmux session status
sleep 2
echo ""
echo "📊 Current tmux sessions:"
tmux list-sessions

