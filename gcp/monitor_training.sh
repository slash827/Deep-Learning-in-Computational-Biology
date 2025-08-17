#!/bin/bash
# GCP Training Monitor Script
# This script helps monitor the training progress on your GCP VM

echo "🔍 RNA-Protein Binding Training Monitor"
echo "======================================"

# Check if tmux session exists
if tmux has-session -t rna_training 2>/dev/null; then
    echo "✅ Training session 'rna_training' is active"
else
    echo "❌ Training session 'rna_training' not found"
    echo "💡 Start training with: ./gcp/run_training.sh"
    exit 1
fi

echo ""
echo "📊 Current System Status:"
echo "------------------------"

# GPU Status
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo ""

# Memory Status
echo "💾 System Memory:"
free -h | grep -E "(Mem|Swap)"
echo ""

# Disk Space
echo "💿 Disk Usage:"
df -h / | tail -1
echo ""

# Latest training run
echo "📁 Latest Training Run:"
LATEST_RUN=$(ls -td runs/phase2_fast_* 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
    echo "   Directory: $LATEST_RUN"
    if [ -f "$LATEST_RUN/training_summary.json" ]; then
        echo "   Status: Training completed ✅"
        # Show key results
        python3 -c "
import json
try:
    with open('$LATEST_RUN/training_summary.json', 'r') as f:
        data = json.load(f)
    print(f'   Best validation correlation: {data.get(\"best_val_correlation\", \"N/A\"):.4f}')
    print(f'   Best epoch: {data.get(\"best_epoch\", \"N/A\")}')
    print(f'   Total epochs: {data.get(\"total_epochs\", \"N/A\")}')
except:
    print('   Could not read training summary')
"
    else
        echo "   Status: Training in progress ⏳"
        # Check for config file to see progress
        if [ -f "$LATEST_RUN/config.json" ]; then
            echo "   Configuration saved ✅"
        fi
        # Check for plots
        if [ -d "$LATEST_RUN/plots" ]; then
            PLOT_COUNT=$(ls "$LATEST_RUN/plots"/*.png 2>/dev/null | wc -l)
            echo "   Generated plots: $PLOT_COUNT"
        fi
    fi
else
    echo "   No training runs found"
fi
echo ""

# Training session status
echo "🔄 Training Session Status:"
echo "   Session: rna_training"
tmux list-sessions -f "#{session_name}: #{session_activity}" | grep rna_training
echo ""

echo "🛠️ Available Commands:"
echo "   View training:     tmux attach -t rna_training"
echo "   Monitor GPU:       watch -n 1 nvidia-smi"
echo "   Check progress:    ./gcp/monitor_training.sh"
echo "   View logs:         ls -la runs/"
echo ""

echo "💡 To detach from training view: Ctrl+B, then D"
echo "🛑 To stop training: tmux kill-session -t rna_training"

