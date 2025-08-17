# Running RNA-Protein Binding Prediction on Google Cloud Platform

This guide helps you run the optimized Phase 2 training configuration that achieved **0.7733 validation correlation** on Google Cloud Platform.

## 🎯 Proven Configuration

The configuration in this setup replicates the exact parameters that achieved excellent results:
- **Validation Correlation**: 0.7733
- **Training Time**: ~75 minutes (561.3s per epoch × 8 epochs)
- **Model Size**: 1,559,041 parameters
- **Expected Speedup**: 3-4x faster than original model

## 🚀 Quick Start

### 1. Set up GCP Project
```bash
# Set your project ID
export GCP_PROJECT_ID="your-project-id"

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

### 2. Deploy VM
```bash
# Make scripts executable
chmod +x gcp/*.sh

# Deploy VM with GPU (modify variables as needed)
cd gcp
./deploy_vm.sh
```

### 3. Upload Your Code
```bash
# Upload project to VM
gcloud compute scp --recurse .. rna-protein-dl-vm:~/Deep-Learning-in-Computational-Biology --zone=us-central1-a

# SSH into VM
gcloud compute ssh rna-protein-dl-vm --zone=us-central1-a
```

### 4. Run Training
```bash
# On the VM
cd ~/Deep-Learning-in-Computational-Biology
./gcp/run_training.sh
```

## 📊 VM Specifications

### Default Configuration
- **Machine Type**: n1-standard-4 (4 vCPUs, 15 GB RAM)
- **GPU**: NVIDIA Tesla T4 (16 GB VRAM)
- **Boot Disk**: 50 GB SSD
- **Image**: PyTorch Deep Learning VM (GPU optimized)
- **Pricing**: Preemptible (up to 80% savings)

### Alternative Configurations

#### High-End Training (Faster)
```bash
export MACHINE_TYPE="n1-standard-8"
export GPU_TYPE="nvidia-tesla-v100"
export BOOT_DISK_SIZE="100GB"
./deploy_vm.sh
```

#### Budget Training (Slower)
```bash
export MACHINE_TYPE="n1-standard-2"
export GPU_TYPE="nvidia-tesla-t4"
export BOOT_DISK_SIZE="30GB"
./deploy_vm.sh
```

## 🔧 Training Parameters

The exact configuration that achieved 0.7733 validation correlation:

```bash
python phase2_fast.py \
    --data_dir "src/data" \
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
```

## 💰 Cost Optimization

### Estimated Costs (US Central)
- **T4 Preemptible**: ~$0.35/hour
- **V100 Preemptible**: ~$0.74/hour
- **Storage**: ~$0.04/GB/month

### Cost-Saving Tips
1. **Use Preemptible Instances**: Up to 80% savings
2. **Stop VM When Not Training**: `gcloud compute instances stop VM_NAME`
3. **Use Smaller Disks**: Only 30-50GB needed
4. **Monitor Usage**: Set up billing alerts
5. **Delete VM When Done**: Don't forget!

## 🔍 Monitoring Training

### During Training
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# View training progress
tmux attach -t rna_training

# Check training outputs
ls -la runs/
```

### Training Progress
- **Epoch 1-2**: Warmup phase
- **Epoch 3-5**: Rapid improvement
- **Epoch 6-8**: Fine-tuning and convergence
- **Expected**: ~0.77 validation correlation

## 📂 Output Files

Training results are saved in `runs/phase2_fast_TIMESTAMP/`:
```
runs/phase2_fast_20250815_194926/
├── config.json                    # Training configuration
├── training_summary.json          # Performance metrics
├── plots/
│   ├── training_history.png      # Loss/correlation curves
│   └── predictions_vs_targets.png # Validation predictions
└── models/
    └── phase2_fast_model.pth      # Trained model
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   nvidia-smi  # Should show GPU info
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Out of Memory**
   - Reduce `--batch_size` to 16 or 12
   - Reduce `--max_protein_length` to 300

3. **Slow Training**
   - Ensure GPU is being used: `watch -n 1 nvidia-smi`
   - Check VM has correct GPU type

4. **Missing Embeddings**
   - Upload `runs/emb_cache/protein_bert.pt` file
   - Or remove `--protein_encoder protbert_cached` to use LSTM

### VM Management
```bash
# Stop VM (saves costs)
gcloud compute instances stop rna-protein-dl-vm --zone=us-central1-a

# Start VM
gcloud compute instances start rna-protein-dl-vm --zone=us-central1-a

# Delete VM (when completely done)
gcloud compute instances delete rna-protein-dl-vm --zone=us-central1-a
```

## 📈 Expected Results

Based on the proven configuration:
- **Best Validation Correlation**: ~0.7733
- **Training Time**: 75 minutes
- **Model Parameters**: 1,559,041
- **Training Stability**: Excellent (early stopping around epoch 6-8)

## 🎉 Success Indicators

✅ **Excellent Performance** (≥0.75 correlation)
- Configuration is optimal
- Use these exact parameters

✅ **Good Performance** (0.65-0.75 correlation)  
- Minor tweaks may help
- Consider increasing hidden size

⚠️ **Lower Performance** (<0.65 correlation)
- Check data loading
- Verify GPU usage
- Consider data preprocessing

## 📞 Support

If you encounter issues:
1. Check the startup script logs: `cat /var/log/startup-script.log`
2. Verify all files uploaded correctly
3. Ensure protein embeddings are available
4. Monitor GPU usage during training

Remember to **stop or delete your VM** when training is complete to avoid unnecessary charges!

