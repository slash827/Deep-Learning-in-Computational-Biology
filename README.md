# Deep Learning for RNA-Protein Binding Prediction

🧬 **Advanced deep learning models for predicting RNA-protein binding affinities**

## 🎯 Project Highlights

- **Best Performance**: **79.74% correlation coefficient**
- **Architecture**: ProteinEmbeddingFusion with ProtBERT embeddings
- **Speed**: 10x training acceleration (20min → 2min per epoch)
- **Innovation**: Novel integration of protein language models

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run best performing model
python phase2_fast.py --data_dir src/dataset --protein_encoder protbert_cached --protein_embedding_path emb_cache/protein_bert.pt

# Analyze training results
python analyze_training.py --compare
```

## 📊 Key Results

| Phase | Architecture | Correlation | Improvement |
|-------|-------------|-------------|-------------|
| Phase 1 | Basic LSTM | 59.21% | Baseline |
| Phase 2A | BiLSTM + Attention | 65.37% | +6.16% |
| Phase 2B | + ProtBERT | 74.74% | +9.37% |
| Phase 2C | Optimized Single Layer | 76.23% | +1.49% |
| **Phase 2D** | **Two-Layer Fusion** | **79.74%** | **+3.51%** |

## 🏗️ Architecture

```
RNA Sequences → BiLSTM (2-layer) → Attention → 
                                              → Fusion → Binding Score
Proteins → ProtBERT → BiLSTM (2-layer) → Attention →
```

## 📁 Project Structure

```
Deep-Learning-in-Computational-Biology/
├── src/
│   ├── data/           # Data loading and processing modules
│   ├── dataset/        # Training and test data files ⭐ NEW LOCATION
│   ├── models/         # Model architectures  
│   ├── training/       # Training utilities
│   └── utils/          # Helper functions
├── experiments/        # All experimental scripts (organized by phase)
├── runs/              # Training run outputs and detailed logs
├── emb_cache/         # Pre-computed ProtBERT embeddings
├── scripts/           # Utility scripts and batch files
├── notebooks/         # Jupyter notebooks for analysis
└── documents/         # Project documentation
```

## 🔧 Key Files

- `phase2_fast.py` - Main training script (best performing)
- `phase2_siamese.py` - Siamese network implementation
- `analyze_training.py` - Training analysis and comparison tool
- `src/dataset/` - **NEW**: All training and test data files
- `experiments/` - All experimental variants and analysis scripts

## 📈 Performance Details

**Best Model Configuration**:
- Model: ProteinEmbeddingFusion
- Layers: 2-layer BiLSTM
- Hidden Size: 96
- Parameters: 1.2M
- Training Time: ~40 minutes for 30 epochs
- Validation Correlation: **79.74%**

## 🔄 Recent Updates

**Data Organization**:
- ✅ Moved all data files to `src/dataset/`
- ✅ Updated all scripts to use new data directory
- ✅ Cleaned up duplicate experimental files
- ✅ Organized batch scripts in `scripts/` directory

## 🚀 Usage Examples

```bash
# Basic training with default settings
python phase2_fast.py

# Training with custom data directory
python phase2_fast.py --data_dir src/dataset --epochs 30

# Siamese network training
python phase2_siamese.py --data_dir src/dataset

# Run comprehensive experiments
cd scripts/
./run_all_experiments.bat
```

## 📚 Documentation

- `PROJECT_REPORT.md` - Comprehensive experimental analysis
- `experiments/README.md` - Experimental scripts documentation
- Individual run logs in `runs/` directory

## 🔄 Next Phase

**Phase 3 - Transformer Architecture**:
- Target: 82-85% correlation
- Focus: Self-attention mechanisms
- Advantage: Better long-range dependencies

---

*Achieving 79.74% correlation through systematic deep learning experimentation*
