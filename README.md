# Deep Learning for RNA-Protein Binding Prediction

Advanced deep learning models for predicting RNA-protein binding affinities

## Project Highlights

- **Best Performance**: **79.74% correlation coefficient**
- **Architecture**: ProteinEmbeddingFusion with ProtBERT embeddings
- **Speed**: 10x training acceleration (20min → 2min per epoch)
- **Innovation**: Novel integration of protein language models

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run best performing model
python phase2_fast.py --protein_encoder protbert_cached --protein_embedding_path emb_cache/protein_bert.pt

# Analyze training results
python analyze_training.py --compare
```

## Key Results

| Phase | Architecture | Correlation | Improvement |
|-------|-------------|-------------|-------------|
| Phase 1 | Basic LSTM | 59.21% | Baseline |
| Phase 2A | BiLSTM + Attention | 65.37% | +6.16% |
| Phase 2B | + ProtBERT | 74.74% | +9.37% |
| Phase 2C | Optimized Single Layer | 76.23% | +1.49% |
| **Phase 2D** | **Two-Layer Fusion** | **79.74%** | **+3.51%** |

## Architecture

```
RNA Sequences → BiLSTM (2-layer) → Attention → 
                                              → Fusion → Binding Score
Proteins → ProtBERT → BiLSTM (2-layer) → Attention →
```

## Project Structure

- `src/` - Core implementation modules
- `experiments/` - All experimental scripts (organized by phase)
- `runs/` - Training run outputs and detailed logs  
- `data/` - Training and validation datasets
- `emb_cache/` - Pre-computed ProtBERT embeddings
- `PROJECT_REPORT.md` - Comprehensive experimental analysis

## Key Files

- `phase2_fast.py` - Main training script (best performing)
- `analyze_training.py` - Training analysis and comparison tool
- `PROJECT_REPORT.md` - Detailed experimental report
- `src/models/` - Model architectures
- `experiments/` - All experimental variants and analysis

## Performance Details

**Best Model Configuration**:
- Model: ProteinEmbeddingFusion
- Layers: 2-layer BiLSTM
- Hidden Size: 96
- Parameters: 1.2M
- Training Time: ~40 minutes for 30 epochs
- Validation Correlation: **79.74%**

## Next Phase

**Phase 3 - Transformer Architecture**:
- Target: 82-85% correlation
- Focus: Self-attention mechanisms
- Advantage: Better long-range dependencies

## Documentation

See `PROJECT_REPORT.md` for:
- Complete experimental progression
- Detailed performance analysis  
- Technical innovations and insights
- Future research directions

---

*Achieving 79.74% correlation through systematic deep learning experimentation*
