# Work Plan - RNA-Protein Binding Prediction Project

## Project Overview

**Objective**: Develop a deep learning model to predict RNA-binding protein (RBP) binding intensity to RNA sequences based on protein sequence and RNA sequence.

**Data**: 
- 200 training sets (each containing protein sequence + ~120K RNA sequences with binding scores)
- 44 test sets for final predictions

**Evaluation Metric**: Pearson correlation between predictions and actual scores

## Development Strategy

### Phase 1: Basic LSTM Model (Week 1)
- Implement simple bidirectional LSTM
- Train on subset of data
- Establish data pipeline

### Phase 2: Add Self-Attention (Week 2)
- Add attention mechanism to RNA sequence
- Compare performance to basic model

### Phase 3: Cross-Attention between RNA-Protein (Week 3)
- Implement interaction between protein and RNA sequences
- Optimize hyperparameters

### Phase 4: Optimization and Performance (Week 4)
- Fine-tune advanced model
- Train on full dataset
- Generate predictions on test sets

### Phase 5 (Optional): Experiment with Transformer
- Only if time and computational resources allow

## Proposed Code Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Data loading and processing
│   │   └── preprocessing.py    # one-hot encoding, padding
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_basic.py       # Basic LSTM
│   │   ├── lstm_attention.py   # LSTM with attention
│   │   └── transformer.py      # Transformer (optional)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loops
│   │   └── evaluation.py       # Metrics and evaluation
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_final_predictions.ipynb
├── main.py                     # Main execution file
├── requirements.txt
└── README.md
```

## Technologies and Resources

- **Framework**: PyTorch
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Experiment tracking**: wandb (recommended)
- **Development**: Jupyter notebooks for exploration + Python files for production code

## Performance Considerations

- **Maximum training time**: 1 hour (per project requirements)
- **Memory**: Adapt to available GPU
- **Optimization**: Use efficient Data Loaders, gradient accumulation if needed

## Success Metrics

1. **Accuracy**: Pearson correlation > 0.3 (based on data distribution)
2. **Efficiency**: Training time < 3600 seconds
3. **Code Quality**: Modular structure, clear documentation

## Important Notes

- Start with small data subset for rapid development
- Save checkpoints of good models
- Monitor overfitting with validation set
- Document all experiments and results

## Final Deliverables

1. **Production Code**: `main.py` that accepts input files and returns predictions
2. **44 Result Files**: For each RBP in test set
3. **Project Report**: 3-5 pages describing model and results
4. **Presentation**: For class presentation

---

## Proposed Timeline

| Week | Main Tasks | Goals |
|------|------------|-------|
| 1 | Data exploration + Basic LSTM | Working model with initial results |
| 2 | Self-attention + optimization | Performance improvement |
| 3 | Cross-attention + advanced tuning | Production-ready model |
| 4 | Final training + submission prep | Final results and report |

**Note**: Timeline is flexible and depends on personal work pace and technical complexity discovered.