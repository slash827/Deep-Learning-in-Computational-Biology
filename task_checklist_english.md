# Task Checklist - RNA-Protein Binding Project

## Phase 1: Setup and Data Exploration

### Environment Setup
- [ ] Create virtual environment
- [ ] Install required packages (torch, pandas, numpy, matplotlib)
- [ ] Set up folder structure according to plan
- [ ] Create requirements.txt

### Initial Data Exploration
- [ ] Load sample data files (2-3 RBPs)
- [ ] Analyze sequence length distributions (RNA and protein)
- [ ] Examine binding score distributions
- [ ] Visualize basic data characteristics
- [ ] Identify outliers or data quality issues

### Basic Data Processing
- [ ] Implement one-hot encoding for RNA sequences (A,U,G,C)
- [ ] Implement encoding for protein sequences (20 amino acids)
- [ ] Implement padding for variable-length sequences
- [ ] Create PyTorch Dataset class
- [ ] Implement DataLoader with batch processing

## Phase 2: Basic LSTM Model

### Model Architecture
- [ ] Implement bidirectional LSTM for RNA sequences
- [ ] Implement bidirectional LSTM for protein sequences
- [ ] Concatenation layer for feature combination
- [ ] Dense layers for final prediction
- [ ] Add Dropout for regularization

### Training Loop
- [ ] Implement basic training loop
- [ ] Implement validation loop
- [ ] Save best model (checkpoint)
- [ ] Monitor loss and metrics in real-time
- [ ] Implement early stopping

### Initial Testing
- [ ] Train on small subset (1-2 RBPs)
- [ ] Calculate Pearson correlation on validation set
- [ ] Check that model doesn't overfit
- [ ] Measure training time for one RBP

## Phase 3: Add Self-Attention

### Attention Implementation
- [ ] Add self-attention layer to RNA features
- [ ] Implement attention weights visualization
- [ ] Compare performance with basic model
- [ ] Tune number of attention heads

### Optimization
- [ ] Tune learning rate
- [ ] Tune batch size
- [ ] Tune hidden dimensions
- [ ] Experiment with different optimizers (Adam, AdamW)

## Phase 4: Cross-Attention between RNA-Protein

### Advanced Architecture
- [ ] Implement cross-attention mechanism
- [ ] Combine RNA and protein features through attention
- [ ] Implement multi-head attention
- [ ] Add positional encoding (if needed)

### Advanced Evaluation
- [ ] Test model on 10-20 RBPs
- [ ] Compare performance at each development stage
- [ ] Analyze attention weights to identify patterns
- [ ] Measure training time on larger datasets

## Phase 5: Production Preparation

### Full Training
- [ ] Train on all 200 data groups
- [ ] Measure total training time
- [ ] Ensure training time < 1 hour
- [ ] Save final model

### Implement main.py
- [ ] Read command line arguments
- [ ] Load trained model
- [ ] Process input files
- [ ] Generate predictions and save results
- [ ] Verify required output format

### Run on Test Data
- [ ] Predict on 44 test groups
- [ ] Generate 44 result files
- [ ] Verify correct output file format
- [ ] Check result reasonableness

## Phase 6: Documentation and Submission

### Write Report
- [ ] Describe problem and data
- [ ] Explain chosen architecture
- [ ] Present experimental results and comparisons
- [ ] Analyze performance (time, memory, accuracy)
- [ ] Conclusions and improvement possibilities

### Prepare Code for Review
- [ ] Clean code and remove unnecessary parts
- [ ] Add comments and documentation
- [ ] Test code on different machine
- [ ] Create README with execution instructions

### Final Submission
- [ ] Upload all files to moodle
- [ ] Verify all files are in correct format
- [ ] Final check that everything works
- [ ] Submit before deadline

## Optional Tasks (If Time Permits)

### Advanced Improvements
- [ ] Experiment with Transformer architecture
- [ ] Add additional regularization techniques
- [ ] Ensemble multiple models
- [ ] Data augmentation techniques

### Advanced Analysis
- [ ] Visualize attention patterns
- [ ] Analyze errors and edge cases
- [ ] Test generalization on unknown RBPs
- [ ] Compare to existing methods from papers

---

## Notes for Working with AI Coding Assistant

**For Progress Tracking**: 
- Mark ✅ next to completed tasks
- Add 🔄 next to tasks in progress  
- Use ❌ for tasks that were needed but not completed

**Example Status Update**:
```
- ✅ Create virtual environment
- 🔄 Install required packages  
- [ ] Set up folder structure according to plan
```

**Tips for Efficient Work**:
- Start complex tasks with small data subsets
- Save working versions of the model after each major phase
- Document decisions and results in separate notebook
- Run small tests before full training