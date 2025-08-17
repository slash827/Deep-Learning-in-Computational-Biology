# Experiments Directory

This directory contains all experimental scripts and analysis files organized by phase and purpose.

## Directory Structure

### phase1_baseline/
- `phase1_main.py`: Initial LSTM baseline implementation
- `phase1_model.pth`: Saved baseline model
- **Performance**: 59.21% correlation
- **Key Issues**: Slow training, basic architecture

### phase2_main_development/
- `phase2_main.py`: Main Phase 2 development script
- `phase2_simple.py`: Simplified Phase 2 implementation  
- `phase2_fast.py`: Optimized Phase 2 implementation (primary)
- **Key Innovation**: ProtBERT embeddings integration

### phase2_optimizations/
- `phase2_ultra_fast.py`: Speed optimization experiments
- `phase2_high_accuracy.py`: Accuracy-focused experiments
- `phase2_improved.py`: General improvements
- `improve_phase2_fast.py`: Specific fast implementation improvements
- `improve_0.7623_run.py`: Optimization from 76.23% baseline
- `run_experiments.py`: Batch experiment runner
- `quick_experiments.py`: Quick experimental tests
- **Focus**: Hyperparameter tuning and architecture optimization

### analysis_scripts/
- `analyze_79_74_success.py`: Analysis of 79.74% breakthrough
- `architecture_guide_and_phase3_prep.py`: Architecture analysis and Phase 3 planning
- `batch_size_analysis.py`: Batch size trade-off analysis
- `optimization_summary.py`: Comprehensive optimization summary
- `solution_summary.py`: Solution methodology summary
- `test_*.py`: Various testing and improvement scripts
- **Purpose**: Deep analysis and understanding of experimental results

### archive/
- Historical files and documentation
- Planning documents and old results
- Deprecated experimental approaches

## Key Experimental Progression

1. **Phase 1**: Basic LSTM (59.21% correlation)
2. **Phase 2A**: Architecture enhancement (65.37% correlation)
3. **Phase 2B**: ProtBERT integration (74.74% correlation) 
4. **Phase 2C**: Single layer optimization (76.23% correlation)
5. **Phase 2D**: Two-layer architecture (79.74% correlation - BEST)

## Current Status

- **Best Performance**: 79.74% correlation
- **Best Model**: ProteinEmbeddingFusion (2-layer)
- **Next Phase**: Transformer architecture (Phase 3)
- **Target**: 82-85% correlation

## Usage

Each phase directory contains the experimental scripts used during that development phase. 
The main working script remains in the root directory (`phase2_fast.py`) while all 
experimental variants are organized here for reference and analysis.
