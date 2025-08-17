#!/usr/bin/env python3
"""
Project Organization Script
Cleans up the main directory and organizes experimental files
"""

import os
import shutil
from pathlib import Path


def organize_project():
    """Organize the project by moving experimental files and creating clean structure."""
    base_dir = Path("c:/Users/gilad/Documents/GitHub/Deep-Learning-in-Computational-Biology")
    experiments_dir = base_dir / "experiments"
    
    # Create subdirectories in experiments
    phase_dirs = {
        "phase1": experiments_dir / "phase1_baseline",
        "phase2_main": experiments_dir / "phase2_main_development", 
        "phase2_optimization": experiments_dir / "phase2_optimizations",
        "analysis": experiments_dir / "analysis_scripts",
        "archive": experiments_dir / "archive"
    }
    
    for dir_path in phase_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Files to move to experiments directory
    experimental_files = {
        # Phase 1 files
        "phase1_main.py": "phase1_baseline",
        "phase1_model.pth": "phase1_baseline",
        
        # Phase 2 main development files
        "phase2_main.py": "phase2_main_development",
        "phase2_simple.py": "phase2_main_development",
        "phase2_fast.py": "phase2_main_development",  # Keep copy in root too
        
        # Phase 2 optimization experiments
        "phase2_ultra_fast.py": "phase2_optimizations",
        "phase2_high_accuracy.py": "phase2_optimizations", 
        "phase2_improved.py": "phase2_optimizations",
        "improve_phase2_fast.py": "phase2_optimizations",
        "improve_0.7623_run.py": "phase2_optimizations",
        
        # Analysis scripts
        "analyze_79_74_success.py": "analysis_scripts",
        "architecture_guide_and_phase3_prep.py": "analysis_scripts",
        "batch_size_analysis.py": "analysis_scripts",
        "optimization_summary.py": "analysis_scripts",
        "solution_summary.py": "analysis_scripts",
        "test_enhanced_summary.py": "analysis_scripts",
        "test_improvements.py": "analysis_scripts", 
        "test_improvements_simple.py": "analysis_scripts",
        "test_improvement_fix.py": "analysis_scripts",
        
        # Experiment runners
        "run_experiments.py": "phase2_optimizations",
        "quick_experiments.py": "phase2_optimizations",
        
        # Archive older files
        "run_results_bilstm.txt": "archive",
        "task_checklist_english.md": "archive",
        "project_plan_english.md": "archive",
        "OPTIMIZATION_README.md": "archive"
    }
    
    print("🗂️  ORGANIZING PROJECT FILES")
    print("="*50)
    
    for filename, target_dir in experimental_files.items():
        source_path = base_dir / filename
        target_path = experiments_dir / target_dir / filename
        
        if source_path.exists():
            try:
                # Copy file to experiments directory
                shutil.copy2(source_path, target_path)
                print(f"✅ Copied {filename} → experiments/{target_dir}/")
                
                # Remove from root (except for key files we want to keep)
                if filename not in ["phase2_fast.py", "analyze_training.py"]:
                    source_path.unlink()
                    print(f"🗑️  Removed {filename} from root")
                else:
                    print(f"📌 Kept {filename} in root (primary file)")
                    
            except Exception as e:
                print(f"❌ Error moving {filename}: {e}")
        else:
            print(f"⚠️  File not found: {filename}")
    
    print(f"\n📁 Created experiment organization:")
    for name, path in phase_dirs.items():
        files = list(path.glob("*.py")) + list(path.glob("*.txt")) + list(path.glob("*.md")) + list(path.glob("*.pth"))
        print(f"   {name}: {len(files)} files")


def create_experiments_readme():
    """Create README for experiments directory."""
    readme_content = """# Experiments Directory

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
"""
    
    experiments_dir = Path("c:/Users/gilad/Documents/GitHub/Deep-Learning-in-Computational-Biology/experiments")
    readme_path = experiments_dir / "README.md"
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Created experiments/README.md")


def create_clean_root_readme():
    """Create a clean README for the root directory."""
    readme_content = """# Deep Learning for RNA-Protein Binding Prediction

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
python phase2_fast.py --protein_encoder protbert_cached --protein_embedding_path emb_cache/protein_bert.pt

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

- `src/` - Core implementation modules
- `experiments/` - All experimental scripts (organized by phase)
- `runs/` - Training run outputs and detailed logs  
- `data/` - Training and validation datasets
- `emb_cache/` - Pre-computed ProtBERT embeddings
- `PROJECT_REPORT.md` - Comprehensive experimental analysis

## 🔬 Key Files

- `phase2_fast.py` - Main training script (best performing)
- `analyze_training.py` - Training analysis and comparison tool
- `PROJECT_REPORT.md` - Detailed experimental report
- `src/models/` - Model architectures
- `experiments/` - All experimental variants and analysis

## 📈 Performance Details

**Best Model Configuration**:
- Model: ProteinEmbeddingFusion
- Layers: 2-layer BiLSTM
- Hidden Size: 96
- Parameters: 1.2M
- Training Time: ~40 minutes for 30 epochs
- Validation Correlation: **79.74%**

## 🔄 Next Phase

**Phase 3 - Transformer Architecture**:
- Target: 82-85% correlation
- Focus: Self-attention mechanisms
- Advantage: Better long-range dependencies

## 📚 Documentation

See `PROJECT_REPORT.md` for:
- Complete experimental progression
- Detailed performance analysis  
- Technical innovations and insights
- Future research directions

---

*Achieving 79.74% correlation through systematic deep learning experimentation*
"""
    
    root_path = Path("c:/Users/gilad/Documents/GitHub/Deep-Learning-in-Computational-Biology/README.md")
    
    with open(root_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Updated root README.md")


def main():
    """Main organization function."""
    print("🗂️  PROJECT ORGANIZATION AND CLEANUP")
    print("="*60)
    print("This script will:")
    print("1. Move experimental files to experiments/ directory")
    print("2. Create organized subdirectories by phase")
    print("3. Keep essential files in root")
    print("4. Create documentation")
    print()
    
    response = input("Proceed with organization? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Organization cancelled")
        return
    
    # Organize project
    organize_project()
    
    # Create documentation
    create_experiments_readme()
    create_clean_root_readme()
    
    print(f"\n✅ PROJECT ORGANIZATION COMPLETE!")
    print("="*50)
    print("📁 Structure:")
    print("   Root: Clean with only essential files")
    print("   experiments/: All experimental scripts organized by phase") 
    print("   PROJECT_REPORT.md: Comprehensive experimental analysis")
    print("   README.md: Clean overview for repository")
    
    print(f"\n📋 Files kept in root:")
    print("   • phase2_fast.py (main training script)")
    print("   • analyze_training.py (analysis tool)")
    print("   • PROJECT_REPORT.md (comprehensive report)")
    print("   • README.md (project overview)")
    print("   • Core directories: src/, runs/, data/, etc.")


if __name__ == "__main__":
    main()
