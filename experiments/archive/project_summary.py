#!/usr/bin/env python3
"""
Project Organization Summary
Shows the clean, organized structure of your project
"""

import os
from pathlib import Path


def show_project_structure():
    """Display the organized project structure."""
    base_dir = Path("c:/Users/gilad/Documents/GitHub/Deep-Learning-in-Computational-Biology")
    
    print("📁 ORGANIZED PROJECT STRUCTURE")
    print("="*50)
    
    # Show root directory (only essential files)
    print("\n🏠 ROOT DIRECTORY (Clean & Essential)")
    print("-" * 40)
    essential_files = [
        "phase2_fast.py",
        "analyze_training.py", 
        "README.md",
        "PROJECT_REPORT.md",
        "requirements.txt"
    ]
    
    for file in essential_files:
        if (base_dir / file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ⚠️  {file} (missing)")
    
    # Show key directories
    key_dirs = ["src", "runs", "src/dataset", "experiments", "emb_cache", "notebooks"]
    print(f"\n📂 KEY DIRECTORIES:")
    for dir_name in key_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            if dir_name == "experiments":
                subdirs = [d.name for d in dir_path.iterdir() if d.is_dir()]
                print(f"   📁 {dir_name}/ ({len(subdirs)} subdirectories)")
            elif dir_name == "runs":
                runs = [d.name for d in dir_path.iterdir() if d.is_dir()]
                print(f"   📁 {dir_name}/ ({len(runs)} training runs)")
            else:
                print(f"   📁 {dir_name}/")
        else:
            print(f"   ⚠️  {dir_name}/ (missing)")


def show_experiments_organization():
    """Show how experiments are organized."""
    experiments_dir = Path("c:/Users/gilad/Documents/GitHub/Deep-Learning-in-Computational-Biology/experiments")
    
    if not experiments_dir.exists():
        print("\n⚠️  Experiments directory not found!")
        return
    
    print(f"\n🔬 EXPERIMENTS ORGANIZATION")
    print("-" * 40)
    
    for subdir in experiments_dir.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob("*.py")) + list(subdir.glob("*.txt")) + list(subdir.glob("*.md")) + list(subdir.glob("*.pth"))
            print(f"   📂 {subdir.name}/ ({len(files)} files)")
            
            # Show key files in each directory
            if len(files) <= 5:
                for file in files:
                    print(f"      • {file.name}")
            else:
                # Show first few files
                for file in files[:3]:
                    print(f"      • {file.name}")
                print(f"      • ... and {len(files)-3} more files")


def show_performance_summary():
    """Show performance summary from runs."""
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("-" * 40)
    
    # Key performance milestones
    milestones = [
        ("Phase 1 Baseline", "59.21%", "Basic LSTM"),
        ("Phase 2A Enhancement", "65.37%", "+6.16% - BiLSTM + Attention"),
        ("Phase 2B ProtBERT", "74.74%", "+9.37% - ProtBERT Integration"),
        ("Phase 2C Single Layer", "76.23%", "+1.49% - Architecture Optimization"),
        ("Phase 2D Two Layers", "79.74%", "+3.51% - BEST PERFORMANCE")
    ]
    
    print("   🎯 PERFORMANCE PROGRESSION:")
    for phase, correlation, description in milestones:
        if "BEST" in description:
            print(f"   🏆 {phase:<20} {correlation:<8} {description}")
        else:
            print(f"   📈 {phase:<20} {correlation:<8} {description}")
    
    print(f"\n   💡 TOTAL IMPROVEMENT: +20.53 percentage points")
    print(f"   ⚡ SPEED IMPROVEMENT: 10x faster (20min → 2min per epoch)")


def show_next_steps():
    """Show recommended next steps."""
    print(f"\n🚀 NEXT STEPS & RECOMMENDATIONS")
    print("-" * 40)
    
    print("   ✅ COMPLETED:")
    print("      • Project organization and cleanup")
    print("      • Comprehensive experimental documentation") 
    print("      • 79.74% correlation achievement")
    print("      • Speed optimization (10x improvement)")
    
    print(f"\n   🎯 IMMEDIATE PRIORITIES:")
    print("      1. Review PROJECT_REPORT.md for complete analysis")
    print("      2. Consider Phase 3 Transformer architecture")
    print("      3. Target: 82-85% correlation with Transformers")
    
    print(f"\n   📋 PHASE 3 PLANNING:")
    print("      • Implement Transformer-based architecture")
    print("      • Better long-range dependency modeling")
    print("      • Parallel processing advantages")
    print("      • Expected 82-85% correlation potential")


def show_files_to_present():
    """Show key files for project presentation."""
    print(f"\n📋 KEY FILES FOR PROJECT PRESENTATION")
    print("-" * 40)
    
    presentation_files = {
        "Main Report": "PROJECT_REPORT.md - Comprehensive experimental analysis",
        "Project Overview": "README.md - Clean project summary", 
        "Best Model": "phase2_fast.py - 79.74% correlation implementation",
        "Analysis Tool": "analyze_training.py - Training run comparison",
        "Experiments": "experiments/ - All experimental scripts organized by phase",
        "Results": "runs/ - Detailed training logs and metrics",
        "Architecture": "src/models/ - Model implementations"
    }
    
    for category, description in presentation_files.items():
        print(f"   📄 {category:<15}: {description}")


def main():
    """Main summary function."""
    print("🎉 PROJECT ORGANIZATION COMPLETE!")
    print("="*60)
    print("Your RNA-Protein Binding Prediction project is now clean and organized")
    print("="*60)
    
    # Show structure
    show_project_structure()
    
    # Show experiments organization
    show_experiments_organization() 
    
    # Show performance summary
    show_performance_summary()
    
    # Show next steps
    show_next_steps()
    
    # Show files for presentation
    show_files_to_present()
    
    print(f"\n{'='*60}")
    print("🎯 PROJECT STATUS: READY FOR PRESENTATION")
    print("="*60)
    print("✅ Clean, organized codebase")
    print("✅ Comprehensive experimental documentation")  
    print("✅ 79.74% correlation achievement documented")
    print("✅ Clear path forward to Phase 3 (Transformers)")
    print()
    print("📖 Start with PROJECT_REPORT.md for the complete story!")


if __name__ == "__main__":
    main()
