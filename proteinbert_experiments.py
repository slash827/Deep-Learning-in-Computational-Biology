#!/usr/bin/env python3
"""
ProteinBERT Embedding Optimization Experiments
"""

import subprocess
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any

class ProteinBERTExperimentRunner:
    def __init__(self):
        self.results = []
        
    def create_embeddings(self, name: str, params: Dict[str, Any]) -> str:
        """Create new ProteinBERT embeddings with given parameters"""
        print(f"\n🧬 CREATING EMBEDDINGS: {name}")
        print(f"Parameters: {params}")
        
        output_path = f"runs/emb_cache/protein_bert_{name}.pt"
        
        # Build command for embedding creation
        cmd = [
            "python", "scripts/cache_proteinbert_embeddings.py",
            "--data_dir", "src/data",
            "--out", output_path
        ]
        
        for key, value in params.items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])
        
        print(f"🔧 Embedding Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                print(f"✅ Embeddings created successfully: {output_path}")
                return output_path
            else:
                print(f"❌ Embedding creation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Embedding creation timeout")
            return None
        except Exception as e:
            print(f"💥 Error creating embeddings: {e}")
            return None
    
    def test_embeddings(self, name: str, embedding_path: str) -> Dict[str, Any]:
        """Test embeddings with Siamese model"""
        print(f"\n🧪 TESTING EMBEDDINGS: {name}")
        
        # Use your best hyperparameters from previous experiments
        test_params = {
            'subset_size': 500,
            'epochs': 5,
            'loss_type': 'hybrid',
            'pair_sampling_ratio': 0.5,
            'protein_embedding_path': embedding_path,
            'learning_rate': 0.002,  # Assuming this was good from hyperparam experiments
            'contrastive_weight': 1.5,
            'regression_weight': 0.3
        }
        
        cmd = ["python", "phase2_siamese.py", "--data_dir", "src/data/"]
        
        for key, value in test_params.items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])
        
        print(f"🔧 Test Command: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            end_time = time.time()
            
            success = result.returncode == 0
            runtime = end_time - start_time
            
            # Extract correlation
            correlation = None
            if success and "Best validation correlation:" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Best validation correlation:" in line:
                        try:
                            correlation = float(line.split(':')[-1].strip())
                        except:
                            pass
            
            return {
                'name': name,
                'embedding_path': embedding_path,
                'success': success,
                'runtime': runtime,
                'correlation': correlation,
                'timestamp': datetime.now().isoformat(),
                'stdout': result.stdout[-1000:] if result.stdout else "",
                'stderr': result.stderr[-500:] if result.stderr else ""
            }
            
        except Exception as e:
            return {
                'name': name,
                'embedding_path': embedding_path,
                'success': False,
                'runtime': time.time() - start_time,
                'correlation': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# =============================================================================
# PROTEINBERT EMBEDDING EXPERIMENTS
# =============================================================================

PROTEINBERT_EXPERIMENTS = [
    
    # 1. DIFFERENT POOLING STRATEGIES
    {
        'name': 'mean_pooling',
        'params': {'pooling': 'mean', 'max_length': 1024}
    },
    {
        'name': 'cls_pooling', 
        'params': {'pooling': 'cls', 'max_length': 1024}
    },
    
    # 2. DIFFERENT SEQUENCE LENGTHS
    {
        'name': 'maxlen_512',
        'params': {'pooling': 'mean', 'max_length': 512}
    },
    {
        'name': 'maxlen_768',
        'params': {'pooling': 'mean', 'max_length': 768}
    },
    {
        'name': 'maxlen_2048',
        'params': {'pooling': 'mean', 'max_length': 2048}
    },
    
    # 3. ALTERNATIVE PROTEIN LANGUAGE MODELS
    {
        'name': 'protbert_bfd',
        'params': {'model': 'Rostlab/prot_bert_bfd', 'pooling': 'mean', 'max_length': 1024}
    },
    {
        'name': 'protbert',
        'params': {'model': 'Rostlab/prot_bert', 'pooling': 'mean', 'max_length': 1024}
    },
    
    # 4. PRECISION EXPERIMENTS (if you get GPU access)
    {
        'name': 'fp16_mean',
        'params': {'pooling': 'mean', 'max_length': 1024, 'fp16': True}
    },
    
    # 5. BATCH SIZE OPTIMIZATION
    {
        'name': 'batch_16',
        'params': {'pooling': 'mean', 'max_length': 1024, 'batch_size': 16}
    },
    {
        'name': 'batch_64',
        'params': {'pooling': 'mean', 'max_length': 1024, 'batch_size': 64}
    },
]

def run_proteinbert_experiments():
    """Run all ProteinBERT embedding experiments"""
    runner = ProteinBERTExperimentRunner()
    results = []
    
    print(f"🧬 Starting {len(PROTEINBERT_EXPERIMENTS)} ProteinBERT experiments...")
    
    for i, exp in enumerate(PROTEINBERT_EXPERIMENTS, 1):
        print(f"\n📊 Progress: {i}/{len(PROTEINBERT_EXPERIMENTS)}")
        
        # Create embeddings
        embedding_path = runner.create_embeddings(exp['name'], exp['params'])
        
        if embedding_path:
            # Test with Siamese model
            result = runner.test_embeddings(exp['name'], embedding_path)
            results.append(result)
            
            # Save intermediate results
            with open(f"proteinbert_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(results, f, indent=2)
        else:
            results.append({
                'name': exp['name'],
                'embedding_path': None,
                'success': False,
                'error': 'embedding_creation_failed',
                'timestamp': datetime.now().isoformat()
            })
    
    # Summary
    successful = [r for r in results if r['success'] and r['correlation']]
    if successful:
        best = max(successful, key=lambda x: x['correlation'])
        print(f"\n🏆 BEST PROTEINBERT CONFIGURATION:")
        print(f"   Experiment: {best['name']}")
        print(f"   Correlation: {best['correlation']:.4f}")
        print(f"   Embedding Path: {best['embedding_path']}")
    
    return results

# =============================================================================
# ADVANCED PROTEINBERT EXPERIMENTS
# =============================================================================

def create_advanced_embedding_script():
    """Create script for advanced ProteinBERT embedding strategies"""
    
    advanced_script = '''#!/usr/bin/env python3
"""
Advanced ProteinBERT Embedding Strategies
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import math

class AdvancedProteinBERTEmbedder:
    def __init__(self, model_name="Rostlab/prot_bert_bfd"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def layered_pooling(self, sequences: List[str], layers=[-4, -3, -2, -1]) -> Dict[str, torch.Tensor]:
        """Pool from multiple transformer layers"""
        results = {}
        
        for seq in sequences:
            spaced = " ".join(list(seq))
            inputs = self.tokenizer(spaced, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                # Pool from specified layers
                layer_embeddings = []
                for layer_idx in layers:
                    layer_emb = hidden_states[layer_idx][0]  # Remove batch dim
                    # Mean pooling over sequence length
                    mask = inputs["attention_mask"][0].unsqueeze(-1)
                    pooled = (layer_emb * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
                    layer_embeddings.append(pooled)
                
                # Concatenate layers
                final_emb = torch.cat(layer_embeddings, dim=0)
                results[seq] = final_emb
        
        return results
    
    def attention_weighted_pooling(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Use attention weights for smarter pooling"""
        results = {}
        
        for seq in sequences:
            spaced = " ".join(list(seq))
            inputs = self.tokenizer(spaced, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                hidden = outputs.last_hidden_state[0]  # Remove batch dim
                attentions = outputs.attentions
                
                # Use last layer attention, average over heads
                last_attention = attentions[-1][0].mean(dim=0)  # [seq_len, seq_len]
                
                # Use CLS token attention to other tokens as weights
                cls_attention = last_attention[0, 1:]  # Exclude CLS->CLS
                cls_attention = cls_attention / cls_attention.sum()
                
                # Weighted pooling
                weighted_emb = (hidden[1:] * cls_attention.unsqueeze(-1)).sum(dim=0)
                results[seq] = weighted_emb
        
        return results
    
    def motif_aware_pooling(self, sequences: List[str], motif_size=3) -> Dict[str, torch.Tensor]:
        """Pool with awareness of protein motifs"""
        results = {}
        
        for seq in sequences:
            spaced = " ".join(list(seq))
            inputs = self.tokenizer(spaced, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state[0][1:-1]  # Remove CLS and SEP
                
                # Create sliding windows for motifs
                seq_len = hidden.shape[0]
                motif_embeddings = []
                
                for i in range(0, seq_len - motif_size + 1, motif_size):
                    motif_emb = hidden[i:i+motif_size].mean(dim=0)
                    motif_embeddings.append(motif_emb)
                
                if motif_embeddings:
                    # Pool motif embeddings
                    final_emb = torch.stack(motif_embeddings).mean(dim=0)
                else:
                    final_emb = hidden.mean(dim=0)
                
                results[seq] = final_emb
        
        return results

def create_advanced_embeddings():
    """Create embeddings with advanced strategies"""
    from pathlib import Path
    
    # Load proteins
    data_dir = Path("src/data")
    protein_files = [data_dir / "training_RBPs2.txt", data_dir / "test_RBPs2.txt"]
    
    proteins = []
    for fp in protein_files:
        if fp.exists():
            with fp.open('r') as f:
                proteins.extend([line.strip() for line in f if line.strip()])
    
    unique_proteins = sorted(list(set(proteins)))
    
    embedder = AdvancedProteinBERTEmbedder()
    
    # Create different embedding types
    strategies = {
        'layered_4layers': lambda: embedder.layered_pooling(unique_proteins),
        'layered_2layers': lambda: embedder.layered_pooling(unique_proteins, layers=[-2, -1]),
        'attention_weighted': lambda: embedder.attention_weighted_pooling(unique_proteins),
        'motif_aware_3': lambda: embedder.motif_aware_pooling(unique_proteins, motif_size=3),
        'motif_aware_5': lambda: embedder.motif_aware_pooling(unique_proteins, motif_size=5),
    }
    
    for name, strategy in strategies.items():
        print(f"Creating {name} embeddings...")
        embeddings = strategy()
        
        output_path = f"runs/emb_cache/protein_bert_{name}.pt"
        torch.save(embeddings, output_path)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    create_advanced_embeddings()
'''
    
    with open("advanced_proteinbert.py", "w") as f:
        f.write(advanced_script)
    
    print("✅ Created advanced_proteinbert.py")

if __name__ == "__main__":
    results = run_proteinbert_experiments()
    print(f"\n✅ Completed {len(results)} ProteinBERT experiments")
    
    # Also create advanced embedding script
    create_advanced_embedding_script()
