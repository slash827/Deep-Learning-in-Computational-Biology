"""
FlexiblePredictor - חיזוי binding scores עם תמיכה בחלבונים חדשים
"""

import torch
import numpy as np
from typing import List, Optional, Union
import os

from .flexible_embedder import FlexibleProteinEmbedder
from ..data.preprocessing import encode_rna_sequence

class FlexiblePredictor:
    """
    Predictor שיכול לטפל בחלבונים חדשים באמצעות on-the-fly embeddings
    """
    
    def __init__(self, 
                 model_path: str,
                 cache_path: Optional[str] = None,
                 device: str = "auto",
                 rna_max_length: int = 60):
        """
        Args:
            model_path: נתיב למודל המאומן (.pt)
            cache_path: נתיב לprotein embedding cache
            device: 'auto', 'cpu', או 'cuda'
            rna_max_length: אורך מקסימלי לRNA
        """
        # קביעת device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.rna_max_length = rna_max_length
        
        # טעינת המודל המאומן
        print(f"🔄 טוען מודל מ-{model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"קובץ המודל לא נמצא: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # כאן צריך לטעון את המודל לפי הarchitecture שלך
        # דוגמה לSiameseProteinBERT או ProteinEmbeddingFusion:
        
        # self.model = YourModelClass(...)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.eval().to(self.device)
        
        # זמנית - placeholder
        self.model = None
        print(f"⚠️ צריך להשלים טעינת המודל בflexible_predictor.py")
        
        # אתחול embedder
        self.embedder = FlexibleProteinEmbedder(
            cache_path=cache_path,
            device=self.device
        )
        
        print(f"✅ FlexiblePredictor מוכן על {self.device}")
    
    def predict_binding_score(self, rna_seq: str, protein_seq: str) -> float:
        """
        חיזוי binding score לזוג RNA-חלבון
        
        Args:
            rna_seq: רצף RNA
            protein_seq: רצף חלבון (יכול להיות חדש!)
            
        Returns:
            float: binding score
        """
        # הכנת RNA encoding
        rna_encoded = encode_rna_sequence(rna_seq, self.rna_max_length)
        rna_tensor = torch.FloatTensor(rna_encoded).unsqueeze(0).to(self.device)
        
        # הכנת protein embedding (חכם - מcache או חדש)
        protein_embedding = self.embedder.get_embedding(protein_seq)
        protein_tensor = protein_embedding.unsqueeze(0).to(self.device)
        
        # חיזוי
        with torch.no_grad():
            if self.model is not None:
                score = self.model(rna_tensor, protein_tensor)
                return score.item()
            else:
                # placeholder - החזר ציון מדומה
                return 0.5
    
    def predict_protein_vs_rnas(self, 
                               protein_seq: str, 
                               rna_seqs: List[str],
                               batch_size: int = 32) -> List[float]:
        """
        חיזוי binding scores לחלבון אחד מול רשימת RNAs
        
        Args:
            protein_seq: רצף חלבון (יכול להיות חדש!)
            rna_seqs: רשימת רצפי RNA
            batch_size: גודל batch לprocessing
            
        Returns:
            List[float]: binding scores באותו סדר כמו rna_seqs
        """
        # הכנת protein embedding פעם אחת
        protein_embedding = self.embedder.get_embedding(protein_seq)
        
        scores = []
        
        # עיבוד ב-batches
        for i in range(0, len(rna_seqs), batch_size):
            batch_rnas = rna_seqs[i:i + batch_size]
            
            # הכנת batch
            batch_rna_tensors = []
            for rna_seq in batch_rnas:
                rna_encoded = encode_rna_sequence(rna_seq, self.rna_max_length)
                batch_rna_tensors.append(torch.FloatTensor(rna_encoded))
            
            # stack לbatch
            rna_batch = torch.stack(batch_rna_tensors).to(self.device)
            
            # repeat protein embedding לכל RNA בbatch
            protein_batch = protein_embedding.unsqueeze(0).repeat(len(batch_rnas), 1).to(self.device)
            
            # חיזוי
            with torch.no_grad():
                if self.model is not None:
                    batch_scores = self.model(rna_batch, protein_batch)
                    scores.extend([score.item() for score in batch_scores])
                else:
                    # placeholder
                    scores.extend([0.5] * len(batch_rnas))
        
        return scores
    
    def predict_rnas_vs_proteins(self,
                                rna_seqs: List[str],
                                protein_seqs: List[str]) -> np.ndarray:
        """
        חיזוי binding scores לmatrix של RNAs × proteins
        
        Args:
            rna_seqs: רשימת רצפי RNA
            protein_seqs: רשימת רצפי חלבונים
            
        Returns:
            np.ndarray: matrix בגודל (len(rna_seqs), len(protein_seqs))
        """
        print(f"🔄 מבצע prediction ל-{len(rna_seqs)} RNAs × {len(protein_seqs)} proteins")
        
        # הכנת protein embeddings לכל החלבונים
        protein_embeddings = self.embedder.get_embeddings_batch(protein_seqs)
        
        results = np.zeros((len(rna_seqs), len(protein_seqs)))
        
        for i, rna_seq in enumerate(rna_seqs):
            if i % 1000 == 0:
                print(f"  עיבוד RNA {i+1}/{len(rna_seqs)}")
                
            for j, protein_seq in enumerate(protein_seqs):
                score = self.predict_binding_score(rna_seq, protein_seq)
                results[i, j] = score
        
        return results
    
    def get_stats(self) -> dict:
        """מחזיר סטטיסטיקות על השימוש"""
        embedder_stats = self.embedder.get_stats()
        return {
            "device": self.device,
            "model_loaded": self.model is not None,
            "rna_max_length": self.rna_max_length,
            **embedder_stats
        }
    
    def save_new_embeddings(self, save_path: str):
        """שומר embeddings חדשים שנוצרו בsession"""
        self.embedder.save_runtime_cache(save_path)
