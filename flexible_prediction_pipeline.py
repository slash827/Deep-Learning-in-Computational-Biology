#!/usr/bin/env python3
"""
Pipeline לprediction עם יכולת טיפול בחלבונים חדשים
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Union
import os
from pathlib import Path

class FlexibleProteinEmbedder:
    """
    יוצר embeddings לחלבונים - מ-cache אם קיים, אחרת בזמן אמת
    """
    
    def __init__(self, 
                 cache_path: Optional[str] = None,
                 model_name: str = "Rostlab/prot_bert_bfd",
                 device: str = "auto",
                 max_length: int = 1024):
        """
        Args:
            cache_path: נתיב ל-cache קיים (אופציונלי)
            model_name: שם המודל מ-HuggingFace
            device: מכשיר (auto/cpu/cuda)
            max_length: אורך מקסימלי של רצף
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # קביעת device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # טעינת cache אם קיים
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            print(f"🔄 טוען cache מ-{cache_path}")
            self.cache = torch.load(cache_path, map_location='cpu')
            print(f"✅ נטענו {len(self.cache)} חלבונים מ-cache")
        
        # אתחול מודל ProteinBERT (lazy loading)
        self.tokenizer = None
        self.model = None
        
    def _init_model(self):
        """אתחול מודל ProteinBERT (רק בפעם הראשונה שצריך)"""
        if self.tokenizer is None:
            print(f"🔄 טוען מודל ProteinBERT: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval().to(self.device)
            print(f"✅ מודל נטען על {self.device}")
    
    def get_embedding(self, protein_seq: str) -> torch.Tensor:
        """
        מחזיר embedding לחלבון - מ-cache או יוצר חדש
        
        Args:
            protein_seq: רצף החלבון
            
        Returns:
            torch.Tensor: embedding של 1024 dim
        """
        # בדוק אם יש ב-cache
        if protein_seq in self.cache:
            embedding = self.cache[protein_seq]
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding)
            return embedding.float()
        
        # אחרת - צור embedding חדש
        print(f"⚡ יוצר embedding חדש לחלבון באורך {len(protein_seq)}")
        return self._create_fresh_embedding(protein_seq)
    
    def _create_fresh_embedding(self, protein_seq: str) -> torch.Tensor:
        """יוצר embedding חדש באמצעות ProteinBERT"""
        self._init_model()  # אתחול מודל אם צריך
        
        # טוקניזציה
        spaced = " ".join(list(protein_seq))
        tokens = self.tokenizer(
            spaced,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # העברה למכשיר
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # יצירת embedding
        with torch.no_grad():
            outputs = self.model(**tokens)
            hidden = outputs.last_hidden_state  # (1, L, 1024)
            
            # Mean pooling (כמו ב-cache)
            mask = tokens["attention_mask"].unsqueeze(-1)  # (1, L, 1)
            summed = (hidden * mask).sum(dim=1)  # (1, 1024)
            lengths = mask.sum(dim=1).clamp(min=1)  # (1, 1)
            embedding = summed / lengths  # (1, 1024)
        
        return embedding.squeeze(0).cpu().float()  # (1024,)
    
    def get_embeddings_batch(self, protein_seqs: List[str]) -> Dict[str, torch.Tensor]:
        """מחזיר embeddings לרשימת חלבונים"""
        results = {}
        new_proteins = []
        
        # איסוף מ-cache
        for seq in protein_seqs:
            if seq in self.cache:
                embedding = self.cache[seq]
                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding)
                results[seq] = embedding.float()
            else:
                new_proteins.append(seq)
        
        # יצירת embeddings חדשים batch
        if new_proteins:
            print(f"⚡ יוצר embeddings חדשים ל-{len(new_proteins)} חלבונים")
            for seq in new_proteins:
                results[seq] = self._create_fresh_embedding(seq)
        
        return results

class FlexiblePredictor:
    """
    Predictor שיכול לטפל בחלבונים חדשים
    """
    
    def __init__(self, 
                 model_path: str,
                 cache_path: Optional[str] = None,
                 device: str = "auto"):
        """
        Args:
            model_path: נתיב למודל המאומן
            cache_path: נתיב ל-protein embeddings cache
            device: מכשיר
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # טעינת המודל המאומן
        print(f"🔄 טוען מודל מ-{model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # כאן צריך לטעון את המודל לפי הarchitecture שלך
        # לדוגמה - SiameseProteinBERT או ProteinEmbeddingFusion
        # self.model = load_your_model(checkpoint)
        # self.model.eval()
        
        # אתחול embedder
        self.embedder = FlexibleProteinEmbedder(
            cache_path=cache_path,
            device=self.device
        )
        
    def predict_binding(self, rna_seq: str, protein_seq: str) -> float:
        """
        חיזוי binding score לזוג RNA-חלבון
        
        Args:
            rna_seq: רצף RNA
            protein_seq: רצף חלבון (יכול להיות חדש!)
            
        Returns:
            float: binding score
        """
        # הכנת RNA encoding
        rna_encoded = self._encode_rna(rna_seq)
        
        # הכנת protein embedding (מ-cache או חדש)
        protein_embedding = self.embedder.get_embedding(protein_seq)
        
        # חיזוי
        with torch.no_grad():
            rna_tensor = torch.FloatTensor(rna_encoded).unsqueeze(0).to(self.device)
            protein_tensor = protein_embedding.unsqueeze(0).to(self.device)
            
            # כאן תקרא למודל שלך
            # score = self.model(rna_tensor, protein_tensor)
            # return score.item()
            
            # זמנית - החזר ציון מדומה
            return 0.5
    
    def predict_protein_vs_rnas(self, protein_seq: str, rna_seqs: List[str]) -> List[float]:
        """
        חיזוי binding scores לחלבון אחד מול רשימת RNAs
        
        Args:
            protein_seq: רצף חלבון (יכול להיות חדש!)
            rna_seqs: רשימת רצפי RNA
            
        Returns:
            List[float]: binding scores
        """
        # הכנת protein embedding פעם אחת
        protein_embedding = self.embedder.get_embedding(protein_seq)
        
        scores = []
        for rna_seq in rna_seqs:
            # הכנת RNA encoding
            rna_encoded = self._encode_rna(rna_seq)
            
            # חיזוי
            with torch.no_grad():
                rna_tensor = torch.FloatTensor(rna_encoded).unsqueeze(0).to(self.device)
                protein_tensor = protein_embedding.unsqueeze(0).to(self.device)
                
                # כאן תקרא למודל שלך
                # score = self.model(rna_tensor, protein_tensor)
                # scores.append(score.item())
                
                # זמנית - ציון מדומה
                scores.append(0.5)
        
        return scores
    
    def _encode_rna(self, rna_seq: str, max_length: int = 60) -> np.ndarray:
        """קידוד RNA (אותה שיטה כמו באימון)"""
        # A=0, U=1, G=2, C=3, N=4
        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        
        # המרה לאינדקסים
        indices = [mapping.get(base, 4) for base in rna_seq.upper()]
        
        # חיתוך/padding
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices.extend([4] * (max_length - len(indices)))  # padding עם N
        
        # one-hot encoding
        one_hot = np.zeros((max_length, 5))
        for i, idx in enumerate(indices):
            one_hot[i, idx] = 1.0
        
        return one_hot

def demo_usage():
    """הדגמה של השימוש"""
    print("🧪 הדגמת השימוש ב-FlexiblePredictor")
    print("=" * 50)
    
    # אתחול
    # predictor = FlexiblePredictor(
    #     model_path="path/to/your/model.pt",
    #     cache_path="runs/emb_cache/protein_bert.pt"
    # )
    
    # חלבון חדש (לא ב-cache)
    new_protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKREQTPVQGRNILKYKGKVHYSQIVVEVVEVGSIVGAYVPMPSDSHQNTPYLWTQSAEVHPFQPESPTAASAHTTVYKRGDVGVAASTKAGKTIHVGVNFGDGAEGVTVRVFQPHGKLTPHTQLLVALGAPVTVIGTFNIYVVDSIDYYHREASPLLGAEKL"
    
    # רצפי RNA
    rna_seqs = [
        "AGCUAGCUAGCU",
        "GGGCCCUUUAAA",
        "AUCGAUCGAUCG"
    ]
    
    print("📊 דוגמת השימוש:")
    print(f"🧬 חלבון חדש: {new_protein[:50]}...")
    print(f"🧬 RNAs: {len(rna_seqs)} רצפים")
    print()
    
    # חיזוי (דמה)
    print("⚡ מבצע חיזוי...")
    print("  - בודק אם החלבון ב-cache...")
    print("  - לא נמצא -> יוצר embedding חדש")
    print("  - מבצע חיזוי לכל RNA")
    print()
    
    # תוצאות מדומות
    scores = [0.72, 0.35, 0.68]
    print("📊 תוצאות:")
    for i, (rna, score) in enumerate(zip(rna_seqs, scores)):
        print(f"  {i+1}. {rna} -> {score:.3f}")

if __name__ == "__main__":
    demo_usage()
