"""
FlexibleProteinEmbedder - יוצר embeddings לחלבונים עם fallback לחלבונים חדשים
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Dict, List
import os

class FlexibleProteinEmbedder:
    """
    יוצר ProteinBERT embeddings עם תמיכה בחלבונים חדשים:
    - חלבון ב-cache -> טוען מהר מהקובץ
    - חלבון חדש -> יוצר embedding בזמן אמת
    """
    
    def __init__(self, 
                 cache_path: Optional[str] = None,
                 model_name: str = "Rostlab/prot_bert_bfd",
                 device: str = "auto",
                 max_length: int = 1024):
        """
        Args:
            cache_path: נתיב לprotein_bert.pt cache (אופציונלי)
            model_name: מודל ProteinBERT
            device: 'auto', 'cpu', או 'cuda'
            max_length: אורך מקסימלי לטוקניזציה
        """
        # קביעת device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.max_length = max_length
        
        # טעינת cache אם קיים
        self.cache = {}
        self.cache_loaded = False
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = torch.load(cache_path, map_location='cpu')
                self.cache_loaded = True
                print(f"✅ נטען cache עם {len(self.cache)} חלבונים מ-{cache_path}")
            except Exception as e:
                print(f"⚠️ שגיאה בטעינת cache: {e}")
        
        # ProteinBERT (lazy loading)
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        
        # cache זמני לחלבונים חדשים
        self.runtime_cache = {}
    
    def get_embedding(self, protein_seq: str) -> torch.Tensor:
        """
        מחזיר embedding לחלבון
        
        Args:
            protein_seq: רצף החלבון
            
        Returns:
            torch.Tensor: embedding בגודל (1024,)
        """
        # בדוק cache ראשי
        if protein_seq in self.cache:
            embedding = self.cache[protein_seq]
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding)
            return embedding.float()
        
        # בדוק runtime cache
        if protein_seq in self.runtime_cache:
            return self.runtime_cache[protein_seq]
        
        # צור embedding חדש
        print(f"⚡ יוצר embedding חדש לחלבון (אורך: {len(protein_seq)})")
        embedding = self._create_embedding(protein_seq)
        
        # שמור בruntime cache
        self.runtime_cache[protein_seq] = embedding
        
        return embedding
    
    def _create_embedding(self, protein_seq: str) -> torch.Tensor:
        """יוצר embedding חדש באמצעות ProteinBERT"""
        self._ensure_model_loaded()
        
        # טוקניזציה (ProteinBERT מצפה לרווחים בין חומצות אמינו)
        spaced_seq = " ".join(list(protein_seq))
        
        inputs = self.tokenizer(
            spaced_seq,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # העברה למכשיר
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # יצירת embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state  # (1, seq_len, 1024)
            
            # Mean pooling (כמו בcache המקורי)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
            masked_hidden = hidden * attention_mask
            summed = masked_hidden.sum(dim=1)  # (1, 1024)
            lengths = attention_mask.sum(dim=1).clamp(min=1)  # (1, 1)
            embedding = summed / lengths  # (1, 1024)
        
        return embedding.squeeze(0).cpu().float()  # (1024,)
    
    def _ensure_model_loaded(self):
        """טוען את ProteinBERT אם עדיין לא נטען"""
        if not self.model_loaded:
            print(f"🔄 טוען ProteinBERT ({self.model_name}) על {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                do_lower_case=False
            )
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval().to(self.device)
            
            self.model_loaded = True
            print(f"✅ ProteinBERT נטען בהצלחה על {self.device}")
    
    def get_embeddings_batch(self, protein_seqs: List[str]) -> Dict[str, torch.Tensor]:
        """
        מחזיר embeddings לרשימת חלבונים
        
        Args:
            protein_seqs: רשימת רצפי חלבונים
            
        Returns:
            Dict[str, torch.Tensor]: מיפוי מרצף לembedding
        """
        results = {}
        new_proteins = []
        
        # איסוף מcaches
        for seq in protein_seqs:
            if seq in self.cache:
                embedding = self.cache[seq]
                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding)
                results[seq] = embedding.float()
            elif seq in self.runtime_cache:
                results[seq] = self.runtime_cache[seq]
            else:
                new_proteins.append(seq)
        
        # יצירת embeddings חדשים
        if new_proteins:
            print(f"⚡ יוצר embeddings חדשים ל-{len(new_proteins)} חלבונים")
            for seq in new_proteins:
                embedding = self._create_embedding(seq)
                self.runtime_cache[seq] = embedding
                results[seq] = embedding
        
        return results
    
    def save_runtime_cache(self, save_path: str):
        """שומר את הruntime cache לקובץ"""
        if self.runtime_cache:
            torch.save(self.runtime_cache, save_path)
            print(f"💾 נשמרו {len(self.runtime_cache)} embeddings חדשים ב-{save_path}")
    
    def get_stats(self) -> Dict:
        """מחזיר סטטיסטיקות על השימוש"""
        return {
            "cache_loaded": self.cache_loaded,
            "original_cache_size": len(self.cache),
            "runtime_cache_size": len(self.runtime_cache),
            "model_loaded": self.model_loaded,
            "device": self.device
        }
