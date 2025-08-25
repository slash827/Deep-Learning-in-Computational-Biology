"""
Prediction module עם תמיכה בחלבונים חדשים
"""

from .flexible_embedder import FlexibleProteinEmbedder
from .flexible_predictor import FlexiblePredictor

__all__ = ['FlexibleProteinEmbedder', 'FlexiblePredictor']
