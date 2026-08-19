"""
src/models/bert_model.py
========================
Compatibility layer mapping BertClassifier to RobertaClassifier.
"""

from src.models.roberta_model import RobertaClassifier as BertClassifier, TextDataset

__all__ = ["BertClassifier", "TextDataset"]
