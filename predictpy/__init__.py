"""Predictpy - A smart predictive text system with personal learning.

This package provides tools for next-word prediction based on n-gram language models,
personal usage patterns, and semantic completion using ChromaDB.
"""

from .predictor import WordPredictor
from .personal import PersonalModel
from .engine import WordPredictionEngine
from .api import Predictpy
from .semantic import SemanticMemory

__all__ = ["WordPredictor", "PersonalModel", "WordPredictionEngine", "Predictpy", "SemanticMemory"]

__version__ = "0.5.2"
