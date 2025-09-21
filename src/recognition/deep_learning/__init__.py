"""
Deep learning face recognition module.
Contains state-of-the-art deep learning models for face detection and recognition.
"""

import os
import warnings

# Suppress albumentations version check warnings
os.environ.setdefault('ALBUMENTATIONS_DISABLE_VERSION_CHECK', '1')
warnings.filterwarnings('ignore', category=UserWarning, module='albumentations')
warnings.filterwarnings('ignore', message='.*Error fetching version info.*')

from .dl_face_detector import DLFaceDetector, MTCNNDetector, Face
from .preprocessing import FacePreprocessor, ImageQualityFilter
from .dl_face_recognizer import DLFaceRecognizer, FaceEmbedding, ArcFaceRecognizer
from .embedding_manager import EmbeddingManager, EmbeddingStorage, EmbeddingMetadata
from .model_loader import DLModelLoader, get_dl_model_loader, load_default_model

__all__ = [
    # Face Detection
    'DLFaceDetector', 
    'MTCNNDetector', 
    'Face',
    
    # Preprocessing
    'FacePreprocessor',
    'ImageQualityFilter',
    
    # Face Recognition
    'DLFaceRecognizer',
    'FaceEmbedding',
    'ArcFaceRecognizer',
    
    # Embedding Management
    'EmbeddingManager',
    'EmbeddingStorage', 
    'EmbeddingMetadata',
    
    # Model Loading
    'DLModelLoader',
    'get_dl_model_loader',
    'load_default_model'
]