# Recognition package for FaceAttend application 

# Classical recognition components
from .classical.face_detector import FaceDetector
from .classical.image_processor import ImageProcessor
from .classical.lbph_recognizer import LBPHRecognizer
from .realtime_recognizer import RealtimeRecognizer

# Deep learning components (when available)
try:
    from .deep_learning import (
        DLFaceDetector, MTCNNDetector, Face, FacePreprocessor, ImageQualityFilter,
        DLFaceRecognizer, FaceEmbedding, ArcFaceRecognizer,
        EmbeddingManager, EmbeddingStorage, EmbeddingMetadata,
        DLModelLoader, get_dl_model_loader, load_default_model
    )
    DL_AVAILABLE = True
except ImportError as e:
    DL_AVAILABLE = False
    import logging
    logging.warning(f"Deep learning components not available: {e}")

# Hybrid recognition factory (when available)
try:
    from .recognition_factory import (
        HybridRecognitionFactory,
        RecognitionMode,
        RecognitionResult,
        get_recognition_factory,
        CLASSICAL_AVAILABLE,
        DL_AVAILABLE as FACTORY_DL_AVAILABLE
    )
    FACTORY_AVAILABLE = True
except ImportError as e:
    FACTORY_AVAILABLE = False
    import logging
    logging.warning(f"Recognition factory not available: {e}")

__all__ = ['FaceDetector', 'ImageProcessor', 'LBPHRecognizer', 'RealtimeRecognizer', 'DL_AVAILABLE']

if DL_AVAILABLE:
    __all__.extend([
        'DLFaceDetector', 'MTCNNDetector', 'Face', 'FacePreprocessor', 'ImageQualityFilter',
        'DLFaceRecognizer', 'FaceEmbedding', 'ArcFaceRecognizer',
        'EmbeddingManager', 'EmbeddingStorage', 'EmbeddingMetadata',
        'DLModelLoader', 'get_dl_model_loader', 'load_default_model'
    ])

if FACTORY_AVAILABLE:
    __all__.extend([
        'HybridRecognitionFactory', 'RecognitionMode', 'RecognitionResult', 
        'get_recognition_factory', 'CLASSICAL_AVAILABLE', 'FACTORY_AVAILABLE'
    ]) 