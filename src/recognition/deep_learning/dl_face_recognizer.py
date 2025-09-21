"""
Deep learning face recognizer using InsightFace ArcFace models.
Generates high-quality 512-dimensional face embeddings for accurate face recognition.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path
import time

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available, using ONNX fallback")

from .preprocessing import FacePreprocessor
from .dl_face_detector import Face
from src.utils.model_utils import get_model_loader

logger = logging.getLogger(__name__)

class FaceEmbedding:
    """Container for face embedding with metadata."""
    
    def __init__(self, embedding: np.ndarray, confidence: float = 1.0, 
                 model_name: str = "", quality_score: float = 0.0):
        """
        Initialize face embedding.
        
        Args:
            embedding: Face embedding vector (512-dim)
            confidence: Embedding generation confidence
            model_name: Name of the model used
            quality_score: Quality assessment score
        """
        self.embedding = embedding
        self.confidence = confidence
        self.model_name = model_name
        self.quality_score = quality_score
        self.dimension = len(embedding) if embedding is not None else 0
        self.norm = np.linalg.norm(embedding) if embedding is not None else 0.0
        
        # Normalize embedding for cosine similarity
        if embedding is not None and self.norm > 0:
            self.normalized_embedding = embedding / self.norm
        else:
            self.normalized_embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'confidence': self.confidence,
            'model_name': self.model_name,
            'quality_score': self.quality_score,
            'dimension': self.dimension,
            'norm': self.norm
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceEmbedding':
        """Create from dictionary."""
        embedding = np.array(data['embedding']) if data['embedding'] is not None else None
        return cls(
            embedding=embedding,
            confidence=data.get('confidence', 1.0),
            model_name=data.get('model_name', ''),
            quality_score=data.get('quality_score', 0.0)
        )
    
    def similarity(self, other: 'FaceEmbedding') -> float:
        """Calculate cosine similarity with another embedding."""
        if self.normalized_embedding is None or other.normalized_embedding is None:
            return 0.0
        
        # Cosine similarity using normalized embeddings
        similarity = np.dot(self.normalized_embedding, other.normalized_embedding)
        return float(np.clip(similarity, -1.0, 1.0))


class ArcFaceRecognizer:
    """ArcFace-based face recognizer using ONNX runtime."""
    
    def __init__(self, model_path: str = None, providers: List[str] = None):
        """
        Initialize ArcFace recognizer.
        
        Args:
            model_path: Path to ONNX model file
            providers: ONNX execution providers
        """
        self.model_path = model_path
        self.providers = providers
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.model_info = {}
        
        # Initialize model
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load ONNX model for inference."""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Get model loader
            model_loader = get_model_loader()
            
            # Set providers if not specified
            if self.providers is None:
                self.providers = model_loader._get_onnx_providers()
            
            # Create inference session
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Store model info
            self.model_info = {
                'model_path': self.model_path,
                'input_shape': self.input_shape,
                'providers': self.providers,
                'input_name': self.input_name,
                'output_name': self.output_name
            }
            
            logger.info(f"ArcFace model loaded: {self.model_path}")
            logger.info(f"Input shape: {self.input_shape}, Providers: {self.providers}")
            
        except Exception as e:
            logger.error(f"Failed to load ArcFace model: {e}")
            raise
    
    def get_embedding(self, face_image: np.ndarray) -> FaceEmbedding:
        """
        Generate face embedding from aligned face image.
        
        Args:
            face_image: Aligned face image (112x112x3)
            
        Returns:
            Face embedding object
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess image for ArcFace
            input_blob = self._preprocess_image(face_image)
            
            # Run inference
            start_time = time.time()
            outputs = self.session.run([self.output_name], {self.input_name: input_blob})
            inference_time = time.time() - start_time
            
            # Extract embedding
            embedding = outputs[0][0]  # Shape: (512,)
            
            # Calculate confidence based on embedding norm
            embedding_norm = np.linalg.norm(embedding)
            confidence = min(1.0, embedding_norm / 10.0)  # Heuristic confidence
            
            logger.debug(f"Generated embedding: dim={len(embedding)}, "
                        f"norm={embedding_norm:.3f}, time={inference_time*1000:.1f}ms")
            
            return FaceEmbedding(
                embedding=embedding,
                confidence=confidence,
                model_name=Path(self.model_path).stem,
                quality_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ArcFace model input."""
        # Ensure correct size (112x112)
        if image.shape[:2] != (112, 112):
            image = cv2.resize(image, (112, 112))
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)   # Add batch dimension
        
        return image
    
    def batch_get_embeddings(self, face_images: List[np.ndarray]) -> List[FaceEmbedding]:
        """Generate embeddings for multiple faces."""
        embeddings = []
        
        for i, face_image in enumerate(face_images):
            try:
                embedding = self.get_embedding(face_image)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding for face {i}: {e}")
                # Add empty embedding
                embeddings.append(FaceEmbedding(
                    embedding=None,
                    confidence=0.0,
                    model_name=Path(self.model_path).stem if self.model_path else "unknown"
                ))
        
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.model_info.copy()


class InsightFaceRecognizer:
    """InsightFace-based recognizer using the official library."""
    
    def __init__(self, model_name: str = 'arcface_r100_v1', device: str = 'cpu'):
        """
        Initialize InsightFace recognizer.
        
        Args:
            model_name: Model name for InsightFace
            device: Device to run on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        
        if INSIGHTFACE_AVAILABLE:
            self._load_model()
        else:
            raise ImportError("InsightFace not available. Use ArcFaceRecognizer instead.")
    
    def _load_model(self):
        """Load InsightFace model."""
        try:
            # Initialize InsightFace app
            app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Get recognition model
            self.model = app.models['recognition']
            
            logger.info(f"InsightFace model loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise
    
    def get_embedding(self, face_image: np.ndarray) -> FaceEmbedding:
        """Generate embedding using InsightFace."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess image
            if face_image.shape[:2] != (112, 112):
                face_image = cv2.resize(face_image, (112, 112))
            
            # Generate embedding
            embedding = self.model.get_feat(face_image)
            
            # Calculate confidence
            embedding_norm = np.linalg.norm(embedding)
            confidence = min(1.0, embedding_norm / 10.0)
            
            return FaceEmbedding(
                embedding=embedding,
                confidence=confidence,
                model_name=self.model_name,
                quality_score=confidence
            )
            
        except Exception as e:
            logger.error(f"InsightFace embedding generation failed: {e}")
            raise


class DLFaceRecognizer:
    """Main deep learning face recognizer interface."""
    
    def __init__(self, model_type: str = 'arcface', model_path: str = None, 
                 preprocessor: FacePreprocessor = None):
        """
        Initialize deep learning face recognizer.
        
        Args:
            model_type: Type of model ('arcface', 'insightface')
            model_path: Path to model file (for ONNX models)
            preprocessor: Face preprocessor instance
        """
        self.model_type = model_type
        self.model_path = model_path
        self.preprocessor = preprocessor or FacePreprocessor(target_size=(112, 112))
        self.recognizer = None
        
        # Initialize recognizer
        self._init_recognizer()
    
    def _init_recognizer(self):
        """Initialize the appropriate recognizer."""
        try:
            if self.model_type == 'arcface':
                if self.model_path is None:
                    # Try to get model path from downloader
                    from models.model_downloader import ModelDownloader
                    downloader = ModelDownloader()
                    self.model_path = downloader.get_model_path('recognition', 'arcface_r100')
                    
                    if self.model_path is None:
                        raise FileNotFoundError("ArcFace model not found. Please download first.")
                
                self.recognizer = ArcFaceRecognizer(self.model_path)
                
            elif self.model_type == 'insightface':
                if not INSIGHTFACE_AVAILABLE:
                    logger.warning("InsightFace not available, falling back to ArcFace")
                    self.model_type = 'arcface'
                    return self._init_recognizer()
                
                self.recognizer = InsightFaceRecognizer()
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize recognizer: {e}")
            raise
    
    def get_embedding(self, face_image: np.ndarray, preprocess: bool = True) -> FaceEmbedding:
        """
        Generate face embedding.
        
        Args:
            face_image: Face image
            preprocess: Whether to apply preprocessing
            
        Returns:
            Face embedding
        """
        if self.recognizer is None:
            raise RuntimeError("Recognizer not initialized")
        
        try:
            # Preprocess if requested
            if preprocess:
                result = self.preprocessor.preprocess_face(face_image)
                if result['preprocessed_image'] is None:
                    raise ValueError("Preprocessing failed")
                
                processed_image = result['preprocessed_image']
                
                # Convert back to uint8 for model input
                if processed_image.dtype == np.float32:
                    # Denormalize
                    processed_image = np.clip(processed_image * 255, 0, 255).astype(np.uint8)
                    
                # Update quality score
                embedding = self.recognizer.get_embedding(processed_image)
                embedding.quality_score = result['quality_score']
                
            else:
                embedding = self.recognizer.get_embedding(face_image)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            raise
    
    def compare_faces(self, embedding1: FaceEmbedding, embedding2: FaceEmbedding) -> float:
        """Compare two face embeddings."""
        return embedding1.similarity(embedding2)
    
    def recognize_face(self, face_image: np.ndarray, known_embeddings: List[FaceEmbedding],
                      threshold: float = 0.3) -> Tuple[Optional[int], float]:
        """
        Recognize face against known embeddings.
        
        Args:
            face_image: Face image to recognize
            known_embeddings: List of known face embeddings
            threshold: Similarity threshold for recognition
            
        Returns:
            Tuple of (best_match_index, similarity_score)
        """
        try:
            # Generate embedding for input face
            query_embedding = self.get_embedding(face_image)
            
            if query_embedding.embedding is None:
                return None, 0.0
            
            # Compare with known embeddings
            best_match_idx = None
            best_similarity = 0.0
            
            for i, known_embedding in enumerate(known_embeddings):
                if known_embedding.embedding is None:
                    continue
                
                similarity = self.compare_faces(query_embedding, known_embedding)
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match_idx = i
            
            return best_match_idx, best_similarity
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0
    
    def batch_get_embeddings(self, face_images: List[np.ndarray]) -> List[FaceEmbedding]:
        """Generate embeddings for multiple faces."""
        if self.recognizer is None:
            raise RuntimeError("Recognizer not initialized")
        
        return self.recognizer.batch_get_embeddings(face_images)
    
    def get_info(self) -> Dict[str, Any]:
        """Get recognizer information."""
        info = {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'preprocessor_info': self.preprocessor.get_preprocessing_stats()
        }
        
        if self.recognizer and hasattr(self.recognizer, 'get_model_info'):
            info['model_info'] = self.recognizer.get_model_info()
        
        return info


# Utility functions
def calculate_similarity_matrix(embeddings: List[FaceEmbedding]) -> np.ndarray:
    """Calculate similarity matrix for a list of embeddings."""
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if embeddings[i].embedding is not None and embeddings[j].embedding is not None:
                similarity_matrix[i, j] = embeddings[i].similarity(embeddings[j])
    
    return similarity_matrix


def find_best_embedding(embeddings: List[FaceEmbedding]) -> Optional[FaceEmbedding]:
    """Find the best quality embedding from a list."""
    if not embeddings:
        return None
    
    valid_embeddings = [e for e in embeddings if e.embedding is not None]
    if not valid_embeddings:
        return None
    
    # Sort by quality score and confidence
    valid_embeddings.sort(key=lambda e: (e.quality_score, e.confidence), reverse=True)
    
    return valid_embeddings[0]