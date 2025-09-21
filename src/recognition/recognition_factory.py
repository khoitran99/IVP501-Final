"""
Recognition Factory for FaceAttend - Hybrid Classical/Deep Learning Support
Provides unified interface for switching between different face recognition engines.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.utils.logger import get_module_logger
from src.utils.exceptions import FaceRecognitionError

# Import classical recognition
try:
    from src.recognition.classical.lbph_recognizer import LBPHRecognizer
    CLASSICAL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Classical recognition not available: {e}")
    CLASSICAL_AVAILABLE = False

# Import deep learning recognition
try:
    from src.recognition.deep_learning.dl_face_recognizer import DLFaceRecognizer
    from src.recognition.deep_learning.embedding_manager import EmbeddingManager
    DL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Deep learning recognition not available: {e}")
    DL_AVAILABLE = False


class RecognitionMode(Enum):
    """Recognition mode enumeration"""
    CLASSICAL = "classical"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class RecognitionResult:
    """Unified recognition result"""
    user_id: Optional[str]
    confidence: float
    method: str
    embedding: Optional[np.ndarray] = None
    face_location: Optional[Tuple[int, int, int, int]] = None
    processing_time: float = 0.0
    quality_score: float = 0.0
    
    @property
    def is_recognized(self) -> bool:
        """Check if face was successfully recognized"""
        return self.user_id is not None and self.confidence > 0.5


@dataclass
class PerformanceMetrics:
    """Performance metrics for recognition engines"""
    total_recognitions: int = 0
    successful_recognitions: int = 0
    average_confidence: float = 0.0
    average_processing_time: float = 0.0
    last_recognition_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_recognitions == 0:
            return 0.0
        return (self.successful_recognitions / self.total_recognitions) * 100


class BaseRecognitionEngine(ABC):
    """Abstract base class for recognition engines"""
    
    @abstractmethod
    def train(self, user_id: str, face_images: List[np.ndarray]) -> bool:
        """Train the recognition model with face images"""
        pass
    
    @abstractmethod
    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize a face in the given image"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the recognition model"""
        pass
    
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if the model is trained and ready"""
        pass


class ClassicalRecognitionEngine(BaseRecognitionEngine):
    """Wrapper for classical LBPH recognition"""
    
    def __init__(self):
        self.logger = get_module_logger("ClassicalEngine")
        if not CLASSICAL_AVAILABLE:
            raise FaceRecognitionError("Classical recognition not available")
        
        self.recognizer = LBPHRecognizer()
        self.metrics = PerformanceMetrics()
    
    def train(self, user_id: str, face_images: List[np.ndarray]) -> bool:
        """Train classical recognizer"""
        try:
            return self.recognizer.train_user(user_id, face_images)
        except Exception as e:
            self.logger.error(f"Classical training failed: {e}")
            return False
    
    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize using classical method"""
        import time
        start_time = time.time()
        
        try:
            result = self.recognizer.recognize_face(face_image)
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics.total_recognitions += 1
            self.metrics.last_recognition_time = processing_time
            
            if result and result.get('user_id'):
                self.metrics.successful_recognitions += 1
                confidence = result.get('confidence', 0.0)
                
                # Update average confidence
                total_successful = self.metrics.successful_recognitions
                if total_successful == 1:
                    self.metrics.average_confidence = confidence
                else:
                    self.metrics.average_confidence = (
                        (self.metrics.average_confidence * (total_successful - 1) + confidence) 
                        / total_successful
                    )
                
                return RecognitionResult(
                    user_id=result['user_id'],
                    confidence=confidence,
                    method="classical_lbph",
                    processing_time=processing_time,
                    quality_score=result.get('quality_score', 0.0)
                )
            else:
                return RecognitionResult(
                    user_id=None,
                    confidence=0.0,
                    method="classical_lbph",
                    processing_time=processing_time
                )
        
        except Exception as e:
            self.logger.error(f"Classical recognition failed: {e}")
            return RecognitionResult(
                user_id=None,
                confidence=0.0,
                method="classical_lbph",
                processing_time=time.time() - start_time
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get classical model information"""
        info = self.recognizer.get_model_info()
        info['engine_type'] = 'classical'
        info['method'] = 'LBPH'
        return info
    
    def is_trained(self) -> bool:
        """Check if classical model is trained"""
        return self.recognizer.get_model_info().get('is_trained', False)


class DeepLearningRecognitionEngine(BaseRecognitionEngine):
    """Wrapper for deep learning recognition"""
    
    def __init__(self):
        self.logger = get_module_logger("DeepLearningEngine")
        if not DL_AVAILABLE:
            raise FaceRecognitionError("Deep learning recognition not available")
        
        self.recognizer = DLFaceRecognizer()
        self.embedding_manager = EmbeddingManager()
        self.metrics = PerformanceMetrics()
    
    def train(self, user_id: str, face_images: List[np.ndarray]) -> bool:
        """Train deep learning recognizer"""
        try:
            # Generate embeddings for all face images
            embeddings = []
            for image in face_images:
                embedding = self.recognizer.generate_embedding(image)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if embeddings:
                # Store embeddings
                success = self.embedding_manager.add_user_embeddings(user_id, embeddings)
                if success:
                    self.logger.info(f"Deep learning training completed for {user_id}: {len(embeddings)} embeddings")
                return success
            else:
                self.logger.warning(f"No valid embeddings generated for {user_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Deep learning training failed: {e}")
            return False
    
    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize using deep learning method"""
        import time
        start_time = time.time()
        
        try:
            # Generate embedding for input image
            query_embedding = self.recognizer.generate_embedding(face_image)
            if query_embedding is None:
                return RecognitionResult(
                    user_id=None,
                    confidence=0.0,
                    method="deep_learning_arcface",
                    processing_time=time.time() - start_time
                )
            
            # Find best match
            result = self.embedding_manager.find_best_match(query_embedding)
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics.total_recognitions += 1
            self.metrics.last_recognition_time = processing_time
            
            if result and result.get('user_id'):
                self.metrics.successful_recognitions += 1
                confidence = result.get('similarity', 0.0)
                
                # Update average confidence
                total_successful = self.metrics.successful_recognitions
                if total_successful == 1:
                    self.metrics.average_confidence = confidence
                else:
                    self.metrics.average_confidence = (
                        (self.metrics.average_confidence * (total_successful - 1) + confidence) 
                        / total_successful
                    )
                
                return RecognitionResult(
                    user_id=result['user_id'],
                    confidence=confidence,
                    method="deep_learning_arcface",
                    embedding=query_embedding.embedding,
                    processing_time=processing_time,
                    quality_score=query_embedding.quality_score
                )
            else:
                return RecognitionResult(
                    user_id=None,
                    confidence=0.0,
                    method="deep_learning_arcface",
                    embedding=query_embedding.embedding,
                    processing_time=processing_time,
                    quality_score=query_embedding.quality_score
                )
        
        except Exception as e:
            self.logger.error(f"Deep learning recognition failed: {e}")
            return RecognitionResult(
                user_id=None,
                confidence=0.0,
                method="deep_learning_arcface",
                processing_time=time.time() - start_time
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get deep learning model information"""
        try:
            model_info = self.recognizer.get_model_info()
            embedding_info = self.embedding_manager.get_embedding_stats()
            
            return {
                'engine_type': 'deep_learning',
                'method': 'ArcFace',
                'model_name': model_info.get('model_name', 'unknown'),
                'embedding_dim': model_info.get('embedding_dim', 512),
                'is_trained': embedding_info.get('total_users', 0) > 0,
                'total_users': embedding_info.get('total_users', 0),
                'total_embeddings': embedding_info.get('total_embeddings', 0),
                'average_quality': embedding_info.get('average_quality', 0.0)
            }
        except Exception as e:
            self.logger.error(f"Error getting deep learning model info: {e}")
            return {
                'engine_type': 'deep_learning',
                'method': 'ArcFace',
                'is_trained': False,
                'error': str(e)
            }
    
    def is_trained(self) -> bool:
        """Check if deep learning model has embeddings"""
        try:
            stats = self.embedding_manager.get_embedding_stats()
            return stats.get('total_users', 0) > 0
        except Exception:
            return False


class HybridRecognitionFactory:
    """Factory for managing multiple recognition engines"""
    
    def __init__(self, default_mode: RecognitionMode = RecognitionMode.AUTO):
        self.logger = get_module_logger("RecognitionFactory")
        self.mode = default_mode
        self.engines: Dict[str, BaseRecognitionEngine] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        # Initialize available engines
        self._initialize_engines()
        
        # Auto-select best engine if in AUTO mode
        if self.mode == RecognitionMode.AUTO:
            self._select_best_engine()
    
    def _initialize_engines(self):
        """Initialize available recognition engines"""
        # Initialize classical engine
        if CLASSICAL_AVAILABLE:
            try:
                self.engines['classical'] = ClassicalRecognitionEngine()
                self.logger.info("Classical recognition engine initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize classical engine: {e}")
        
        # Initialize deep learning engine
        if DL_AVAILABLE:
            try:
                self.engines['deep_learning'] = DeepLearningRecognitionEngine()
                self.logger.info("Deep learning recognition engine initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize deep learning engine: {e}")
        
        if not self.engines:
            raise FaceRecognitionError("No recognition engines available")
        
        self.logger.info(f"Initialized {len(self.engines)} recognition engines: {list(self.engines.keys())}")
    
    def _select_best_engine(self):
        """Auto-select the best performing engine"""
        if not self.engines:
            return
        
        # Prefer deep learning if available and trained
        if 'deep_learning' in self.engines and self.engines['deep_learning'].is_trained():
            self.mode = RecognitionMode.DEEP_LEARNING
            self.logger.info("Auto-selected deep learning engine (trained)")
            return
        
        # Fall back to classical if available and trained
        if 'classical' in self.engines and self.engines['classical'].is_trained():
            self.mode = RecognitionMode.CLASSICAL
            self.logger.info("Auto-selected classical engine (trained)")
            return
        
        # Use any available engine
        engine_name = list(self.engines.keys())[0]
        if engine_name == 'classical':
            self.mode = RecognitionMode.CLASSICAL
        elif engine_name == 'deep_learning':
            self.mode = RecognitionMode.DEEP_LEARNING
        
        self.logger.info(f"Auto-selected {engine_name} engine (default)")
    
    def set_recognition_mode(self, mode: RecognitionMode):
        """Set the recognition mode"""
        if mode == RecognitionMode.CLASSICAL and 'classical' not in self.engines:
            raise FaceRecognitionError("Classical recognition not available")
        elif mode == RecognitionMode.DEEP_LEARNING and 'deep_learning' not in self.engines:
            raise FaceRecognitionError("Deep learning recognition not available")
        
        self.mode = mode
        self.logger.info(f"Recognition mode set to: {mode.value}")
        
        if mode == RecognitionMode.AUTO:
            self._select_best_engine()
    
    def get_recognition_mode(self) -> RecognitionMode:
        """Get current recognition mode"""
        return self.mode
    
    def train_user(self, user_id: str, face_images: List[np.ndarray]) -> Dict[str, bool]:
        """Train user with multiple engines"""
        results = {}
        
        # Train with specific engine based on mode
        if self.mode == RecognitionMode.CLASSICAL:
            if 'classical' in self.engines:
                results['classical'] = self.engines['classical'].train(user_id, face_images)
        
        elif self.mode == RecognitionMode.DEEP_LEARNING:
            if 'deep_learning' in self.engines:
                results['deep_learning'] = self.engines['deep_learning'].train(user_id, face_images)
        
        elif self.mode in [RecognitionMode.HYBRID, RecognitionMode.AUTO]:
            # Train with all available engines
            for engine_name, engine in self.engines.items():
                results[engine_name] = engine.train(user_id, face_images)
        
        # Log results
        successful_engines = [name for name, success in results.items() if success]
        self.logger.info(f"Training completed for {user_id}: {successful_engines}")
        
        return results
    
    def recognize_face(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize face using current mode"""
        if self.mode == RecognitionMode.CLASSICAL:
            return self._recognize_with_engine('classical', face_image)
        
        elif self.mode == RecognitionMode.DEEP_LEARNING:
            return self._recognize_with_engine('deep_learning', face_image)
        
        elif self.mode == RecognitionMode.HYBRID:
            return self._recognize_hybrid(face_image)
        
        elif self.mode == RecognitionMode.AUTO:
            return self._recognize_auto(face_image)
        
        else:
            raise FaceRecognitionError(f"Unknown recognition mode: {self.mode}")
    
    def _recognize_with_engine(self, engine_name: str, face_image: np.ndarray) -> RecognitionResult:
        """Recognize with specific engine"""
        if engine_name not in self.engines:
            raise FaceRecognitionError(f"Engine {engine_name} not available")
        
        return self.engines[engine_name].recognize(face_image)
    
    def _recognize_hybrid(self, face_image: np.ndarray) -> RecognitionResult:
        """Recognize using hybrid approach (both engines, best result)"""
        results = []
        
        # Run recognition with all available engines
        for engine_name, engine in self.engines.items():
            try:
                result = engine.recognize(face_image)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Recognition failed with {engine_name}: {e}")
        
        if not results:
            return RecognitionResult(
                user_id=None,
                confidence=0.0,
                method="hybrid_no_results"
            )
        
        # Select best result based on confidence
        best_result = max(results, key=lambda r: r.confidence)
        best_result.method = f"hybrid_{best_result.method}"
        
        return best_result
    
    def _recognize_auto(self, face_image: np.ndarray) -> RecognitionResult:
        """Auto recognition with intelligent engine selection"""
        # For AUTO mode, use the currently selected engine from _select_best_engine
        if self.mode == RecognitionMode.AUTO:
            # Re-evaluate best engine
            self._select_best_engine()
        
        # Use the appropriate engine based on current mode
        if hasattr(self, '_current_auto_engine'):
            return self._recognize_with_engine(self._current_auto_engine, face_image)
        
        # Fallback to hybrid
        return self._recognize_hybrid(face_image)
    
    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all engines"""
        info = {}
        for engine_name, engine in self.engines.items():
            try:
                info[engine_name] = engine.get_model_info()
            except Exception as e:
                info[engine_name] = {'error': str(e)}
        
        return info
    
    def get_performance_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics for all engines"""
        metrics = {}
        for engine_name, engine in self.engines.items():
            if hasattr(engine, 'metrics'):
                metrics[engine_name] = engine.metrics
        
        return metrics
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engines"""
        return list(self.engines.keys())
    
    def is_engine_trained(self, engine_name: str) -> bool:
        """Check if specific engine is trained"""
        if engine_name not in self.engines:
            return False
        return self.engines[engine_name].is_trained()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'current_mode': self.mode.value,
            'available_engines': self.get_available_engines(),
            'engine_info': self.get_engine_info(),
            'performance_metrics': {
                name: {
                    'total_recognitions': metrics.total_recognitions,
                    'success_rate': metrics.success_rate,
                    'avg_confidence': metrics.average_confidence,
                    'avg_processing_time': metrics.average_processing_time
                }
                for name, metrics in self.get_performance_metrics().items()
            },
            'classical_available': CLASSICAL_AVAILABLE,
            'deep_learning_available': DL_AVAILABLE
        }


# Global factory instance
_recognition_factory = None

def get_recognition_factory() -> HybridRecognitionFactory:
    """Get global recognition factory instance"""
    global _recognition_factory
    if _recognition_factory is None:
        _recognition_factory = HybridRecognitionFactory()
    return _recognition_factory

def reset_recognition_factory():
    """Reset global recognition factory (for testing)"""
    global _recognition_factory
    _recognition_factory = None