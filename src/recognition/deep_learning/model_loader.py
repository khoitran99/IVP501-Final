"""
Model loader for deep learning face recognition models.
Handles loading and management of ArcFace, InsightFace, and other recognition models.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from models.model_downloader import ModelDownloader
from src.utils.model_utils import ModelValidator, get_model_loader
from .dl_face_recognizer import ArcFaceRecognizer, InsightFaceRecognizer

logger = logging.getLogger(__name__)

class DLModelLoader:
    """Deep learning model loader for face recognition."""
    
    def __init__(self, auto_download: bool = True):
        """
        Initialize model loader.
        
        Args:
            auto_download: Whether to automatically download missing models
        """
        self.auto_download = auto_download
        self.downloader = ModelDownloader()
        self.validator = ModelValidator()
        self.loaded_models = {}
        
        # Model configurations
        self.model_configs = {
            'arcface_r100': {
                'type': 'onnx',
                'category': 'recognition',
                'name': 'arcface_r100',
                'input_size': (112, 112),
                'embedding_dim': 512,
                'description': 'ArcFace ResNet100 - High accuracy'
            },
            'arcface_mobilenet': {
                'type': 'onnx', 
                'category': 'recognition',
                'name': 'arcface_mobilenet',
                'input_size': (112, 112),
                'embedding_dim': 128,
                'description': 'ArcFace MobileFaceNet - Fast inference'
            },
            'insightface_r50': {
                'type': 'insightface',
                'category': 'recognition', 
                'name': 'insightface_r50',
                'input_size': (112, 112),
                'embedding_dim': 512,
                'description': 'InsightFace ResNet50 - Official implementation'
            }
        }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models with their configurations."""
        available = {}
        
        for model_name, config in self.model_configs.items():
            model_info = config.copy()
            
            # Check if model is downloaded
            if config['type'] == 'onnx':
                model_path = self.downloader.get_model_path(config['category'], config['name'])
                model_info['downloaded'] = model_path is not None
                model_info['path'] = model_path
                
                if model_path:
                    # Validate model
                    is_valid, message = self.validator.validate_onnx_model(model_path)
                    model_info['valid'] = is_valid
                    model_info['validation_message'] = message
            else:
                model_info['downloaded'] = True  # InsightFace models are part of the package
                model_info['valid'] = True
            
            available[model_name] = model_info
        
        return available
    
    def download_model(self, model_name: str) -> bool:
        """
        Download a specific model.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            Success status
        """
        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        
        if config['type'] == 'onnx':
            try:
                model_path = self.downloader.download_model(config['category'], config['name'])
                logger.info(f"Downloaded model {model_name} to {model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to download model {model_name}: {e}")
                return False
        else:
            logger.info(f"Model {model_name} is part of the package, no download needed")
            return True
    
    def load_model(self, model_name: str, force_reload: bool = False) -> Union[ArcFaceRecognizer, InsightFaceRecognizer]:
        """
        Load a face recognition model.
        
        Args:
            model_name: Name of the model to load
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded model instance
        """
        # Check if already loaded
        if model_name in self.loaded_models and not force_reload:
            logger.debug(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        
        try:
            if config['type'] == 'onnx':
                # Load ONNX model
                model_path = self.downloader.get_model_path(config['category'], config['name'])
                
                if model_path is None:
                    if self.auto_download:
                        logger.info(f"Model {model_name} not found, downloading...")
                        if not self.download_model(model_name):
                            raise RuntimeError(f"Failed to download model {model_name}")
                        model_path = self.downloader.get_model_path(config['category'], config['name'])
                    else:
                        raise FileNotFoundError(f"Model {model_name} not found and auto-download disabled")
                
                # Validate model before loading
                is_valid, message = self.validator.validate_onnx_model(model_path)
                if not is_valid:
                    raise RuntimeError(f"Model validation failed: {message}")
                
                # Create recognizer instance
                recognizer = ArcFaceRecognizer(model_path)
                
            elif config['type'] == 'insightface':
                # Load InsightFace model
                recognizer = InsightFaceRecognizer(model_name=config['name'])
                
            else:
                raise ValueError(f"Unsupported model type: {config['type']}")
            
            # Cache loaded model
            self.loaded_models[model_name] = recognizer
            
            logger.info(f"Successfully loaded model: {model_name}")
            return recognizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        if model_name not in self.model_configs:
            return {}
        
        config = self.model_configs[model_name].copy()
        
        # Add runtime information
        if model_name in self.loaded_models:
            config['loaded'] = True
            config['load_time'] = datetime.now().isoformat()
            
            # Get model-specific info
            model = self.loaded_models[model_name]
            if hasattr(model, 'get_model_info'):
                config['model_details'] = model.get_model_info()
        else:
            config['loaded'] = False
        
        return config
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            Success status
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Unloaded model: {model_name}")
            return True
        return False
    
    def clear_cache(self):
        """Clear all loaded models from memory."""
        self.loaded_models.clear()
        logger.info("Cleared model cache")
    
    def get_recommended_model(self, priority: str = 'accuracy') -> str:
        """
        Get recommended model based on priority.
        
        Args:
            priority: 'accuracy', 'speed', or 'balanced'
            
        Returns:
            Recommended model name
        """
        if priority == 'accuracy':
            return 'arcface_r100'
        elif priority == 'speed':
            return 'arcface_mobilenet'
        elif priority == 'balanced':
            return 'arcface_r100'  # Good balance of speed and accuracy
        else:
            return 'arcface_r100'  # Default
    
    def benchmark_models(self, test_image_size: tuple = (112, 112), 
                        iterations: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Benchmark available models for speed comparison.
        
        Args:
            test_image_size: Size of test images
            iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        import time
        import numpy as np
        
        results = {}
        
        # Create dummy test image
        test_image = np.random.randint(0, 255, (test_image_size[0], test_image_size[1], 3), dtype=np.uint8)
        
        for model_name in self.model_configs:
            try:
                # Load model
                model = self.load_model(model_name)
                
                # Warmup
                for _ in range(3):
                    try:
                        model.get_embedding(test_image)
                    except:
                        pass
                
                # Benchmark
                times = []
                for _ in range(iterations):
                    start_time = time.time()
                    try:
                        embedding = model.get_embedding(test_image)
                        end_time = time.time()
                        if embedding.embedding is not None:
                            times.append(end_time - start_time)
                    except Exception as e:
                        logger.warning(f"Benchmark failed for {model_name}: {e}")
                        break
                
                if times:
                    results[model_name] = {
                        'avg_time_ms': np.mean(times) * 1000,
                        'min_time_ms': np.min(times) * 1000,
                        'max_time_ms': np.max(times) * 1000,
                        'std_time_ms': np.std(times) * 1000,
                        'fps': 1.0 / np.mean(times),
                        'successful_runs': len(times)
                    }
                else:
                    results[model_name] = {
                        'error': 'All benchmark runs failed'
                    }
                
            except Exception as e:
                results[model_name] = {
                    'error': f'Failed to load model: {str(e)}'
                }
        
        return results
    
    def get_loader_stats(self) -> Dict[str, Any]:
        """Get model loader statistics."""
        return {
            'available_models': len(self.model_configs),
            'loaded_models': len(self.loaded_models),
            'auto_download': self.auto_download,
            'loaded_model_names': list(self.loaded_models.keys()),
            'model_configs': self.model_configs
        }


# Global model loader instance
_dl_model_loader = None

def get_dl_model_loader() -> DLModelLoader:
    """Get global deep learning model loader instance."""
    global _dl_model_loader
    if _dl_model_loader is None:
        _dl_model_loader = DLModelLoader()
    return _dl_model_loader


def load_default_model() -> Union[ArcFaceRecognizer, InsightFaceRecognizer]:
    """Load the default face recognition model."""
    loader = get_dl_model_loader()
    default_model = loader.get_recommended_model('balanced')
    return loader.load_model(default_model)


def get_model_recommendations() -> Dict[str, str]:
    """Get model recommendations for different use cases."""
    return {
        'high_accuracy': 'arcface_r100',
        'real_time': 'arcface_mobilenet',
        'balanced': 'arcface_r100',
        'research': 'insightface_r50'
    }