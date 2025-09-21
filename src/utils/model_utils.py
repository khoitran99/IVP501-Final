"""
Model utilities for deep learning components.
Provides common functionality for model loading, validation, and management.
"""

import os
import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Unified model loader for different model formats."""
    
    def __init__(self):
        """Initialize model loader with device detection."""
        self.device = self._get_optimal_device()
        self.loaded_models = {}
    
    def _get_optimal_device(self) -> str:
        """Determine optimal device for model inference."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon GPU
        else:
            return 'cpu'
    
    def load_onnx_model(self, model_path: str, providers: Optional[list] = None) -> ort.InferenceSession:
        """
        Load ONNX model for inference.
        
        Args:
            model_path: Path to ONNX model file
            providers: ONNX execution providers
            
        Returns:
            ONNX inference session
        """
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set providers based on device availability
        if providers is None:
            providers = self._get_onnx_providers()
        
        try:
            session = ort.InferenceSession(model_path, providers=providers)
            self.loaded_models[model_path] = session
            logger.info(f"Loaded ONNX model: {model_path}")
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX model {model_path}: {e}")
            raise
    
    def load_pytorch_model(self, model_path: str, model_class=None) -> torch.nn.Module:
        """
        Load PyTorch model for inference.
        
        Args:
            model_path: Path to PyTorch model file
            model_class: Model class for loading
            
        Returns:
            PyTorch model
        """
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load model weights
            if model_class:
                model = model_class()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                model = torch.load(model_path, map_location=self.device)
            
            model.eval()
            model.to(self.device)
            self.loaded_models[model_path] = model
            logger.info(f"Loaded PyTorch model: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load PyTorch model {model_path}: {e}")
            raise
    
    def _get_onnx_providers(self) -> list:
        """Get available ONNX execution providers."""
        providers = []
        
        # Try to use GPU providers first
        if self.device == 'cuda':
            providers.extend(['CUDAExecutionProvider'])
        elif self.device == 'mps':
            # Note: CoreMLExecutionProvider for Apple Silicon
            providers.extend(['CoreMLExecutionProvider'])
        
        # Always include CPU as fallback
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about a model file."""
        if not os.path.exists(model_path):
            return {'exists': False}
        
        stat = os.stat(model_path)
        info = {
            'exists': True,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': stat.st_mtime,
            'format': self._detect_model_format(model_path)
        }
        
        # Try to get model-specific info
        try:
            if info['format'] == 'onnx':
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                info['inputs'] = [inp.name for inp in session.get_inputs()]
                info['outputs'] = [out.name for out in session.get_outputs()]
                info['input_shapes'] = [inp.shape for inp in session.get_inputs()]
                info['output_shapes'] = [out.shape for out in session.get_outputs()]
        except Exception as e:
            logger.warning(f"Could not extract model info from {model_path}: {e}")
        
        return info
    
    def _detect_model_format(self, model_path: str) -> str:
        """Detect model file format."""
        ext = Path(model_path).suffix.lower()
        
        if ext == '.onnx':
            return 'onnx'
        elif ext in ['.pt', '.pth']:
            return 'pytorch'
        elif ext == '.pb':
            return 'tensorflow'
        else:
            return 'unknown'
    
    def clear_cache(self):
        """Clear loaded model cache."""
        self.loaded_models.clear()
        logger.info("Model cache cleared")


class ModelValidator:
    """Validates model files and their integrity."""
    
    @staticmethod
    def validate_onnx_model(model_path: str) -> Tuple[bool, str]:
        """
        Validate ONNX model file.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import onnx
            
            if not os.path.exists(model_path):
                return False, "Model file does not exist"
            
            # Load and check ONNX model
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            # Try to create inference session
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            return True, "Valid ONNX model"
            
        except Exception as e:
            return False, f"Invalid ONNX model: {str(e)}"
    
    @staticmethod
    def validate_pytorch_model(model_path: str) -> Tuple[bool, str]:
        """
        Validate PyTorch model file.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(model_path):
                return False, "Model file does not exist"
            
            # Try to load model
            torch.load(model_path, map_location='cpu')
            
            return True, "Valid PyTorch model"
            
        except Exception as e:
            return False, f"Invalid PyTorch model: {str(e)}"
    
    @staticmethod
    def check_model_compatibility(model_path: str, expected_inputs: list = None, 
                                expected_outputs: list = None) -> Tuple[bool, str]:
        """
        Check if model is compatible with expected inputs/outputs.
        
        Args:
            model_path: Path to model file
            expected_inputs: List of expected input names
            expected_outputs: List of expected output names
            
        Returns:
            Tuple of (is_compatible, message)
        """
        try:
            if model_path.endswith('.onnx'):
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                
                actual_inputs = [inp.name for inp in session.get_inputs()]
                actual_outputs = [out.name for out in session.get_outputs()]
                
                if expected_inputs and set(expected_inputs) != set(actual_inputs):
                    return False, f"Input mismatch. Expected: {expected_inputs}, Got: {actual_inputs}"
                
                if expected_outputs and set(expected_outputs) != set(actual_outputs):
                    return False, f"Output mismatch. Expected: {expected_outputs}, Got: {actual_outputs}"
                
                return True, "Model is compatible"
            
            else:
                return True, "Compatibility check not implemented for this model format"
                
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"


class ModelBenchmark:
    """Benchmark model performance."""
    
    def __init__(self, model_loader: ModelLoader):
        """Initialize benchmark with model loader."""
        self.model_loader = model_loader
    
    def benchmark_inference_speed(self, model_path: str, input_shape: Tuple[int, ...], 
                                 num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            model_path: Path to model file
            input_shape: Shape of input tensor
            num_iterations: Number of inference iterations
            
        Returns:
            Performance metrics dictionary
        """
        import time
        
        try:
            if model_path.endswith('.onnx'):
                session = self.model_loader.load_onnx_model(model_path)
                input_name = session.get_inputs()[0].name
                
                # Create dummy input
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
                
                # Warmup
                for _ in range(10):
                    session.run(None, {input_name: dummy_input})
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    session.run(None, {input_name: dummy_input})
                end_time = time.time()
                
                total_time = end_time - start_time
                avg_time = total_time / num_iterations
                fps = 1.0 / avg_time
                
                return {
                    'total_time_seconds': total_time,
                    'average_time_ms': avg_time * 1000,
                    'fps': fps,
                    'iterations': num_iterations
                }
            
            else:
                return {'error': 'Benchmarking only supported for ONNX models currently'}
                
        except Exception as e:
            return {'error': f'Benchmarking failed: {str(e)}'}
    
    def estimate_memory_usage(self, model_path: str) -> Dict[str, Union[float, str]]:
        """
        Estimate model memory usage.
        
        Returns:
            Memory usage information
        """
        try:
            import psutil
            import gc
            
            # Get baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load model
            if model_path.endswith('.onnx'):
                session = self.model_loader.load_onnx_model(model_path)
            else:
                model = self.model_loader.load_pytorch_model(model_path)
            
            # Measure memory after loading
            loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
            model_memory = loaded_memory - baseline_memory
            
            return {
                'baseline_memory_mb': baseline_memory,
                'loaded_memory_mb': loaded_memory,
                'model_memory_mb': model_memory,
                'file_size_mb': os.path.getsize(model_path) / 1024 / 1024
            }
            
        except Exception as e:
            return {'error': f'Memory estimation failed: {str(e)}'}


# Global model loader instance
_model_loader = None

def get_model_loader() -> ModelLoader:
    """Get global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader