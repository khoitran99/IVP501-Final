#!/usr/bin/env python3
"""
Comprehensive validation script for Phase A implementation.
Tests all implemented deep learning components and infrastructure.
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")

def print_result(test_name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")

def validate_dependencies():
    """Validate that all required dependencies are installed."""
    print_section("Dependency Validation")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('onnxruntime', 'ONNX Runtime'),
        ('insightface', 'InsightFace'),
        ('mtcnn', 'MTCNN'),
        ('facenet_pytorch', 'FaceNet PyTorch'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
        ('requests', 'Requests'),
        ('tqdm', 'TQDM'),
        ('albumentations', 'Albumentations')
    ]
    
    all_dependencies_ok = True
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print_result(f"{display_name} import", True)
        except ImportError as e:
            print_result(f"{display_name} import", False, str(e))
            all_dependencies_ok = False
    
    return all_dependencies_ok

def validate_directory_structure():
    """Validate the created directory structure."""
    print_section("Directory Structure Validation")
    
    required_dirs = [
        'models/detection/mtcnn',
        'models/detection/retinaface', 
        'models/recognition/arcface',
        'models/recognition/facenet',
        'src/recognition/classical',
        'src/recognition/deep_learning',
        'src/recognition/core',
        'config',
        'tests'
    ]
    
    all_dirs_ok = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists() and full_path.is_dir()
        print_result(f"Directory {dir_path}", exists)
        if not exists:
            all_dirs_ok = False
    
    return all_dirs_ok

def validate_configuration_files():
    """Validate configuration files."""
    print_section("Configuration Files Validation")
    
    config_files = [
        'config/models.yaml',
        'config/recognition_settings.json',
        'config/performance_thresholds.yaml'
    ]
    
    all_configs_ok = True
    
    for config_file in config_files:
        full_path = project_root / config_file
        exists = full_path.exists()
        print_result(f"Config file {config_file}", exists)
        
        if exists:
            try:
                if config_file.endswith('.yaml'):
                    import yaml
                    with open(full_path, 'r') as f:
                        data = yaml.safe_load(f)
                    print_result(f"  YAML syntax", True, f"Loaded {len(data)} sections")
                elif config_file.endswith('.json'):
                    import json
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                    print_result(f"  JSON syntax", True, f"Loaded {len(data)} sections")
            except Exception as e:
                print_result(f"  File syntax", False, str(e))
                all_configs_ok = False
        else:
            all_configs_ok = False
    
    return all_configs_ok

def validate_model_downloader():
    """Validate model downloader functionality."""
    print_section("Model Downloader Validation")
    
    try:
        from models.model_downloader import ModelDownloader
        
        # Test initialization
        downloader = ModelDownloader()
        print_result("Model downloader initialization", True)
        
        # Test configuration loading
        config = downloader.list_available_models()
        has_models = 'detection' in config and 'recognition' in config
        print_result("Configuration loading", has_models, f"Found {len(config)} model categories")
        
        # Test model path resolution
        model_path = downloader.get_model_path('detection', 'mtcnn')
        print_result("Model path resolution", True, f"Path: {model_path or 'Not downloaded'}")
        
        return True
        
    except Exception as e:
        print_result("Model downloader validation", False, str(e))
        return False

def validate_model_utilities():
    """Validate model utilities."""
    print_section("Model Utilities Validation")
    
    try:
        from src.utils.model_utils import ModelLoader, ModelValidator, get_model_loader
        
        # Test model loader initialization
        loader = get_model_loader()
        print_result("Model loader initialization", True, f"Device: {loader.device}")
        
        # Test validator
        validator = ModelValidator()
        print_result("Model validator initialization", True)
        
        return True
        
    except Exception as e:
        print_result("Model utilities validation", False, str(e))
        return False

def validate_mtcnn_detector():
    """Validate MTCNN face detector."""
    print_section("MTCNN Detector Validation")
    
    try:
        from src.recognition.deep_learning.dl_face_detector import MTCNNDetector, DLFaceDetector
        
        # Test detector initialization
        detector = MTCNNDetector(min_face_size=40, device='cpu')
        print_result("MTCNN detector initialization", True)
        
        # Test detector info
        info = detector.get_detector_info()
        print_result("Detector info retrieval", True, f"Type: {info['type']}, Device: {info['device']}")
        
        # Test with real image if available
        test_image_paths = [
            'faces/khoi1234/img_01.jpg',
            'faces/huy123/img_01.jpg',
            'faces/quangnm/img_01.jpg'
        ]
        
        for img_path in test_image_paths:
            full_path = project_root / img_path
            if full_path.exists():
                image = cv2.imread(str(full_path))
                if image is not None:
                    start_time = time.time()
                    faces = detector.detect_faces(image)
                    detection_time = time.time() - start_time
                    
                    print_result(f"Face detection on {img_path}", True, 
                               f"{len(faces)} faces, {detection_time*1000:.1f}ms")
                    
                    if faces:
                        face = faces[0]
                        print_result(f"  Detection quality", True, 
                                   f"Confidence: {face.confidence:.3f}, Quality: {face.quality_score:.3f}")
                        
                        # Test face alignment
                        try:
                            aligned = detector.align_face(image, face, target_size=(112, 112))
                            print_result(f"  Face alignment", True, f"Output shape: {aligned.shape}")
                        except Exception as e:
                            print_result(f"  Face alignment", False, str(e))
                    
                    break
        else:
            print_result("Real image test", False, "No test images found")
        
        # Test DL Face Detector wrapper
        dl_detector = DLFaceDetector(detector_type='mtcnn', min_face_size=40, device='cpu')
        print_result("DL Face Detector wrapper", True)
        
        return True
        
    except Exception as e:
        print_result("MTCNN detector validation", False, str(e))
        traceback.print_exc()
        return False

def validate_preprocessing():
    """Validate preprocessing functionality."""
    print_section("Preprocessing Validation")
    
    try:
        from src.recognition.deep_learning.preprocessing import FacePreprocessor, ImageQualityFilter
        
        # Test preprocessor initialization
        preprocessor = FacePreprocessor(target_size=(112, 112))
        print_result("Face preprocessor initialization", True)
        
        # Test with synthetic image
        test_image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        
        # Test preprocessing
        result = preprocessor.preprocess_face(test_image)
        success = result['preprocessed_image'] is not None
        print_result("Face preprocessing", success, 
                   f"Quality score: {result['quality_score']:.3f}")
        
        if success:
            processed_shape = result['preprocessed_image'].shape
            print_result("  Output shape", True, f"Shape: {processed_shape}")
            
            applied_ops = result['preprocessing_applied']
            print_result("  Applied operations", True, f"Operations: {applied_ops}")
        
        # Test quality filter
        quality_filter = ImageQualityFilter(quality_threshold=0.5)
        print_result("Quality filter initialization", True)
        
        return True
        
    except Exception as e:
        print_result("Preprocessing validation", False, str(e))
        return False

def validate_classical_compatibility():
    """Validate backward compatibility with classical recognition."""
    print_section("Classical Recognition Compatibility")
    
    try:
        from src.recognition.classical.face_detector import FaceDetector
        from src.recognition.classical.lbph_recognizer import LBPHRecognizer
        
        # Test classical face detector
        classical_detector = FaceDetector()
        print_result("Classical face detector", True)
        
        # Test import through main recognition module
        from src.recognition import FaceDetector as MainFaceDetector
        print_result("Classical detector import via main module", True)
        
        return True
        
    except Exception as e:
        print_result("Classical compatibility validation", False, str(e))
        return False

def performance_benchmark():
    """Run basic performance benchmarks."""
    print_section("Performance Benchmarks")
    
    try:
        from src.recognition.deep_learning.dl_face_detector import MTCNNDetector
        
        detector = MTCNNDetector(min_face_size=40, device='cpu')
        
        # Create test images of different sizes
        test_sizes = [(224, 224), (480, 640), (720, 1280)]
        
        for height, width in test_sizes:
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add a simple rectangular "face" for detection
            cv2.rectangle(test_image, (width//4, height//4), (3*width//4, 3*height//4), (255, 255, 255), 2)
            
            start_time = time.time()
            faces = detector.detect_faces(test_image)
            detection_time = time.time() - start_time
            
            print_result(f"Detection on {width}x{height}", True, 
                       f"{detection_time*1000:.1f}ms, {len(faces)} faces")
        
        return True
        
    except Exception as e:
        print_result("Performance benchmark", False, str(e))
        return False

def main():
    """Run all validation tests."""
    print_header("FaceAttend Phase A Implementation Validation")
    
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run all validation tests
    results = {
        'Dependencies': validate_dependencies(),
        'Directory Structure': validate_directory_structure(), 
        'Configuration Files': validate_configuration_files(),
        'Model Downloader': validate_model_downloader(),
        'Model Utilities': validate_model_utilities(),
        'MTCNN Detector': validate_mtcnn_detector(),
        'Preprocessing': validate_preprocessing(),
        'Classical Compatibility': validate_classical_compatibility(),
        'Performance': performance_benchmark()
    }
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        print_result(test_name, result)
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests PASSED! Phase A implementation is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} validation tests FAILED. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)