#!/usr/bin/env python3
"""
Quick validation script to test core functionality.
Run this for a fast check of the implementation.
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports work."""
    print("Testing basic imports...")
    
    try:
        # Test model downloader
        from models.model_downloader import ModelDownloader
        downloader = ModelDownloader()
        print("✅ Model downloader works")
        
        # Test MTCNN detector
        from src.recognition.deep_learning.dl_face_detector import MTCNNDetector
        detector = MTCNNDetector(device='cpu')
        print("✅ MTCNN detector works")
        
        # Test preprocessing
        from src.recognition.deep_learning.preprocessing import FacePreprocessor
        preprocessor = FacePreprocessor()
        print("✅ Preprocessing works")
        
        # Test model utilities
        from src.utils.model_utils import get_model_loader
        loader = get_model_loader()
        print(f"✅ Model utilities work (device: {loader.device})")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_face_detection():
    """Test face detection on an existing image."""
    print("\nTesting face detection...")
    
    try:
        import cv2
        import numpy as np
        from src.recognition.deep_learning.dl_face_detector import MTCNNDetector
        
        detector = MTCNNDetector(device='cpu')
        
        # Try to find an existing face image
        test_paths = [
            'faces/khoi1234/img_01.jpg',
            'faces/huy123/img_01.jpg', 
            'faces/quangnm/img_01.jpg'
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                image = cv2.imread(path)
                if image is not None:
                    faces = detector.detect_faces(image)
                    print(f"✅ Detected {len(faces)} faces in {path}")
                    if faces:
                        face = faces[0]
                        print(f"   Confidence: {face.confidence:.3f}, Quality: {face.quality_score:.3f}")
                    return True
        
        # If no real images, test with synthetic
        print("No test images found, testing with synthetic image...")
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        faces = detector.detect_faces(test_image)
        print(f"✅ Synthetic test completed, detected {len(faces)} faces")
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        import json
        
        # Test YAML config
        with open('config/models.yaml', 'r') as f:
            models_config = yaml.safe_load(f)
        print(f"✅ Models config loaded: {len(models_config['models'])} categories")
        
        # Test JSON config  
        with open('config/recognition_settings.json', 'r') as f:
            settings = json.load(f)
        print(f"✅ Recognition settings loaded: {settings['recognition_settings']['primary_method']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run quick tests."""
    print("🔍 Quick Validation Test for FaceAttend Phase A\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Face Detection", test_face_detection), 
        ("Configuration", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All quick tests PASSED!")
        print("\nYou can now run the full validation with:")
        print("python validate_implementation.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()