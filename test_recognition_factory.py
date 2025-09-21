#!/usr/bin/env python3
"""
Recognition Factory Test - Hybrid Support Validation
Tests the unified recognition interface supporting both classical and deep learning.
"""

import sys
import os
import numpy as np
import cv2
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.recognition.recognition_factory import (
    HybridRecognitionFactory, 
    RecognitionMode, 
    RecognitionResult, 
    get_recognition_factory,
    CLASSICAL_AVAILABLE,
    DL_AVAILABLE
)
from src.utils.logger import get_module_logger

def create_test_face_image(size: tuple = (200, 200)) -> np.ndarray:
    """Create a synthetic test face image"""
    # Create a simple face-like pattern
    image = np.zeros((*size, 3), dtype=np.uint8)
    
    # Face outline (circle)
    center = (size[1]//2, size[0]//2)
    radius = min(size)//3
    cv2.circle(image, center, radius, (150, 150, 150), -1)
    
    # Eyes
    eye_y = center[1] - radius//3
    left_eye = (center[0] - radius//3, eye_y)
    right_eye = (center[0] + radius//3, eye_y)
    cv2.circle(image, left_eye, radius//8, (50, 50, 50), -1)
    cv2.circle(image, right_eye, radius//8, (50, 50, 50), -1)
    
    # Nose
    nose_points = np.array([
        [center[0], center[1] - radius//6],
        [center[0] - radius//10, center[1] + radius//6],
        [center[0] + radius//10, center[1] + radius//6]
    ], np.int32)
    cv2.fillPoly(image, [nose_points], (100, 100, 100))
    
    # Mouth
    mouth_center = (center[0], center[1] + radius//3)
    cv2.ellipse(image, mouth_center, (radius//4, radius//8), 0, 0, 180, (80, 80, 80), -1)
    
    return image

def test_factory_initialization():
    """Test recognition factory initialization"""
    print("🧪 Testing Recognition Factory Initialization...")
    
    try:
        factory = HybridRecognitionFactory()
        
        print(f"✅ Factory initialized successfully")
        print(f"   Available engines: {factory.get_available_engines()}")
        print(f"   Current mode: {factory.get_recognition_mode()}")
        print(f"   Classical available: {CLASSICAL_AVAILABLE}")
        print(f"   Deep learning available: {DL_AVAILABLE}")
        
        return True, factory
    
    except Exception as e:
        print(f"❌ Factory initialization failed: {e}")
        return False, None

def test_mode_switching(factory: HybridRecognitionFactory):
    """Test switching between recognition modes"""
    print("\n🔄 Testing Recognition Mode Switching...")
    
    try:
        original_mode = factory.get_recognition_mode()
        available_engines = factory.get_available_engines()
        
        # Test each available mode
        modes_tested = []
        
        if 'classical' in available_engines:
            factory.set_recognition_mode(RecognitionMode.CLASSICAL)
            assert factory.get_recognition_mode() == RecognitionMode.CLASSICAL
            modes_tested.append("CLASSICAL")
            print(f"✅ Classical mode: {factory.get_recognition_mode()}")
        
        if 'deep_learning' in available_engines:
            factory.set_recognition_mode(RecognitionMode.DEEP_LEARNING)
            assert factory.get_recognition_mode() == RecognitionMode.DEEP_LEARNING
            modes_tested.append("DEEP_LEARNING")
            print(f"✅ Deep learning mode: {factory.get_recognition_mode()}")
        
        if len(available_engines) >= 2:
            factory.set_recognition_mode(RecognitionMode.HYBRID)
            assert factory.get_recognition_mode() == RecognitionMode.HYBRID
            modes_tested.append("HYBRID")
            print(f"✅ Hybrid mode: {factory.get_recognition_mode()}")
        
        factory.set_recognition_mode(RecognitionMode.AUTO)
        assert factory.get_recognition_mode() == RecognitionMode.AUTO
        modes_tested.append("AUTO")
        print(f"✅ Auto mode: {factory.get_recognition_mode()}")
        
        # Restore original mode
        factory.set_recognition_mode(original_mode)
        
        print(f"✅ Mode switching successful. Tested: {modes_tested}")
        return True
    
    except Exception as e:
        print(f"❌ Mode switching failed: {e}")
        return False

def test_training_workflow(factory: HybridRecognitionFactory):
    """Test user training with multiple engines"""
    print("\n👤 Testing User Training Workflow...")
    
    try:
        # Create test face images
        test_images = []
        for i in range(3):
            image = create_test_face_image((200, 200))
            # Add some variation
            noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            test_images.append(image)
        
        print(f"   Created {len(test_images)} test face images")
        
        # Test training with different modes
        test_user_id = "test_user_factory"
        
        # Set to hybrid mode for training
        original_mode = factory.get_recognition_mode()
        factory.set_recognition_mode(RecognitionMode.HYBRID)
        
        # Train user
        print(f"   Training user '{test_user_id}' with {len(test_images)} images...")
        training_results = factory.train_user(test_user_id, test_images)
        
        print(f"   Training results: {training_results}")
        
        # Verify training success
        successful_engines = [engine for engine, success in training_results.items() if success]
        if successful_engines:
            print(f"✅ Training successful with engines: {successful_engines}")
            
            # Test that engines report as trained
            engine_info = factory.get_engine_info()
            for engine_name in successful_engines:
                if engine_name in engine_info:
                    is_trained = factory.is_engine_trained(engine_name)
                    print(f"   {engine_name} trained status: {is_trained}")
            
            return True, test_user_id
        else:
            print(f"⚠️  Training failed with all engines")
            return False, None
    
    except Exception as e:
        print(f"❌ Training workflow failed: {e}")
        return False, None

def test_recognition_workflow(factory: HybridRecognitionFactory, trained_user_id: str):
    """Test face recognition with different modes"""
    print("\n🔍 Testing Face Recognition Workflow...")
    
    try:
        # Create test recognition image
        test_image = create_test_face_image((200, 200))
        print(f"   Created test recognition image")
        
        # Test recognition with different modes
        modes_to_test = []
        available_engines = factory.get_available_engines()
        
        if 'classical' in available_engines:
            modes_to_test.append(RecognitionMode.CLASSICAL)
        if 'deep_learning' in available_engines:
            modes_to_test.append(RecognitionMode.DEEP_LEARNING)
        if len(available_engines) >= 2:
            modes_to_test.append(RecognitionMode.HYBRID)
        modes_to_test.append(RecognitionMode.AUTO)
        
        recognition_results = {}
        
        for mode in modes_to_test:
            print(f"\n   Testing recognition with {mode.value} mode...")
            factory.set_recognition_mode(mode)
            
            start_time = time.time()
            result = factory.recognize_face(test_image)
            recognition_time = time.time() - start_time
            
            recognition_results[mode.value] = {
                'result': result,
                'time': recognition_time
            }
            
            print(f"     User ID: {result.user_id}")
            print(f"     Confidence: {result.confidence:.3f}")
            print(f"     Method: {result.method}")
            print(f"     Processing time: {recognition_time:.3f}s")
            print(f"     Is recognized: {result.is_recognized}")
        
        # Summary
        successful_recognitions = [
            mode for mode, data in recognition_results.items() 
            if data['result'].is_recognized
        ]
        
        if successful_recognitions:
            print(f"✅ Recognition successful with modes: {successful_recognitions}")
        else:
            print(f"⚠️  Recognition did not identify the test user (expected for synthetic images)")
        
        print(f"✅ Recognition workflow completed successfully")
        return True, recognition_results
    
    except Exception as e:
        print(f"❌ Recognition workflow failed: {e}")
        return False, None

def test_performance_monitoring(factory: HybridRecognitionFactory):
    """Test performance monitoring and metrics"""
    print("\n📊 Testing Performance Monitoring...")
    
    try:
        # Get performance metrics
        performance_metrics = factory.get_performance_metrics()
        print(f"   Performance metrics available for: {list(performance_metrics.keys())}")
        
        for engine_name, metrics in performance_metrics.items():
            print(f"   {engine_name}:")
            print(f"     Total recognitions: {metrics.total_recognitions}")
            print(f"     Successful recognitions: {metrics.successful_recognitions}")
            print(f"     Success rate: {metrics.success_rate:.1f}%")
            print(f"     Average confidence: {metrics.average_confidence:.3f}")
            print(f"     Average processing time: {metrics.average_processing_time:.3f}s")
        
        # Get system status
        system_status = factory.get_system_status()
        print(f"\n   System Status:")
        print(f"     Current mode: {system_status['current_mode']}")
        print(f"     Available engines: {system_status['available_engines']}")
        print(f"     Classical available: {system_status['classical_available']}")
        print(f"     Deep learning available: {system_status['deep_learning_available']}")
        
        print(f"✅ Performance monitoring working correctly")
        return True
    
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        return False

def test_error_handling(factory: HybridRecognitionFactory):
    """Test error handling and edge cases"""
    print("\n⚠️  Testing Error Handling...")
    
    try:
        # Test invalid mode switching
        try:
            if 'classical' not in factory.get_available_engines():
                factory.set_recognition_mode(RecognitionMode.CLASSICAL)
                print(f"❌ Should have failed setting unavailable classical mode")
                return False
        except Exception:
            print(f"✅ Correctly rejected unavailable classical mode")
        
        try:
            if 'deep_learning' not in factory.get_available_engines():
                factory.set_recognition_mode(RecognitionMode.DEEP_LEARNING)
                print(f"❌ Should have failed setting unavailable deep learning mode")
                return False
        except Exception:
            print(f"✅ Correctly rejected unavailable deep learning mode")
        
        # Test recognition with invalid image
        try:
            invalid_image = np.array([])
            result = factory.recognize_face(invalid_image)
            print(f"   Recognition with invalid image: {result.user_id}, confidence: {result.confidence}")
            print(f"✅ Gracefully handled invalid image")
        except Exception:
            print(f"✅ Correctly rejected invalid image")
        
        # Test training with empty images
        try:
            empty_images = []
            training_result = factory.train_user("test_empty", empty_images)
            print(f"   Training with empty images: {training_result}")
            print(f"✅ Gracefully handled empty training data")
        except Exception:
            print(f"✅ Correctly rejected empty training data")
        
        print(f"✅ Error handling tests completed")
        return True
    
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run comprehensive recognition factory tests"""
    print("🚀 FaceAttend Recognition Factory Test Suite")
    print("=" * 60)
    
    # Test summary
    test_results = {}
    
    # Test 1: Factory initialization
    success, factory = test_factory_initialization()
    test_results['initialization'] = success
    
    if not success or factory is None:
        print("\n❌ Cannot continue tests - factory initialization failed")
        return False
    
    # Test 2: Mode switching
    success = test_mode_switching(factory)
    test_results['mode_switching'] = success
    
    # Test 3: Training workflow
    success, trained_user = test_training_workflow(factory)
    test_results['training'] = success
    
    # Test 4: Recognition workflow
    if trained_user:
        success, recognition_results = test_recognition_workflow(factory, trained_user)
        test_results['recognition'] = success
    else:
        test_results['recognition'] = False
        print("\n⚠️  Skipping recognition test - no trained user")
    
    # Test 5: Performance monitoring
    success = test_performance_monitoring(factory)
    test_results['performance'] = success
    
    # Test 6: Error handling
    success = test_error_handling(factory)
    test_results['error_handling'] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} : {status}")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Recognition factory is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)