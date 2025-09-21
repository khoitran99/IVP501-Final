#!/usr/bin/env python3
"""
Basic Recognition Factory Test - Structure and Interface Validation
Tests the recognition factory structure without requiring actual model downloads.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_factory_imports():
    """Test that factory imports work correctly"""
    print("🔧 Testing Recognition Factory Imports...")
    
    try:
        from src.recognition.recognition_factory import (
            HybridRecognitionFactory, 
            RecognitionMode, 
            RecognitionResult, 
            PerformanceMetrics,
            BaseRecognitionEngine,
            get_recognition_factory,
            CLASSICAL_AVAILABLE,
            DL_AVAILABLE
        )
        
        print(f"✅ All factory imports successful")
        print(f"   Classical available: {CLASSICAL_AVAILABLE}")
        print(f"   Deep learning available: {DL_AVAILABLE}")
        
        return True
    
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_recognition_result():
    """Test RecognitionResult data structure"""
    print("\n📊 Testing RecognitionResult Structure...")
    
    try:
        from src.recognition.recognition_factory import RecognitionResult
        
        # Test successful recognition result
        result1 = RecognitionResult(
            user_id="test_user",
            confidence=0.85,
            method="test_method",
            processing_time=0.05
        )
        
        assert result1.is_recognized == True
        print(f"✅ Successful recognition result: {result1.user_id} ({result1.confidence})")
        
        # Test failed recognition result
        result2 = RecognitionResult(
            user_id=None,
            confidence=0.2,
            method="test_method"
        )
        
        assert result2.is_recognized == False
        print(f"✅ Failed recognition result: {result2.user_id} ({result2.confidence})")
        
        return True
    
    except Exception as e:
        print(f"❌ RecognitionResult test failed: {e}")
        return False

def test_performance_metrics():
    """Test PerformanceMetrics structure"""
    print("\n📈 Testing PerformanceMetrics Structure...")
    
    try:
        from src.recognition.recognition_factory import PerformanceMetrics
        
        # Test empty metrics
        metrics = PerformanceMetrics()
        assert metrics.success_rate == 0.0
        print(f"✅ Empty metrics: {metrics.success_rate}% success rate")
        
        # Test with data
        metrics.total_recognitions = 10
        metrics.successful_recognitions = 8
        assert metrics.success_rate == 80.0
        print(f"✅ Metrics with data: {metrics.success_rate}% success rate")
        
        return True
    
    except Exception as e:
        print(f"❌ PerformanceMetrics test failed: {e}")
        return False

def test_recognition_modes():
    """Test RecognitionMode enumeration"""
    print("\n🔄 Testing RecognitionMode Enumeration...")
    
    try:
        from src.recognition.recognition_factory import RecognitionMode
        
        modes = [
            RecognitionMode.CLASSICAL,
            RecognitionMode.DEEP_LEARNING,
            RecognitionMode.HYBRID,
            RecognitionMode.AUTO
        ]
        
        for mode in modes:
            print(f"   {mode.name}: {mode.value}")
        
        print(f"✅ All recognition modes available")
        return True
    
    except Exception as e:
        print(f"❌ RecognitionMode test failed: {e}")
        return False

def test_base_engine_interface():
    """Test BaseRecognitionEngine interface"""
    print("\n🏗️  Testing BaseRecognitionEngine Interface...")
    
    try:
        from src.recognition.recognition_factory import BaseRecognitionEngine
        
        # Verify abstract methods exist
        required_methods = ['train', 'recognize', 'get_model_info', 'is_trained']
        
        for method_name in required_methods:
            assert hasattr(BaseRecognitionEngine, method_name)
            print(f"   ✅ {method_name} method defined")
        
        print(f"✅ BaseRecognitionEngine interface complete")
        return True
    
    except Exception as e:
        print(f"❌ BaseRecognitionEngine test failed: {e}")
        return False

def test_factory_creation_graceful_degradation():
    """Test factory handles missing engines gracefully"""
    print("\n🛡️  Testing Graceful Degradation...")
    
    try:
        from src.recognition.recognition_factory import (
            HybridRecognitionFactory, 
            CLASSICAL_AVAILABLE,
            DL_AVAILABLE
        )
        
        print(f"   Classical available: {CLASSICAL_AVAILABLE}")
        print(f"   Deep learning available: {DL_AVAILABLE}")
        
        if not CLASSICAL_AVAILABLE and not DL_AVAILABLE:
            print(f"   No engines available - testing error handling")
            try:
                factory = HybridRecognitionFactory()
                print(f"❌ Should have failed with no engines available")
                return False
            except Exception as e:
                print(f"✅ Correctly failed with no engines: {type(e).__name__}")
                return True
        
        # If at least one engine is available, factory should initialize
        try:
            factory = HybridRecognitionFactory()
            available_engines = factory.get_available_engines()
            print(f"✅ Factory created with engines: {available_engines}")
            
            # Test mode queries
            current_mode = factory.get_recognition_mode()
            print(f"   Current mode: {current_mode}")
            
            # Test system status
            status = factory.get_system_status()
            print(f"   System status keys: {list(status.keys())}")
            
            return True
        
        except Exception as e:
            print(f"⚠️  Factory creation with available engines failed: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Graceful degradation test failed: {e}")
        return False

def test_global_factory_pattern():
    """Test global factory pattern"""
    print("\n🌐 Testing Global Factory Pattern...")
    
    try:
        from src.recognition.recognition_factory import (
            get_recognition_factory,
            reset_recognition_factory,
            CLASSICAL_AVAILABLE,
            DL_AVAILABLE
        )
        
        # Reset factory
        reset_recognition_factory()
        
        if CLASSICAL_AVAILABLE or DL_AVAILABLE:
            # Get factory instance
            factory1 = get_recognition_factory()
            factory2 = get_recognition_factory()
            
            # Should be the same instance
            assert factory1 is factory2
            print(f"✅ Singleton pattern working correctly")
            
            # Reset and verify new instance
            reset_recognition_factory()
            factory3 = get_recognition_factory()
            
            assert factory3 is not factory1
            print(f"✅ Factory reset working correctly")
        else:
            print(f"⚠️  No engines available - cannot test global factory")
        
        return True
    
    except Exception as e:
        print(f"❌ Global factory pattern test failed: {e}")
        return False

def main():
    """Run basic recognition factory tests"""
    print("🚀 FaceAttend Recognition Factory Basic Test Suite")
    print("=" * 60)
    
    # Test summary
    test_results = {}
    
    # Test 1: Imports
    success = test_factory_imports()
    test_results['imports'] = success
    
    if not success:
        print("\n❌ Cannot continue tests - imports failed")
        return False
    
    # Test 2: Data structures
    success = test_recognition_result()
    test_results['recognition_result'] = success
    
    success = test_performance_metrics()
    test_results['performance_metrics'] = success
    
    success = test_recognition_modes()
    test_results['recognition_modes'] = success
    
    # Test 3: Interface
    success = test_base_engine_interface()
    test_results['base_interface'] = success
    
    # Test 4: Factory behavior
    success = test_factory_creation_graceful_degradation()
    test_results['graceful_degradation'] = success
    
    # Test 5: Global pattern
    success = test_global_factory_pattern()
    test_results['global_pattern'] = success
    
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
        print("🎉 All basic tests passed! Recognition factory structure is correct.")
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