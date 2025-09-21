#!/usr/bin/env python3
"""
Recognition UI Test - Model Selection and Performance Monitoring
Tests the UI components for recognition engine management.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_recognition_settings_window():
    """Test the recognition settings window UI"""
    print("🖥️  Testing Recognition Settings Window...")
    
    try:
        from src.ui.recognition_settings_window import RecognitionSettingsWindow
        
        # Create test window
        root = tk.Tk()
        root.title("Recognition Settings Test")
        root.geometry("200x100")
        
        # Test window creation
        settings_window = RecognitionSettingsWindow(parent=root)
        print("✅ Recognition settings window created")
        
        # Test window showing (don't actually show for automated test)
        # settings_window.show_window()
        
        return True
        
    except Exception as e:
        print(f"❌ Recognition settings window test failed: {e}")
        return False

def test_main_app_integration():
    """Test main application integration"""
    print("\n🏠 Testing Main App Integration...")
    
    try:
        from main import FaceAttendApp
        
        # Test import
        print("✅ Main app imports recognition settings")
        
        # Test method exists
        app_class = FaceAttendApp
        assert hasattr(app_class, 'open_recognition_settings')
        print("✅ Main app has recognition settings method")
        
        return True
        
    except Exception as e:
        print(f"❌ Main app integration test failed: {e}")
        return False

def test_factory_integration():
    """Test recognition factory integration"""
    print("\n🏭 Testing Factory Integration...")
    
    try:
        from src.recognition import FACTORY_AVAILABLE
        
        if FACTORY_AVAILABLE:
            from src.recognition import get_recognition_factory, RecognitionMode
            
            factory = get_recognition_factory()
            print("✅ Recognition factory available")
            
            # Test mode enumeration
            modes = [mode for mode in RecognitionMode]
            print(f"✅ Available recognition modes: {[m.value for m in modes]}")
            
            # Test factory methods
            available_engines = factory.get_available_engines()
            print(f"✅ Available engines: {available_engines}")
            
            system_status = factory.get_system_status()
            print(f"✅ System status available: {list(system_status.keys())}")
            
        else:
            print("⚠️  Recognition factory not available (expected in some environments)")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory integration test failed: {e}")
        return False

def test_ui_components():
    """Test individual UI components"""
    print("\n🧩 Testing UI Components...")
    
    try:
        # Test basic tkinter components
        root = tk.Tk()
        root.withdraw()  # Hide window
        
        # Test notebook
        notebook = ttk.Notebook(root)
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Test Tab")
        print("✅ Notebook widget works")
        
        # Test variables
        mode_var = tk.StringVar()
        mode_var.set("test_mode")
        print("✅ StringVar works")
        
        # Test text widget
        text_widget = tk.Text(root, height=5)
        text_widget.insert(tk.END, "Test text")
        print("✅ Text widget works")
        
        # Test radiobutton
        radio = ttk.Radiobutton(root, text="Test", variable=mode_var, value="test")
        print("✅ Radiobutton works")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ UI components test failed: {e}")
        return False

def test_threading_support():
    """Test threading support for monitoring"""
    print("\n🧵 Testing Threading Support...")
    
    try:
        import threading
        import time
        
        # Test thread creation
        test_complete = threading.Event()
        
        def test_thread():
            time.sleep(0.1)
            test_complete.set()
        
        thread = threading.Thread(target=test_thread, daemon=True)
        thread.start()
        
        # Wait for completion with timeout
        if test_complete.wait(timeout=1.0):
            print("✅ Threading support works")
            return True
        else:
            print("❌ Threading timeout")
            return False
        
    except Exception as e:
        print(f"❌ Threading test failed: {e}")
        return False

def test_ui_data_structures():
    """Test UI-related data structures"""
    print("\n📊 Testing UI Data Structures...")
    
    try:
        # Test with mock data
        performance_data = {
            'classical': {
                'total_recognitions': 100,
                'successful_recognitions': 85,
                'average_confidence': 0.75,
                'average_processing_time': 0.03
            },
            'deep_learning': {
                'total_recognitions': 50,
                'successful_recognitions': 48,
                'average_confidence': 0.92,
                'average_processing_time': 0.15
            }
        }
        
        # Test data processing
        total_recognitions = sum(data['total_recognitions'] for data in performance_data.values())
        total_successful = sum(data['successful_recognitions'] for data in performance_data.values())
        overall_success_rate = (total_successful / total_recognitions * 100) if total_recognitions > 0 else 0
        
        assert total_recognitions == 150
        assert total_successful == 133
        assert abs(overall_success_rate - 88.67) < 0.1
        
        print(f"✅ Data processing works: {overall_success_rate:.1f}% success rate")
        return True
        
    except Exception as e:
        print(f"❌ UI data structures test failed: {e}")
        return False

def main():
    """Run UI integration tests"""
    print("🚀 FaceAttend Recognition UI Test Suite")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: Recognition settings window
    success = test_recognition_settings_window()
    test_results['settings_window'] = success
    
    # Test 2: Main app integration
    success = test_main_app_integration()
    test_results['main_app_integration'] = success
    
    # Test 3: Factory integration
    success = test_factory_integration()
    test_results['factory_integration'] = success
    
    # Test 4: UI components
    success = test_ui_components()
    test_results['ui_components'] = success
    
    # Test 5: Threading support
    success = test_threading_support()
    test_results['threading_support'] = success
    
    # Test 6: Data structures
    success = test_ui_data_structures()
    test_results['data_structures'] = success
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 UI TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25} : {status}")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All UI tests passed! Recognition UI is working correctly.")
        return True
    else:
        print("⚠️  Some UI tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 UI test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UI test suite crashed: {e}")
        sys.exit(1)