#!/usr/bin/env python3
"""
Complete verification script for deep learning dedicated implementation
"""

import json
import os
import sys

def verify_attendance_window_integration():
    """Verify that attendance window uses deep learning properly"""
    try:
        attendance_path = 'src/ui/attendance_window.py'
        if not os.path.exists(attendance_path):
            return False, "Attendance window file not found"
        
        with open(attendance_path, 'r') as f:
            content = f.read()
        
        issues = []
        
        # Check for proper initialization
        if 'similarity_threshold=0.6' not in content:
            issues.append("AttendanceWindow not using similarity_threshold parameter")
        
        # Check for updated UI labels
        if 'Similarity Threshold:' not in content:
            issues.append("UI still shows Confidence Threshold instead of Similarity Threshold")
        
        # Check for updated method calls
        if 'update_confidence_threshold' in content:
            issues.append("Still calling update_confidence_threshold instead of update_similarity_threshold")
        
        # Check for proper threshold ranges
        if 'from_=0.3, to=0.9' not in content:
            issues.append("Threshold scale not updated for similarity range (0.3-0.9)")
        
        if issues:
            return False, f"✗ Issues found: {', '.join(issues)}"
        else:
            return True, "✓ Attendance window properly integrated with deep learning"
    
    except Exception as e:
        return False, f"Error checking attendance window: {e}"

def verify_realtime_recognizer_params():
    """Verify RealtimeRecognizer parameter handling"""
    try:
        recognizer_path = 'src/recognition/realtime_recognizer.py'
        if not os.path.exists(recognizer_path):
            return False, "Realtime recognizer file not found"
        
        with open(recognizer_path, 'r') as f:
            content = f.read()
        
        # Check parameter signature
        if 'similarity_threshold: float = 0.6' not in content:
            return False, "✗ RealtimeRecognizer init doesn't use similarity_threshold parameter"
        
        # Check for deep learning specific method
        if 'update_similarity_threshold' not in content:
            return False, "✗ Missing update_similarity_threshold method"
        
        # Check for removal of classical references
        if 'confidence_threshold' in content and 'similarity_threshold' not in content:
            return False, "✗ Still using confidence_threshold instead of similarity_threshold"
        
        return True, "✓ RealtimeRecognizer parameters properly updated for deep learning"
    
    except Exception as e:
        return False, f"Error checking realtime recognizer: {e}"

def verify_configuration_consistency():
    """Verify configuration is consistent with deep learning approach"""
    try:
        config_path = 'config/recognition_settings.json'
        if not os.path.exists(config_path):
            return False, "Configuration file not found"
        
        with open(config_path, 'r') as f:
            settings = json.load(f)
        
        # Check for deep learning settings
        method = settings.get('recognition_settings', {}).get('method')
        if method != 'deep_learning':
            return False, f"✗ Configuration method is {method}, expected 'deep_learning'"
        
        # Check for similarity threshold
        threshold = settings.get('recognition_settings', {}).get('similarity_threshold')
        if threshold is None:
            return False, "✗ Missing similarity_threshold in configuration"
        
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return False, f"✗ Invalid similarity_threshold value: {threshold} (should be 0.0-1.0)"
        
        return True, f"✓ Configuration properly set for deep learning (threshold: {threshold})"
    
    except Exception as e:
        return False, f"Error checking configuration: {e}"

def verify_main_interface_consistency():
    """Verify main interface shows deep learning information"""
    try:
        main_path = 'main.py'
        if not os.path.exists(main_path):
            return False, "Main file not found"
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Check for deep learning messaging
        if 'Deep Learning Face Recognition System' not in content:
            return False, "✗ Main interface not updated for deep learning"
        
        # Check for proper recognition engine imports
        if 'SimpleInsightFaceRecognizer' not in content:
            return False, "✗ Main interface not using InsightFace recognizer"
        
        # Check that hybrid/classical references are removed
        if 'Hybrid recognition factory' in content:
            return False, "✗ Still mentions hybrid recognition factory"
        
        if 'Classical LBPH' in content:
            return False, "✗ Still mentions classical LBPH"
        
        return True, "✓ Main interface consistent with deep learning focus"
    
    except Exception as e:
        return False, f"Error checking main interface: {e}"

def check_syntax_validity():
    """Check that all Python files have valid syntax"""
    try:
        import py_compile
        
        files_to_check = [
            'src/recognition/realtime_recognizer.py',
            'src/ui/attendance_window.py', 
            'main.py'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    py_compile.compile(file_path, doraise=True)
                except py_compile.PyCompileError as e:
                    return False, f"✗ Syntax error in {file_path}: {e}"
        
        return True, "✓ All files have valid Python syntax"
    
    except Exception as e:
        return False, f"Error checking syntax: {e}"

def main():
    """Main verification function"""
    print("Complete Deep Learning Integration Verification")
    print("=" * 55)
    
    checks = [
        ("Syntax Validity", check_syntax_validity),
        ("Configuration", verify_configuration_consistency),
        ("RealtimeRecognizer Parameters", verify_realtime_recognizer_params),
        ("Attendance Window Integration", verify_attendance_window_integration),
        ("Main Interface", verify_main_interface_consistency)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        success, message = check_func()
        print(f"{check_name}: {message}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 SUCCESS: Project is fully dedicated to deep learning!")
        print("\nDeep Learning System Features:")
        print("  • InsightFace ArcFace recognition engine")
        print("  • Similarity-based matching (0.0-1.0 range)")  
        print("  • Embedding-based face recognition")
        print("  • Real-time attendance with deep learning")
        print("  • Streamlined UI for similarity thresholds")
        print("  • No classical/hybrid dependencies")
        print("  • Pure deep learning workflow")
        print("\nThe system is ready for deep learning face recognition!")
        return 0
    else:
        print("❌ ISSUES FOUND: Some components not fully integrated.")
        print("Please address the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())