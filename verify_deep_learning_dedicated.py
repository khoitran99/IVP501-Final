#!/usr/bin/env python3
"""
Verification script for deep learning dedicated implementation
"""

import json
import os
import sys

def verify_configuration():
    """Verify that configuration is dedicated to deep learning"""
    try:
        config_path = 'config/recognition_settings.json'
        if not os.path.exists(config_path):
            return False, "Configuration file not found"
        
        with open(config_path, 'r') as f:
            settings = json.load(f)
        
        # Check for only deep learning method
        method = settings.get('recognition_settings', {}).get('method')
        if method == 'deep_learning':
            # Check that classical/hybrid options are removed
            if 'fallback_method' in settings.get('recognition_settings', {}):
                return False, "Fallback method still present in config"
            if 'confidence_threshold' in settings.get('recognition_settings', {}):
                return False, "Classical confidence_threshold still present in config"
            return True, f"✓ Configuration dedicated to deep learning only"
        else:
            return False, f"✗ Configuration method is {method}, expected 'deep_learning'"
    
    except Exception as e:
        return False, f"Error reading configuration: {e}"

def verify_realtime_recognizer():
    """Verify that realtime recognizer is simplified for deep learning only"""
    try:
        recognizer_path = 'src/recognition/realtime_recognizer.py'
        if not os.path.exists(recognizer_path):
            return False, "Realtime recognizer file not found"
        
        with open(recognizer_path, 'r') as f:
            content = f.read()
        
        # Check for removal of classical/hybrid code
        issues = []
        
        if 'use_factory' in content:
            issues.append("use_factory flag still present")
        if 'lbph_recognizer' in content.lower():
            issues.append("LBPH recognizer references still present")
        if 'recognition_factory' in content.lower():
            issues.append("Recognition factory references still present")
        if 'confidence_threshold' in content and 'similarity_threshold' not in content:
            issues.append("Still using confidence_threshold instead of similarity_threshold")
        
        # Check for deep learning specific features
        if 'similarity_threshold' not in content:
            issues.append("Missing similarity_threshold")
        if 'SimpleInsightFaceRecognizer' not in content:
            issues.append("Missing InsightFace recognizer")
        
        if issues:
            return False, f"✗ Issues found: {', '.join(issues)}"
        else:
            return True, "✓ Realtime recognizer dedicated to deep learning"
    
    except Exception as e:
        return False, f"Error reading realtime recognizer: {e}"

def verify_main_interface():
    """Verify that main interface reflects deep learning focus"""
    try:
        main_path = 'main.py'
        if not os.path.exists(main_path):
            return False, "Main file not found"
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Check for deep learning focused messaging
        if 'Deep Learning Face Recognition System' in content:
            return True, "✓ Main interface updated for deep learning focus"
        elif 'Hybrid recognition factory' in content:
            return False, "✗ Still mentions hybrid recognition factory"
        else:
            return False, "✗ Main interface not updated for deep learning focus"
    
    except Exception as e:
        return False, f"Error reading main interface: {e}"

def verify_no_classical_imports():
    """Verify that unnecessary classical recognition imports are minimized"""
    try:
        recognizer_path = 'src/recognition/realtime_recognizer.py'
        if not os.path.exists(recognizer_path):
            return False, "Realtime recognizer file not found"
        
        with open(recognizer_path, 'r') as f:
            content = f.read()
        
        # Check that classical imports are removed
        lines = content.split('\n')
        import_lines = [line.strip() for line in lines if line.strip().startswith('from') or line.strip().startswith('import')]
        
        problematic_imports = []
        for line in import_lines:
            if 'lbph_recognizer' in line.lower():
                problematic_imports.append(line)
            if 'recognition_factory' in line.lower():
                problematic_imports.append(line)
        
        if problematic_imports:
            return False, f"✗ Unnecessary imports found: {problematic_imports}"
        else:
            return True, "✓ Classical recognition imports cleaned up"
    
    except Exception as e:
        return False, f"Error checking imports: {e}"

def main():
    """Main verification function"""
    print("Deep Learning Dedicated Project Verification")
    print("=" * 50)
    
    checks = [
        ("Configuration", verify_configuration),
        ("Realtime Recognizer", verify_realtime_recognizer),
        ("Main Interface", verify_main_interface),
        ("Import Cleanup", verify_no_classical_imports)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        success, message = check_func()
        print(f"{check_name}: {message}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ SUCCESS: Project is now fully dedicated to deep learning!")
        print("\nDeep Learning Features:")
        print("  • InsightFace ArcFace recognition engine")
        print("  • Similarity-based matching (0.0-1.0 range)")
        print("  • Embedding-based face recognition")
        print("  • No classical/hybrid fallbacks")
        print("  • Streamlined architecture")
        print("  • Pure deep learning workflow")
        return 0
    else:
        print("✗ ISSUES FOUND: Project not fully dedicated to deep learning.")
        print("Please address the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())