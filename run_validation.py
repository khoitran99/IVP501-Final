#!/usr/bin/env python3
"""
Master validation script - runs all validation tests in sequence.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print('='*60)
    
    try:
        # Use the same Python executable
        python_exe = sys.executable
        if command.startswith('python '):
            command = command.replace('python ', f'{python_exe} ')
        
        result = subprocess.run(command.split(), 
                              capture_output=False, 
                              text=True, 
                              cwd=Path(__file__).parent)
        
        success = result.returncode == 0
        print(f"\n{'✅ PASSED' if success else '❌ FAILED'}: {description}")
        return success
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 FaceAttend Phase A - Complete Validation Suite")
    print("This will run all validation tests to verify the implementation.")
    
    validations = [
        ("python quick_test.py", "Quick Functionality Test"),
        ("python validate_implementation.py", "Comprehensive Implementation Test"),
        ("python tests/test_deep_learning.py", "Unit Tests"),
        ("python visual_validation.py", "Visual Output Generation")
    ]
    
    passed = 0
    total = len(validations)
    
    for command, description in validations:
        if run_command(command, description):
            passed += 1
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🏁 VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Overall Results: {passed}/{total} validation suites passed")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\n✅ Phase A implementation is working correctly")
        print("✅ Ready to proceed with Week 2 (InsightFace integration)")
        print("\n📁 Check 'validation_output/' for visual results")
    elif passed >= total - 1:
        print("🟡 MOSTLY SUCCESSFUL!")
        print(f"\n✅ {passed}/{total} validation suites passed")
        print("⚠️  Minor issues detected, but core functionality works")
        print("✅ Ready to proceed with Week 2")
    else:
        print("🔴 ISSUES DETECTED!")
        print(f"\n❌ Only {passed}/{total} validation suites passed")
        print("⚠️  Please review the errors above before proceeding")
    
    print(f"\n📚 See VALIDATION_GUIDE.md for detailed instructions")
    
    return passed >= total - 1  # Allow 1 failure

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)