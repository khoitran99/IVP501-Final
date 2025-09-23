#!/usr/bin/env python3
"""
Diagnostic script for face recognition issues
"""

import os
import sys
import json
from pathlib import Path

def check_configuration():
    """Check recognition configuration"""
    print("🔍 CHECKING CONFIGURATION")
    print("-" * 40)
    
    try:
        config_path = 'config/recognition_settings.json'
        if not os.path.exists(config_path):
            print("❌ Configuration file missing")
            return False
        
        with open(config_path, 'r') as f:
            settings = json.load(f)
        
        threshold = settings.get('recognition_settings', {}).get('similarity_threshold', 0.6)
        print(f"✅ Similarity threshold: {threshold}")
        
        if threshold > 0.7:
            print(f"⚠️  WARNING: Threshold {threshold} might be too high!")
            print("   Try lowering to 0.5-0.6 for better recognition")
        elif threshold < 0.4:
            print(f"⚠️  WARNING: Threshold {threshold} might be too low!")
            print("   Try raising to 0.5-0.6 for better security")
        else:
            print("✅ Threshold looks reasonable")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_registered_users():
    """Check registered users and their data"""
    print("\n🔍 CHECKING REGISTERED USERS")
    print("-" * 40)
    
    try:
        faces_dir = Path('faces')
        if not faces_dir.exists():
            print("❌ Faces directory doesn't exist")
            return False
        
        # Check for user directories
        user_dirs = [d for d in faces_dir.iterdir() if d.is_dir() and d.name != 'metadata']
        
        if not user_dirs:
            print("❌ No users registered!")
            print("   Please register at least one user first")
            return False
        
        print(f"✅ Found {len(user_dirs)} registered user(s):")
        
        total_images = 0
        for user_dir in user_dirs:
            user_id = user_dir.name
            images = list(user_dir.glob('*.jpg')) + list(user_dir.glob('*.png'))
            total_images += len(images)
            print(f"   • {user_id}: {len(images)} images")
            
            if len(images) < 3:
                print(f"     ⚠️  WARNING: Only {len(images)} images for {user_id}")
                print("     Consider registering more images for better recognition")
        
        print(f"✅ Total images: {total_images}")
        return True
        
    except Exception as e:
        print(f"❌ User data error: {e}")
        return False

def check_model_training():
    """Check if the model is trained with embeddings"""
    print("\n🔍 CHECKING MODEL TRAINING")
    print("-" * 40)
    
    try:
        embeddings_file = Path('faces/metadata/insightface_embeddings.pkl')
        
        if not embeddings_file.exists():
            print("❌ No embeddings file found!")
            print("   The deep learning model hasn't generated embeddings yet")
            print("   Solution: Try retraining the model or restart recognition")
            return False
        
        # Check file size
        file_size = embeddings_file.stat().st_size
        print(f"✅ Embeddings file exists ({file_size} bytes)")
        
        if file_size < 100:
            print("⚠️  WARNING: Embeddings file is very small")
            print("   This might indicate training issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("\n🔍 CHECKING DEPENDENCIES")
    print("-" * 40)
    
    dependencies = {
        'cv2': 'OpenCV',
        'insightface': 'InsightFace', 
        'numpy': 'NumPy',
        'PIL': 'Pillow'
    }
    
    all_available = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} missing - install with: pip install {module}")
            all_available = False
    
    return all_available

def provide_solutions():
    """Provide troubleshooting solutions"""
    print("\n🔧 TROUBLESHOOTING SOLUTIONS")
    print("-" * 40)
    
    solutions = [
        "1. LOWER SIMILARITY THRESHOLD:",
        "   • Open attendance window",
        "   • Move similarity slider to 0.4-0.5",
        "   • Try recognition again",
        "",
        "2. RETRAIN THE MODEL:",
        "   • In attendance window, click 'Retrain Model'",
        "   • Wait for training to complete",
        "   • Try recognition again",
        "",
        "3. RE-REGISTER WITH MORE IMAGES:",
        "   • Register 5-10 different face angles",
        "   • Include different lighting conditions",
        "   • Ensure face is clear and well-lit",
        "",
        "4. CHECK LIGHTING CONDITIONS:",
        "   • Ensure good lighting when recognizing",
        "   • Avoid backlighting or shadows",
        "   • Use consistent lighting as registration",
        "",
        "5. FACE POSITIONING:",
        "   • Look directly at camera",
        "   • Keep face centered in frame",
        "   • Maintain similar distance as registration",
        "",
        "6. RESTART THE APPLICATION:",
        "   • Close all windows",
        "   • Restart the application",
        "   • Try recognition again"
    ]
    
    for solution in solutions:
        print(solution)

def create_test_script():
    """Create a test script for manual verification"""
    test_script = '''#!/usr/bin/env python3
"""
Manual recognition test script
Run this to test recognition with debug output
"""

import sys
import os
sys.path.append('.')

try:
    from src.recognition.simple_insightface_recognizer import SimpleInsightFaceRecognizer
    
    print("Initializing recognizer...")
    recognizer = SimpleInsightFaceRecognizer(confidence_threshold=0.4)  # Lower threshold for testing
    
    model_info = recognizer.get_model_info()
    print(f"Model info: {model_info}")
    
    if model_info.get('total_users', 0) == 0:
        print("❌ No users trained! Please register a user first.")
    else:
        print(f"✅ {model_info.get('total_users', 0)} users trained")
        print(f"✅ {model_info.get('total_embeddings', 0)} embeddings generated")
        print("Recognition system ready for testing")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check the diagnosis output above")
'''
    
    with open('test_recognition.py', 'w') as f:
        f.write(test_script)
    
    print(f"\n📝 Created 'test_recognition.py' for manual testing")

def main():
    """Main diagnostic function"""
    print("🔍 FACE RECOGNITION DIAGNOSTIC TOOL")
    print("=" * 50)
    
    checks = [
        check_dependencies,
        check_configuration, 
        check_registered_users,
        check_model_training
    ]
    
    all_passed = True
    
    for check in checks:
        if not check():
            all_passed = False
    
    provide_solutions()
    create_test_script()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All checks passed! Recognition should work.")
        print("If still not working, try solutions above.")
    else:
        print("❌ Issues found! Please address them using solutions above.")
    
    print("\nNext steps:")
    print("1. Run: python test_recognition.py")
    print("2. Try lowering similarity threshold to 0.4-0.5")
    print("3. Retrain model if needed")

if __name__ == "__main__":
    main()