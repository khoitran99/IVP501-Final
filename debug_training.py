#!/usr/bin/env python3
"""
Debug training issues and provide solutions
"""

import os
import sys
import traceback
from pathlib import Path

def test_basic_imports():
    """Test if we can import required modules"""
    print("🔍 TESTING BASIC IMPORTS")
    print("-" * 40)
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import insightface
        print(f"✅ InsightFace imported successfully")
    except ImportError as e:
        print(f"❌ InsightFace import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_insightface_initialization():
    """Test InsightFace initialization"""
    print("\n🔍 TESTING INSIGHTFACE INITIALIZATION")
    print("-" * 40)
    
    try:
        import insightface
        print("Creating FaceAnalysis app...")
        
        # Try with CPU provider only
        app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        print("✅ FaceAnalysis created with CPU provider")
        
        print("Preparing context...")
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ InsightFace initialization failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def test_simple_recognizer():
    """Test the simple recognizer initialization"""
    print("\n🔍 TESTING SIMPLE RECOGNIZER")
    print("-" * 40)
    
    try:
        sys.path.append('.')
        from src.recognition.simple_insightface_recognizer import SimpleInsightFaceRecognizer
        
        print("Creating SimpleInsightFaceRecognizer...")
        recognizer = SimpleInsightFaceRecognizer(confidence_threshold=0.5)
        print("✅ SimpleInsightFaceRecognizer created successfully")
        
        model_info = recognizer.get_model_info()
        print(f"Model info: {model_info}")
        
        return True, recognizer
        
    except Exception as e:
        print(f"❌ SimpleInsightFaceRecognizer failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False, None

def test_manual_training():
    """Test manual training with a single user"""
    print("\n🔍 TESTING MANUAL TRAINING")
    print("-" * 40)
    
    success, recognizer = test_simple_recognizer()
    if not success:
        return False
    
    try:
        # Get first user for testing
        faces_dir = Path('faces')
        user_dirs = [d for d in faces_dir.iterdir() if d.is_dir() and d.name != 'metadata']
        
        if not user_dirs:
            print("❌ No users found for training")
            return False
        
        # Test with first user
        test_user = user_dirs[0].name
        print(f"Testing training with user: {test_user}")
        
        # Get user images
        from src.storage.face_storage import FaceStorage
        face_storage = FaceStorage()
        user_images = face_storage.get_user_face_images(test_user)
        
        if not user_images:
            print(f"❌ No images found for user {test_user}")
            return False
        
        print(f"Found {len(user_images)} images for {test_user}")
        
        # Try training one user
        print("Attempting to train user...")
        success = recognizer.train_user(test_user, user_images)
        
        if success:
            print(f"✅ Successfully trained user {test_user}")
            return True
        else:
            print(f"❌ Failed to train user {test_user}")
            return False
            
    except Exception as e:
        print(f"❌ Manual training failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def create_training_workaround():
    """Create a workaround training script"""
    print("\n🔧 CREATING TRAINING WORKAROUND")
    print("-" * 40)
    
    workaround_script = '''#!/usr/bin/env python3
"""
Manual training workaround script
Run this if normal retraining fails
"""

import sys
import os
sys.path.append('.')

def manual_train():
    try:
        print("Starting manual training...")
        
        # Import required modules
        from src.recognition.simple_insightface_recognizer import SimpleInsightFaceRecognizer
        from src.storage.face_storage import FaceStorage
        
        # Create recognizer with lower threshold
        print("Creating recognizer...")
        recognizer = SimpleInsightFaceRecognizer(confidence_threshold=0.4)
        
        # Get face storage
        face_storage = FaceStorage()
        users = face_storage.list_users()
        
        if not users:
            print("❌ No users found! Please register a user first.")
            return False
        
        print(f"Found {len(users)} users: {users}")
        
        # Train each user individually
        trained_count = 0
        for user_id in users:
            print(f"\\nTraining user: {user_id}")
            
            user_images = face_storage.get_user_face_images(user_id)
            if not user_images:
                print(f"  ⚠️  No images for {user_id}")
                continue
            
            print(f"  Found {len(user_images)} images")
            
            # Train this user
            success = recognizer.train_user(user_id, user_images)
            if success:
                print(f"  ✅ Successfully trained {user_id}")
                trained_count += 1
            else:
                print(f"  ❌ Failed to train {user_id}")
        
        if trained_count > 0:
            print(f"\\n🎉 Successfully trained {trained_count}/{len(users)} users!")
            print("You can now try face recognition.")
            return True
        else:
            print("\\n❌ No users could be trained.")
            return False
            
    except Exception as e:
        print(f"❌ Manual training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 MANUAL TRAINING WORKAROUND")
    print("=" * 40)
    manual_train()
'''
    
    with open('manual_train.py', 'w') as f:
        f.write(workaround_script)
    
    print("✅ Created 'manual_train.py' workaround script")

def provide_dependency_fixes():
    """Provide dependency installation commands"""
    print("\n🔧 DEPENDENCY INSTALLATION FIXES")
    print("-" * 40)
    
    commands = [
        "# Try these commands one by one:",
        "",
        "# 1. Install with conda (recommended):",
        "conda install -c conda-forge opencv",
        "pip install insightface onnxruntime",
        "",
        "# 2. Or install with pip:",
        "pip uninstall opencv-python",
        "pip install opencv-python==4.8.1.78",
        "pip install insightface==0.7.3",
        "pip install onnxruntime==1.16.3",
        "",
        "# 3. Alternative approach:",
        "pip install --upgrade pip",
        "pip install --no-cache-dir opencv-python",
        "pip install --no-cache-dir insightface",
        "",
        "# 4. If on Apple Silicon Mac:",
        "pip install onnxruntime-silicon",
        "",
        "# 5. After installation, restart terminal and try again"
    ]
    
    for cmd in commands:
        print(cmd)

def main():
    """Main debug function"""
    print("🔧 TRAINING DEBUG TOOL")
    print("=" * 50)
    
    # Run tests
    imports_ok = test_basic_imports()
    if not imports_ok:
        provide_dependency_fixes()
        return
    
    insightface_ok = test_insightface_initialization()
    if not insightface_ok:
        provide_dependency_fixes()
        return
    
    training_ok = test_manual_training()
    
    # Create workaround regardless
    create_training_workaround()
    
    print("\n" + "=" * 50)
    if training_ok:
        print("✅ Training should work! Try the manual script:")
        print("   python manual_train.py")
    else:
        print("❌ Training still has issues. Try:")
        print("1. Fix dependencies using commands above")
        print("2. Run: python manual_train.py")
        print("3. Restart the application")

if __name__ == "__main__":
    main()