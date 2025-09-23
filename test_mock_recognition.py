#!/usr/bin/env python3
"""
Test the mock recognition system without full dependencies
"""

import sys
import os
sys.path.append('.')

def test_mock_recognizer():
    """Test the mock recognizer directly"""
    print("🧪 TESTING MOCK RECOGNIZER")
    print("-" * 40)
    
    try:
        from src.recognition.mock_recognizer import SimpleInsightFaceRecognizer
        
        print("Creating mock recognizer...")
        recognizer = SimpleInsightFaceRecognizer(confidence_threshold=0.5)
        
        print("Getting model info...")
        model_info = recognizer.get_model_info()
        print(f"✅ Model info: {model_info}")
        
        print("Testing training...")
        training_result = recognizer.train_model()
        print(f"✅ Training result: {training_result}")
        
        print("Testing recognition...")
        import numpy as np
        fake_image = np.random.random((100, 100, 3))
        user_id, similarity = recognizer.recognize_face(fake_image)
        print(f"✅ Recognition result: {user_id}, similarity: {similarity}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock recognizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embeddings_file():
    """Test if embeddings file was created"""
    print("\n🧪 TESTING EMBEDDINGS FILE")
    print("-" * 40)
    
    try:
        import pickle
        from pathlib import Path
        
        embeddings_file = Path('faces/metadata/insightface_embeddings.pkl')
        
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            print(f"✅ Embeddings file exists")
            print(f"✅ Users in embeddings: {list(embeddings.keys())}")
            print(f"✅ Total embeddings: {sum(len(emb) for emb in embeddings.values())}")
            return True
        else:
            print("❌ Embeddings file not found")
            return False
            
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 MOCK RECOGNITION SYSTEM TEST")
    print("=" * 50)
    
    test1 = test_embeddings_file()
    test2 = test_mock_recognizer()
    
    print("\n" + "=" * 50)
    if test1 and test2:
        print("✅ MOCK SYSTEM WORKING!")
        print()
        print("Now you can:")
        print("1. Start your application (python main.py)")
        print("2. Open 'Start Attendance' window")
        print("3. Click 'Retrain Model' - should work now")
        print("4. Set similarity threshold to 0.4-0.5")
        print("5. Start recognition - should recognize first user")
        print()
        print("The mock system will always recognize the first registered user")
        print("with a similarity of 0.75 when threshold is ≤ 0.75")
    else:
        print("❌ Some tests failed. Check errors above.")

if __name__ == "__main__":
    main()