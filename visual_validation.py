#!/usr/bin/env python3
"""
Visual validation script - shows face detection results with bounding boxes.
Creates output images to visually verify the implementation works correctly.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def draw_detection_results(image, faces, title=""):
    """Draw face detection results on image."""
    result_image = image.copy()
    
    for i, face in enumerate(faces):
        x, y, w, h = face.bbox
        
        # Draw bounding box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw confidence and quality scores
        label = f"Face {i+1}: {face.confidence:.3f}"
        quality_label = f"Quality: {face.quality_score:.3f}"
        
        cv2.putText(result_image, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(result_image, quality_label, (x, y + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw landmarks if available
        if face.landmarks is not None:
            landmarks = face.landmarks.astype(int)
            for point in landmarks:
                cv2.circle(result_image, tuple(point), 2, (255, 0, 0), -1)
    
    # Add title
    if title:
        cv2.putText(result_image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result_image

def test_face_detection_on_images():
    """Test face detection on all available face images."""
    print("Testing face detection on existing images...")
    
    try:
        from src.recognition.deep_learning.dl_face_detector import MTCNNDetector
        
        detector = MTCNNDetector(min_face_size=30, device='cpu')
        
        # Find all face images
        face_dirs = ['faces/khoi1234', 'faces/huy123', 'faces/quangnm']
        output_dir = Path('validation_output')
        output_dir.mkdir(exist_ok=True)
        
        total_detections = 0
        processed_images = 0
        
        for face_dir in face_dirs:
            face_path = Path(face_dir)
            if not face_path.exists():
                continue
                
            user_name = face_path.name
            print(f"\nProcessing user: {user_name}")
            
            # Process each image
            for img_file in face_path.glob('*.jpg'):
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                # Detect faces
                start_time = time.time()
                faces = detector.detect_faces(image)
                detection_time = time.time() - start_time
                
                total_detections += len(faces)
                processed_images += 1
                
                print(f"  {img_file.name}: {len(faces)} faces, {detection_time*1000:.1f}ms")
                
                # Create visualization
                result_image = draw_detection_results(
                    image, faces, 
                    f"{user_name} - {img_file.name} - {len(faces)} faces"
                )
                
                # Save result
                output_file = output_dir / f"{user_name}_{img_file.name}"
                cv2.imwrite(str(output_file), result_image)
        
        print(f"\nSummary:")
        print(f"  Processed images: {processed_images}")
        print(f"  Total faces detected: {total_detections}")
        print(f"  Average faces per image: {total_detections/max(1, processed_images):.1f}")
        print(f"  Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error during visual validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_alignment():
    """Test face alignment on detected faces."""
    print("\nTesting face alignment...")
    
    try:
        from src.recognition.deep_learning.dl_face_detector import MTCNNDetector
        
        detector = MTCNNDetector(min_face_size=40, device='cpu')
        output_dir = Path('validation_output/aligned_faces')
        output_dir.mkdir(exist_ok=True)
        
        # Find a test image
        test_paths = ['faces/khoi1234/img_01.jpg', 'faces/huy123/img_01.jpg']
        
        for test_path in test_paths:
            if not Path(test_path).exists():
                continue
                
            image = cv2.imread(test_path)
            if image is None:
                continue
            
            faces = detector.detect_faces(image)
            
            if faces:
                user_name = Path(test_path).parent.name
                
                for i, face in enumerate(faces):
                    # Test different alignment sizes
                    for size in [(112, 112), (224, 224)]:
                        try:
                            aligned = detector.align_face(image, face, target_size=size)
                            
                            # Save aligned face
                            output_file = output_dir / f"{user_name}_face{i}_{size[0]}x{size[1]}.jpg"
                            cv2.imwrite(str(output_file), aligned)
                            
                            print(f"  Aligned face saved: {output_file.name}")
                            
                        except Exception as e:
                            print(f"  Alignment failed for {size}: {e}")
                
                break  # Only test first available image
        
        return True
        
    except Exception as e:
        print(f"Error during alignment test: {e}")
        return False

def test_preprocessing_visualization():
    """Test preprocessing with visualization."""
    print("\nTesting preprocessing with visualization...")
    
    try:
        from src.recognition.deep_learning.preprocessing import FacePreprocessor
        
        preprocessor = FacePreprocessor(target_size=(112, 112))
        output_dir = Path('validation_output/preprocessing')
        output_dir.mkdir(exist_ok=True)
        
        # Test with existing face image
        test_paths = ['faces/khoi1234/img_01.jpg', 'faces/huy123/img_01.jpg']
        
        for test_path in test_paths:
            if not Path(test_path).exists():
                continue
                
            image = cv2.imread(test_path)
            if image is None:
                continue
            
            # Crop a face region (simple crop for demo)
            h, w = image.shape[:2]
            face_crop = image[h//4:3*h//4, w//4:3*w//4]
            
            # Preprocess
            result = preprocessor.preprocess_face(face_crop)
            
            if result['preprocessed_image'] is not None:
                # Denormalize for visualization
                processed = result['preprocessed_image']
                
                # Convert back to uint8 for saving
                if processed.dtype == np.float32:
                    # Simple denormalization
                    processed = np.clip(processed * 255, 0, 255).astype(np.uint8)
                
                user_name = Path(test_path).parent.name
                output_file = output_dir / f"{user_name}_preprocessed.jpg"
                cv2.imwrite(str(output_file), processed)
                
                print(f"  Preprocessed image saved: {output_file.name}")
                print(f"    Quality score: {result['quality_score']:.3f}")
                print(f"    Applied operations: {result['preprocessing_applied']}")
                
                break
        
        return True
        
    except Exception as e:
        print(f"Error during preprocessing visualization: {e}")
        return False

def create_comparison_image():
    """Create side-by-side comparison of classical vs deep learning detection."""
    print("\nCreating classical vs deep learning comparison...")
    
    try:
        from src.recognition.classical.face_detector import FaceDetector as ClassicalDetector
        from src.recognition.deep_learning.dl_face_detector import MTCNNDetector
        
        classical_detector = ClassicalDetector()
        dl_detector = MTCNNDetector(min_face_size=40, device='cpu')
        
        # Find test image
        test_paths = ['faces/khoi1234/img_01.jpg', 'faces/huy123/img_01.jpg']
        
        for test_path in test_paths:
            if not Path(test_path).exists():
                continue
                
            image = cv2.imread(test_path)
            if image is None:
                continue
            
            # Classical detection
            classical_faces = classical_detector.detect_faces(image)
            classical_result = image.copy()
            
            for face in classical_faces:
                x, y, w, h = face
                cv2.rectangle(classical_result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(classical_result, "Classical", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Deep learning detection
            dl_faces = dl_detector.detect_faces(image)
            dl_result = image.copy()
            
            for face in dl_faces:
                x, y, w, h = face.bbox
                cv2.rectangle(dl_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(dl_result, f"MTCNN: {face.confidence:.2f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Create side-by-side comparison
            comparison = np.hstack([classical_result, dl_result])
            
            # Add labels
            h, w = comparison.shape[:2]
            cv2.putText(comparison, "Classical Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(comparison, "Deep Learning Detection", (w//2 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save comparison
            output_dir = Path('validation_output')
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"comparison_{Path(test_path).parent.name}.jpg"
            cv2.imwrite(str(output_file), comparison)
            
            print(f"  Comparison saved: {output_file}")
            print(f"    Classical detected: {len(classical_faces)} faces")
            print(f"    Deep learning detected: {len(dl_faces)} faces")
            
            break
        
        return True
        
    except Exception as e:
        print(f"Error creating comparison: {e}")
        return False

def main():
    """Run visual validation tests."""
    print("🎨 Visual Validation for FaceAttend Phase A Implementation")
    print("=" * 60)
    
    # Clean up previous results
    output_dir = Path('validation_output')
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    tests = [
        ("Face Detection Visualization", test_face_detection_on_images),
        ("Face Alignment Test", test_face_alignment),
        ("Preprocessing Visualization", test_preprocessing_visualization),
        ("Classical vs Deep Learning Comparison", create_comparison_image)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                print(f"✅ {test_name} completed successfully")
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print(f"\n📊 Visual Validation Results: {passed}/{total} tests passed")
    
    if passed > 0:
        print(f"\n📁 Check the 'validation_output' directory for visual results")
        print("   - Face detection results with bounding boxes")
        print("   - Aligned face images") 
        print("   - Preprocessed face images")
        print("   - Classical vs Deep Learning comparison")
    
    return passed == total

if __name__ == "__main__":
    main()