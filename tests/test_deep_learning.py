"""
Unit tests for deep learning components.
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestMTCNNDetector(unittest.TestCase):
    """Test MTCNN face detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.recognition.deep_learning.dl_face_detector import MTCNNDetector
        self.detector = MTCNNDetector(min_face_size=40, device='cpu')
    
    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.min_face_size, 40)
        self.assertEqual(self.detector.device, 'cpu')
    
    def test_detector_info(self):
        """Test detector info retrieval."""
        info = self.detector.get_detector_info()
        self.assertIn('type', info)
        self.assertIn('device', info)
        self.assertIn('min_face_size', info)
    
    def test_face_detection_synthetic(self):
        """Test face detection on synthetic image."""
        # Create synthetic test image
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Should not crash
        faces = self.detector.detect_faces(image)
        self.assertIsInstance(faces, list)
    
    def test_face_alignment(self):
        """Test face alignment."""
        from src.recognition.deep_learning.dl_face_detector import Face
        
        # Create test image and face
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        test_face = Face(bbox=(50, 50, 100, 100), confidence=0.9)
        
        # Test alignment
        aligned = self.detector.align_face(image, test_face, target_size=(112, 112))
        self.assertEqual(aligned.shape, (112, 112, 3))
    
    def test_batch_detection(self):
        """Test batch face detection."""
        images = [
            np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        ]
        
        results = self.detector.batch_detect(images)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], list)
        self.assertIsInstance(results[1], list)


class TestFacePreprocessor(unittest.TestCase):
    """Test face preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.recognition.deep_learning.preprocessing import FacePreprocessor
        self.preprocessor = FacePreprocessor(target_size=(112, 112))
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        self.assertIsNotNone(self.preprocessor)
        self.assertEqual(self.preprocessor.target_size, (112, 112))
    
    def test_face_preprocessing(self):
        """Test face preprocessing."""
        # Create test image
        test_image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        
        result = self.preprocessor.preprocess_face(test_image)
        
        # Check result structure
        self.assertIn('preprocessed_image', result)
        self.assertIn('quality_score', result)
        self.assertIn('quality_metrics', result)
        self.assertIn('preprocessing_applied', result)
        
        # Check processed image
        processed = result['preprocessed_image']
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape[:2], (112, 112))
    
    def test_quality_assessment(self):
        """Test quality assessment."""
        # Create high quality image (sharp, good contrast)
        high_quality = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.rectangle(high_quality, (20, 20), (80, 80), (255, 255, 255), 2)
        
        result = self.preprocessor.preprocess_face(high_quality)
        quality_score = result['quality_score']
        
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing."""
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        ]
        
        results = self.preprocessor.batch_preprocess(images)
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIn('preprocessed_image', result)
            self.assertIn('quality_score', result)


class TestImageQualityFilter(unittest.TestCase):
    """Test image quality filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.recognition.deep_learning.preprocessing import ImageQualityFilter
        self.filter = ImageQualityFilter(quality_threshold=0.5)
    
    def test_filter_initialization(self):
        """Test filter initializes correctly."""
        self.assertIsNotNone(self.filter)
        self.assertEqual(self.filter.quality_threshold, 0.5)
    
    def test_face_filtering(self):
        """Test face filtering."""
        # Create test images
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        ]
        
        filtered_images, quality_reports = self.filter.filter_faces(images)
        
        self.assertIsInstance(filtered_images, list)
        self.assertIsInstance(quality_reports, list)
        self.assertEqual(len(quality_reports), 2)
        
        for report in quality_reports:
            self.assertIn('quality_score', report)
            self.assertIn('accepted', report)
            self.assertIn('metrics', report)
    
    def test_quality_statistics(self):
        """Test quality statistics calculation."""
        # Mock quality reports
        reports = [
            {'quality_score': 0.8, 'accepted': True, 'metrics': {}},
            {'quality_score': 0.3, 'accepted': False, 'metrics': {}},
            {'quality_score': 0.7, 'accepted': True, 'metrics': {}}
        ]
        
        stats = self.filter.get_quality_statistics(reports)
        
        self.assertIn('total_images', stats)
        self.assertIn('accepted_images', stats)
        self.assertIn('rejection_rate', stats)
        self.assertEqual(stats['total_images'], 3)
        self.assertEqual(stats['accepted_images'], 2)


class TestModelDownloader(unittest.TestCase):
    """Test model downloader."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.model_downloader import ModelDownloader
        self.downloader = ModelDownloader()
    
    def test_downloader_initialization(self):
        """Test downloader initializes correctly."""
        self.assertIsNotNone(self.downloader)
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = self.downloader.list_available_models()
        
        self.assertIn('models', models)
        self.assertIn('detection', models['models'])
        self.assertIn('recognition', models['models'])
    
    def test_model_path_resolution(self):
        """Test model path resolution."""
        # Should not crash even if model doesn't exist
        path = self.downloader.get_model_path('detection', 'mtcnn')
        # Path can be None if model not downloaded
        self.assertTrue(path is None or isinstance(path, str))


class TestModelUtilities(unittest.TestCase):
    """Test model utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.utils.model_utils import ModelLoader, ModelValidator, get_model_loader
        self.loader = get_model_loader()
        self.validator = ModelValidator()
    
    def test_model_loader_initialization(self):
        """Test model loader initializes correctly."""
        self.assertIsNotNone(self.loader)
        self.assertIn(self.loader.device, ['cpu', 'cuda', 'mps'])
    
    def test_device_detection(self):
        """Test device detection."""
        device = self.loader._get_optimal_device()
        self.assertIn(device, ['cpu', 'cuda', 'mps'])
    
    def test_model_validator(self):
        """Test model validator."""
        self.assertIsNotNone(self.validator)
        
        # Test with non-existent file
        is_valid, message = self.validator.validate_onnx_model('nonexistent.onnx')
        self.assertFalse(is_valid)
        self.assertIn('does not exist', message)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)