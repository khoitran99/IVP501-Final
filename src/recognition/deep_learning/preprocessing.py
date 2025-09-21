"""
Deep learning preprocessing utilities for face recognition.
Handles image preprocessing, augmentation, and quality assessment.
"""

import os
# Suppress albumentations version check warnings
os.environ.setdefault('ALBUMENTATIONS_DISABLE_VERSION_CHECK', '1')

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging
from PIL import Image, ImageEnhance
import albumentations as A

logger = logging.getLogger(__name__)

class FacePreprocessor:
    """Face preprocessing for deep learning models."""
    
    def __init__(self, target_size: Tuple[int, int] = (112, 112), 
                 normalize: bool = True, quality_check: bool = True):
        """
        Initialize face preprocessor.
        
        Args:
            target_size: Target size for face images
            normalize: Whether to normalize pixel values
            quality_check: Whether to perform quality assessment
        """
        self.target_size = target_size
        self.normalize = normalize
        self.quality_check = quality_check
        
        # Define preprocessing pipeline
        self.augmentation_pipeline = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),  # Alternative to GaussNoise
            A.Blur(blur_limit=3, p=0.1),
        ])
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            'min_sharpness': 50.0,
            'min_brightness': 30.0,
            'max_brightness': 220.0,
            'min_contrast': 20.0,
            'max_blur': 15.0
        }
    
    def preprocess_face(self, face_image: np.ndarray, augment: bool = False) -> Dict[str, Any]:
        """
        Preprocess face image for deep learning inference.
        
        Args:
            face_image: Input face image
            augment: Whether to apply data augmentation
            
        Returns:
            Dictionary with processed image and metadata
        """
        try:
            result = {
                'original_shape': face_image.shape,
                'preprocessed_image': None,
                'quality_score': 0.0,
                'quality_metrics': {},
                'preprocessing_applied': []
            }
            
            # Convert to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Assume BGR input, convert to RGB
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                result['preprocessing_applied'].append('bgr_to_rgb')
            else:
                rgb_image = face_image.copy()
            
            # Quality assessment
            if self.quality_check:
                quality_metrics = self._assess_quality(rgb_image)
                result['quality_metrics'] = quality_metrics
                result['quality_score'] = self._calculate_quality_score(quality_metrics)
            
            # Resize to target size
            if rgb_image.shape[:2] != self.target_size:
                resized_image = cv2.resize(rgb_image, self.target_size)
                result['preprocessing_applied'].append('resize')
            else:
                resized_image = rgb_image.copy()
            
            # Apply histogram equalization
            enhanced_image = self._enhance_image(resized_image)
            if not np.array_equal(enhanced_image, resized_image):
                result['preprocessing_applied'].append('histogram_equalization')
                resized_image = enhanced_image
            
            # Data augmentation (if requested and training)
            if augment:
                augmented = self.augmentation_pipeline(image=resized_image)['image']
                resized_image = augmented
                result['preprocessing_applied'].append('augmentation')
            
            # Normalization
            if self.normalize:
                normalized_image = self._normalize_image(resized_image)
                result['preprocessing_applied'].append('normalization')
            else:
                normalized_image = resized_image.astype(np.float32)
            
            result['preprocessed_image'] = normalized_image
            
            return result
            
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            raise
    
    def _assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess face image quality metrics."""
        metrics = {}
        
        try:
            # Convert to grayscale for some metrics
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Sharpness (Laplacian variance)
            metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Brightness (mean intensity)
            metrics['brightness'] = np.mean(gray)
            
            # Contrast (standard deviation)
            metrics['contrast'] = np.std(gray)
            
            # Blur estimation (FFT-based)
            metrics['blur_estimate'] = self._estimate_blur_fft(gray)
            
            # Dynamic range
            metrics['dynamic_range'] = np.ptp(gray)  # Peak-to-peak
            
            # Histogram uniformity
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / hist.sum()
            metrics['histogram_entropy'] = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            # Return default metrics
            metrics = {
                'sharpness': 100.0,
                'brightness': 128.0,
                'contrast': 50.0,
                'blur_estimate': 5.0,
                'dynamic_range': 255.0,
                'histogram_entropy': 6.0
            }
        
        return metrics
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from metrics."""
        score = 1.0
        
        # Sharpness component
        if metrics['sharpness'] < self.quality_thresholds['min_sharpness']:
            score *= metrics['sharpness'] / self.quality_thresholds['min_sharpness']
        
        # Brightness component
        brightness = metrics['brightness']
        if brightness < self.quality_thresholds['min_brightness']:
            score *= brightness / self.quality_thresholds['min_brightness']
        elif brightness > self.quality_thresholds['max_brightness']:
            score *= (255 - brightness) / (255 - self.quality_thresholds['max_brightness'])
        
        # Contrast component
        if metrics['contrast'] < self.quality_thresholds['min_contrast']:
            score *= metrics['contrast'] / self.quality_thresholds['min_contrast']
        
        # Blur component
        if metrics['blur_estimate'] > self.quality_thresholds['max_blur']:
            score *= self.quality_thresholds['max_blur'] / metrics['blur_estimate']
        
        return max(0.0, min(1.0, score))
    
    def _estimate_blur_fft(self, image: np.ndarray) -> float:
        """Estimate blur using FFT analysis."""
        try:
            # Apply FFT
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Calculate high frequency content
            h, w = image.shape
            center_h, center_w = h // 2, w // 2
            
            # Create high-pass filter
            high_pass = np.zeros((h, w))
            radius = min(h, w) // 4
            y, x = np.ogrid[:h, :w]
            mask = (x - center_w)**2 + (y - center_h)**2 >= radius**2
            high_pass[mask] = 1
            
            # Apply filter and calculate energy
            high_freq_energy = np.sum(magnitude_spectrum * high_pass)
            total_energy = np.sum(magnitude_spectrum)
            
            # Blur estimate (higher ratio = less blur)
            blur_ratio = high_freq_energy / (total_energy + 1e-10)
            blur_estimate = max(1.0, 50.0 * (1.0 - blur_ratio))
            
            return blur_estimate
            
        except Exception:
            return 10.0  # Default medium blur estimate
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image using CLAHE and other techniques."""
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                
                # Convert back to RGB
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for deep learning models."""
        # Convert to float32
        normalized = image.astype(np.float32)
        
        # Standard normalization: [0, 255] -> [0, 1] -> standardize
        normalized = normalized / 255.0
        
        # Use ImageNet statistics (common for pretrained models)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if len(normalized.shape) == 3 and normalized.shape[2] == 3:
            normalized = (normalized - mean) / std
        else:
            # Grayscale normalization
            normalized = (normalized - 0.5) / 0.5
        
        return normalized
    
    def batch_preprocess(self, face_images: List[np.ndarray], 
                        augment: bool = False) -> List[Dict[str, Any]]:
        """
        Preprocess multiple face images.
        
        Args:
            face_images: List of face images
            augment: Whether to apply augmentation
            
        Returns:
            List of preprocessing results
        """
        results = []
        
        for i, face_image in enumerate(face_images):
            try:
                result = self.preprocess_face(face_image, augment=augment)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch preprocessing failed for image {i}: {e}")
                # Add empty result
                results.append({
                    'original_shape': face_image.shape if face_image is not None else (0, 0),
                    'preprocessed_image': None,
                    'quality_score': 0.0,
                    'quality_metrics': {},
                    'preprocessing_applied': []
                })
        
        return results
    
    def create_training_batch(self, face_images: List[np.ndarray], 
                            labels: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Create training batch with augmentation.
        
        Args:
            face_images: List of face images
            labels: Optional labels for supervised learning
            
        Returns:
            Training batch dictionary
        """
        processed_images = []
        valid_labels = []
        
        for i, face_image in enumerate(face_images):
            try:
                result = self.preprocess_face(face_image, augment=True)
                if result['preprocessed_image'] is not None:
                    processed_images.append(result['preprocessed_image'])
                    if labels is not None:
                        valid_labels.append(labels[i])
            except Exception as e:
                logger.warning(f"Failed to process training image {i}: {e}")
        
        batch = {
            'images': np.array(processed_images) if processed_images else np.array([]),
            'count': len(processed_images)
        }
        
        if labels is not None and valid_labels:
            batch['labels'] = np.array(valid_labels)
        
        return batch
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing configuration and statistics."""
        return {
            'target_size': self.target_size,
            'normalize': self.normalize,
            'quality_check': self.quality_check,
            'quality_thresholds': self.quality_thresholds,
            'augmentation_pipeline': str(self.augmentation_pipeline)
        }


class ImageQualityFilter:
    """Filter images based on quality metrics."""
    
    def __init__(self, quality_threshold: float = 0.5):
        """
        Initialize quality filter.
        
        Args:
            quality_threshold: Minimum quality score to accept
        """
        self.quality_threshold = quality_threshold
        self.preprocessor = FacePreprocessor(quality_check=True)
    
    def filter_faces(self, face_images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Filter face images based on quality.
        
        Args:
            face_images: List of face images to filter
            
        Returns:
            Tuple of (filtered_images, quality_reports)
        """
        filtered_images = []
        quality_reports = []
        
        for i, face_image in enumerate(face_images):
            try:
                result = self.preprocessor.preprocess_face(face_image, augment=False)
                quality_score = result['quality_score']
                
                quality_report = {
                    'index': i,
                    'quality_score': quality_score,
                    'accepted': quality_score >= self.quality_threshold,
                    'metrics': result['quality_metrics']
                }
                
                if quality_score >= self.quality_threshold:
                    filtered_images.append(face_image)
                
                quality_reports.append(quality_report)
                
            except Exception as e:
                logger.error(f"Quality filtering failed for image {i}: {e}")
                quality_reports.append({
                    'index': i,
                    'quality_score': 0.0,
                    'accepted': False,
                    'metrics': {},
                    'error': str(e)
                })
        
        return filtered_images, quality_reports
    
    def get_quality_statistics(self, quality_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality statistics from reports."""
        if not quality_reports:
            return {}
        
        accepted_reports = [r for r in quality_reports if r.get('accepted', False)]
        
        stats = {
            'total_images': len(quality_reports),
            'accepted_images': len(accepted_reports),
            'rejection_rate': 1.0 - (len(accepted_reports) / len(quality_reports)),
            'quality_threshold': self.quality_threshold
        }
        
        if accepted_reports:
            quality_scores = [r['quality_score'] for r in accepted_reports]
            stats.update({
                'avg_quality_score': np.mean(quality_scores),
                'min_quality_score': np.min(quality_scores),
                'max_quality_score': np.max(quality_scores),
                'std_quality_score': np.std(quality_scores)
            })
        
        return stats