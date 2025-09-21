"""
Deep learning face detection using MTCNN and other state-of-the-art detectors.
Provides high-accuracy face detection with precise face alignment.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from PIL import Image
import torch

# Import detection models
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logging.warning("MTCNN not available")

# Keep legacy import for fallback
try:
    from mtcnn import MTCNN as LegacyMTCNN
    LEGACY_MTCNN_AVAILABLE = True
except ImportError:
    LEGACY_MTCNN_AVAILABLE = False

logger = logging.getLogger(__name__)

class Face:
    """Face detection result container."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], landmarks: Optional[np.ndarray] = None, 
                 confidence: float = 0.0, quality_score: float = 0.0):
        """
        Initialize face detection result.
        
        Args:
            bbox: Bounding box (x, y, width, height)
            landmarks: Facial landmarks array
            confidence: Detection confidence score
            quality_score: Face quality assessment score
        """
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = confidence
        self.quality_score = quality_score
        
        # Derived properties
        self.x, self.y, self.width, self.height = bbox
        self.center = (self.x + self.width // 2, self.y + self.height // 2)
        self.area = self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert face to dictionary representation."""
        return {
            'bbox': self.bbox,
            'landmarks': self.landmarks.tolist() if self.landmarks is not None else None,
            'confidence': self.confidence,
            'quality_score': self.quality_score,
            'center': self.center,
            'area': self.area
        }


class MTCNNDetector:
    """MTCNN-based face detector with alignment capabilities."""
    
    def __init__(self, min_face_size: int = 40, thresholds: List[float] = None, 
                 device: str = 'auto'):
        """
        Initialize MTCNN detector.
        
        Args:
            min_face_size: Minimum face size for detection
            thresholds: Detection thresholds for P-Net, R-Net, O-Net
            device: Device to run on ('cpu', 'cuda', 'auto')
        """
        self.min_face_size = min_face_size
        self.thresholds = thresholds or [0.6, 0.7, 0.7]
        self.device = self._get_device(device)
        
        # Initialize detector
        self.detector = None
        self._init_detector()
    
    def _get_device(self, device: str) -> str:
        """Determine optimal device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # Use MPS for Apple Silicon
            else:
                return 'cpu'
        return device
    
    def _init_detector(self):
        """Initialize MTCNN detector based on available implementations."""
        try:
            if MTCNN_AVAILABLE:
                # Use facenet-pytorch MTCNN (preferred)
                self.detector = MTCNN(
                    min_face_size=self.min_face_size,
                    thresholds=self.thresholds,
                    device=self.device,
                    keep_all=True,
                    post_process=False
                )
                self.detector_type = 'facenet_mtcnn'
                logger.info("Initialized Facenet-PyTorch MTCNN detector")
                
            elif LEGACY_MTCNN_AVAILABLE:
                # Use mtcnn package as fallback
                self.detector = LegacyMTCNN(
                    min_face_size=self.min_face_size,
                    thresholds=self.thresholds
                )
                self.detector_type = 'mtcnn'
                logger.info("Initialized Legacy MTCNN detector")
                
            else:
                raise ImportError("No MTCNN implementation available")
                
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN detector: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray, return_landmarks: bool = True) -> List[Face]:
        """
        Detect faces in image using MTCNN.
        
        Args:
            image: Input image (BGR format)
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            List of detected faces
        """
        if self.detector is None:
            raise RuntimeError("Detector not initialized")
        
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            faces = []
            
            if self.detector_type == 'facenet_mtcnn':
                # Use facenet-pytorch MTCNN
                pil_image = Image.fromarray(rgb_image)
                
                boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=return_landmarks)
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        if box is not None and probs[i] is not None:
                            # Convert to integer coordinates
                            x1, y1, x2, y2 = map(int, box)
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Get landmarks if available
                            face_landmarks = None
                            if return_landmarks and landmarks is not None and landmarks[i] is not None:
                                if hasattr(landmarks[i], 'numpy'):
                                    face_landmarks = landmarks[i].numpy()
                                else:
                                    face_landmarks = landmarks[i]
                            
                            # Calculate quality score (based on size and confidence)
                            area = width * height
                            quality_score = min(1.0, (area / (100 * 100)) * probs[i])
                            
                            face = Face(
                                bbox=(x1, y1, width, height),
                                landmarks=face_landmarks,
                                confidence=float(probs[i]),
                                quality_score=quality_score
                            )
                            faces.append(face)
            
            elif self.detector_type == 'mtcnn':
                # Use legacy mtcnn package
                result = self.detector.detect_faces(rgb_image)
                
                for detection in result:
                    bbox = detection['box']
                    confidence = detection['confidence']
                    
                    # Get landmarks if available
                    face_landmarks = None
                    if return_landmarks and 'keypoints' in detection:
                        keypoints = detection['keypoints']
                        face_landmarks = np.array([
                            [keypoints['left_eye'][0], keypoints['left_eye'][1]],
                            [keypoints['right_eye'][0], keypoints['right_eye'][1]],
                            [keypoints['nose'][0], keypoints['nose'][1]],
                            [keypoints['mouth_left'][0], keypoints['mouth_left'][1]],
                            [keypoints['mouth_right'][0], keypoints['mouth_right'][1]]
                        ])
                    
                    # Calculate quality score
                    area = bbox[2] * bbox[3]
                    quality_score = min(1.0, (area / (100 * 100)) * confidence)
                    
                    face = Face(
                        bbox=tuple(bbox),
                        landmarks=face_landmarks,
                        confidence=confidence,
                        quality_score=quality_score
                    )
                    faces.append(face)
            
            # Filter faces by minimum size and quality
            filtered_faces = []
            for face in faces:
                if (face.width >= self.min_face_size and 
                    face.height >= self.min_face_size and 
                    face.confidence > 0.5):
                    filtered_faces.append(face)
            
            # Sort by confidence (highest first)
            filtered_faces.sort(key=lambda f: f.confidence, reverse=True)
            
            logger.debug(f"Detected {len(filtered_faces)} faces in image")
            return filtered_faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def align_face(self, image: np.ndarray, face: Face, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        Align face using detected landmarks.
        
        Args:
            image: Original image
            face: Face detection result with landmarks
            target_size: Target size for aligned face
            
        Returns:
            Aligned face image
        """
        if face.landmarks is None:
            # Simple crop without alignment
            return self._simple_crop(image, face, target_size)
        
        try:
            # Use 5-point landmark alignment
            return self._align_face_5points(image, face.landmarks, target_size)
            
        except Exception as e:
            logger.warning(f"Landmark-based alignment failed: {e}, using simple crop")
            return self._simple_crop(image, face, target_size)
    
    def _simple_crop(self, image: np.ndarray, face: Face, target_size: Tuple[int, int]) -> np.ndarray:
        """Simple face cropping without alignment."""
        x, y, w, h = face.bbox
        
        # Add padding
        padding = 0.2
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        cropped = image[y1:y2, x1:x2]
        
        # Resize to target size
        aligned = cv2.resize(cropped, target_size)
        return aligned
    
    def _align_face_5points(self, image: np.ndarray, landmarks: np.ndarray, 
                           target_size: Tuple[int, int]) -> np.ndarray:
        """Align face using 5-point landmarks."""
        # Standard 5-point landmarks for target image
        target_landmarks = np.array([
            [30.2946, 51.6963],  # left eye
            [65.5318, 51.5014],  # right eye
            [48.0252, 71.7366],  # nose
            [33.5493, 92.3655],  # left mouth
            [62.7299, 92.2041]   # right mouth
        ])
        
        # Scale target landmarks to match target size
        target_landmarks[:, 0] *= target_size[0] / 96
        target_landmarks[:, 1] *= target_size[1] / 112
        
        # Estimate transformation matrix
        transform_matrix = cv2.estimateAffinePartial2D(landmarks[:5], target_landmarks)[0]
        
        # Apply transformation
        aligned = cv2.warpAffine(image, transform_matrix, target_size)
        return aligned
    
    def batch_detect(self, images: List[np.ndarray]) -> List[List[Face]]:
        """
        Detect faces in multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of face detection results for each image
        """
        results = []
        for i, image in enumerate(images):
            try:
                faces = self.detect_faces(image)
                results.append(faces)
            except Exception as e:
                logger.error(f"Batch detection failed for image {i}: {e}")
                results.append([])
        
        return results
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the detector."""
        return {
            'type': self.detector_type,
            'min_face_size': self.min_face_size,
            'thresholds': self.thresholds,
            'device': self.device,
            'available_implementations': {
                'facenet_mtcnn': FACENET_MTCNN_AVAILABLE,
                'mtcnn': MTCNN_AVAILABLE
            }
        }


class DLFaceDetector:
    """Main deep learning face detector interface."""
    
    def __init__(self, detector_type: str = 'mtcnn', **kwargs):
        """
        Initialize deep learning face detector.
        
        Args:
            detector_type: Type of detector ('mtcnn', 'retinaface')
            **kwargs: Additional arguments for specific detectors
        """
        self.detector_type = detector_type
        self.detector = None
        
        if detector_type == 'mtcnn':
            self.detector = MTCNNDetector(**kwargs)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
    
    def detect_faces(self, image: np.ndarray, **kwargs) -> List[Face]:
        """Detect faces using the configured detector."""
        return self.detector.detect_faces(image, **kwargs)
    
    def align_face(self, image: np.ndarray, face: Face, **kwargs) -> np.ndarray:
        """Align face using the configured detector."""
        return self.detector.align_face(image, face, **kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """Get detector information."""
        return {
            'detector_type': self.detector_type,
            'detector_info': self.detector.get_detector_info() if self.detector else None
        }