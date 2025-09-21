"""
Classical computer vision face recognition module.
Contains LBPH-based recognition system with Haar cascade detection.
"""

from .face_detector import FaceDetector
from .image_processor import ImageProcessor
from .lbph_recognizer import LBPHRecognizer

__all__ = ['FaceDetector', 'ImageProcessor', 'LBPHRecognizer']