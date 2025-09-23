"""
Simple InsightFace-based recognizer for FaceAttend
Uses the readily available insightface models for face recognition
"""

import cv2
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import insightface
from src.utils.logger import get_module_logger
from src.utils.exceptions import FaceRecognitionError


class SimpleInsightFaceRecognizer:
    """Simple face recognizer using InsightFace"""
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize the InsightFace recognizer
        
        Args:
            confidence_threshold: Minimum similarity score for recognition (0.0-1.0)
        """
        self.logger = get_module_logger("SimpleInsightFaceRecognizer")
        self.confidence_threshold = confidence_threshold
        
        # Initialize InsightFace app
        try:
            self.app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            raise FaceRecognitionError(f"InsightFace initialization failed: {e}")
        
        # Storage for user embeddings
        self.user_embeddings: Dict[str, List[np.ndarray]] = {}
        self.embeddings_file = "faces/metadata/insightface_embeddings.pkl"
        
        # Load existing embeddings if available
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load user embeddings from file"""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.user_embeddings = pickle.load(f)
                self.logger.info(f"Loaded embeddings for {len(self.user_embeddings)} users")
            else:
                self.user_embeddings = {}
                self.logger.info("No existing embeddings found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            self.user_embeddings = {}
    
    def _save_embeddings(self):
        """Save user embeddings to file"""
        try:
            os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.user_embeddings, f)
            self.logger.info(f"Saved embeddings for {len(self.user_embeddings)} users")
        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {e}")
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate face embedding from image"""
        try:
            # Ensure image is in BGR format (InsightFace expects BGR)
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Already BGR
                pass
            else:
                self.logger.warning("Unexpected image format")
                return None
            
            # Detect faces and generate embeddings
            faces = self.app.get(face_image)
            
            if len(faces) == 0:
                self.logger.warning("No face detected in image")
                return None
            
            if len(faces) > 1:
                self.logger.warning("Multiple faces detected, using the largest one")
            
            # Use the face with the largest area
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Return the embedding
            return largest_face.embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def train_user(self, user_id: str, face_images: List[np.ndarray]) -> bool:
        """Train/add a user with their face images"""
        try:
            embeddings = []
            
            for i, image in enumerate(face_images):
                embedding = self.generate_embedding(image)
                if embedding is not None:
                    embeddings.append(embedding)
                    self.logger.debug(f"Generated embedding {i+1}/{len(face_images)} for user {user_id}")
                else:
                    self.logger.warning(f"Failed to generate embedding {i+1}/{len(face_images)} for user {user_id}")
            
            if embeddings:
                self.user_embeddings[user_id] = embeddings
                self._save_embeddings()
                self.logger.info(f"Trained user {user_id} with {len(embeddings)}/{len(face_images)} valid embeddings")
                return True
            else:
                self.logger.error(f"No valid embeddings generated for user {user_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Training failed for user {user_id}: {e}")
            return False
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face in the given image
        
        Returns:
            Tuple of (user_id, confidence) where confidence is similarity score (0.0-1.0)
        """
        try:
            # Generate embedding for the query image
            query_embedding = self.generate_embedding(face_image)
            if query_embedding is None:
                return None, 0.0
            
            if not self.user_embeddings:
                self.logger.warning("No user embeddings available for recognition")
                return None, 0.0
            
            best_user_id = None
            best_similarity = 0.0
            
            # Compare with all user embeddings
            for user_id, embeddings in self.user_embeddings.items():
                # Calculate similarity with all embeddings for this user
                similarities = []
                for embedding in embeddings:
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    similarities.append(similarity)
                
                # Use the maximum similarity for this user
                max_similarity = max(similarities) if similarities else 0.0
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_user_id = user_id
            
            # Check if similarity meets threshold
            if best_similarity >= self.confidence_threshold:
                self.logger.debug(f"Recognized user {best_user_id} with similarity {best_similarity:.3f}")
                return best_user_id, best_similarity
            else:
                self.logger.debug(f"No recognition above threshold. Best: {best_user_id} ({best_similarity:.3f})")
                return None, best_similarity
                
        except Exception as e:
            self.logger.error(f"Recognition failed: {e}")
            return None, 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': 'InsightFace',
            'model_type': 'deep_learning',
            'provider': 'CPU',
            'confidence_threshold': self.confidence_threshold,
            'total_users': len(self.user_embeddings),
            'total_embeddings': sum(len(embeddings) for embeddings in self.user_embeddings.values()),
            'is_trained': len(self.user_embeddings) > 0
        }
    
    def is_model_trained(self) -> bool:
        """Check if the model has any user embeddings"""
        return len(self.user_embeddings) > 0
    
    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Confidence threshold updated to {self.confidence_threshold}")
    
    def train_model(self) -> Dict[str, Any]:
        """
        Train the model with all available users (compatibility method)
        For InsightFace, this is essentially a no-op since training happens per-user
        """
        from src.storage.face_storage import FaceStorage
        
        try:
            face_storage = FaceStorage()
            users = face_storage.list_users()
            
            total_trained = 0
            for user_id in users:
                user_images = face_storage.get_user_face_images(user_id)
                if user_images and self.train_user(user_id, user_images):
                    total_trained += 1
            
            return {
                'success': total_trained > 0,
                'users_count': total_trained,
                'total_users': len(users),
                'method': 'insightface'
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {
                'success': False,
                'users_count': 0,
                'error': str(e)
            }