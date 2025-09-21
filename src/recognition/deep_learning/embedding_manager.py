"""
Embedding management system for deep learning face recognition.
Handles storage, retrieval, and management of face embeddings with metadata.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import pickle

from .dl_face_recognizer import FaceEmbedding
from src.utils.logger import get_module_logger

logger = get_module_logger(__name__)

class EmbeddingMetadata:
    """Metadata for face embeddings."""
    
    def __init__(self, user_id: str, image_path: str = "", model_name: str = "",
                 created_at: str = "", quality_score: float = 0.0, 
                 confidence: float = 0.0, embedding_id: str = ""):
        """
        Initialize embedding metadata.
        
        Args:
            user_id: User identifier
            image_path: Path to source image
            model_name: Model used for generation
            created_at: Creation timestamp
            quality_score: Quality assessment score
            confidence: Embedding confidence
            embedding_id: Unique embedding identifier
        """
        self.user_id = user_id
        self.image_path = image_path
        self.model_name = model_name
        self.created_at = created_at or datetime.now().isoformat()
        self.quality_score = quality_score
        self.confidence = confidence
        self.embedding_id = embedding_id or f"{user_id}_{int(datetime.now().timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'image_path': self.image_path,
            'model_name': self.model_name,
            'created_at': self.created_at,
            'quality_score': self.quality_score,
            'confidence': self.confidence,
            'embedding_id': self.embedding_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingMetadata':
        """Create from dictionary."""
        return cls(**data)


class EmbeddingStorage:
    """Storage manager for face embeddings."""
    
    def __init__(self, base_path: str = "faces"):
        """
        Initialize embedding storage.
        
        Args:
            base_path: Base directory for face data
        """
        self.base_path = Path(base_path)
        self.embedding_format = 'numpy'  # 'numpy' or 'pickle'
        
        # Ensure base directory exists
        self.base_path.mkdir(exist_ok=True)
    
    def save_embedding(self, user_id: str, embedding: FaceEmbedding, 
                      metadata: EmbeddingMetadata) -> str:
        """
        Save face embedding with metadata.
        
        Args:
            user_id: User identifier
            embedding: Face embedding to save
            metadata: Embedding metadata
            
        Returns:
            Path to saved embedding file
        """
        try:
            # Create user directory structure
            user_dir = self.base_path / user_id
            embeddings_dir = user_dir / 'embeddings'
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            embedding_file = embeddings_dir / f"{metadata.model_name}_{timestamp}.npy"
            metadata_file = embeddings_dir / f"{metadata.model_name}_{timestamp}.json"
            
            # Save embedding
            if self.embedding_format == 'numpy':
                np.save(embedding_file, embedding.embedding)
            else:
                with open(embedding_file.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(embedding, f)
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            logger.info(f"Saved embedding for user {user_id}: {embedding_file}")
            return str(embedding_file)
            
        except Exception as e:
            logger.error(f"Failed to save embedding for user {user_id}: {e}")
            raise
    
    def load_embedding(self, embedding_path: str) -> Tuple[FaceEmbedding, EmbeddingMetadata]:
        """
        Load embedding and metadata from file.
        
        Args:
            embedding_path: Path to embedding file
            
        Returns:
            Tuple of (embedding, metadata)
        """
        try:
            embedding_path = Path(embedding_path)
            
            # Load embedding
            if embedding_path.suffix == '.npy':
                embedding_data = np.load(embedding_path)
                
                # Create FaceEmbedding object
                embedding = FaceEmbedding(embedding=embedding_data)
                
            elif embedding_path.suffix == '.pkl':
                with open(embedding_path, 'rb') as f:
                    embedding = pickle.load(f)
            else:
                raise ValueError(f"Unsupported embedding format: {embedding_path.suffix}")
            
            # Load metadata
            metadata_path = embedding_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = EmbeddingMetadata.from_dict(metadata_dict)
            else:
                # Create default metadata
                metadata = EmbeddingMetadata(
                    user_id=embedding_path.parent.parent.name,
                    image_path="",
                    model_name=embedding_path.stem.split('_')[0],
                    quality_score=embedding.quality_score,
                    confidence=embedding.confidence
                )
            
            return embedding, metadata
            
        except Exception as e:
            logger.error(f"Failed to load embedding from {embedding_path}: {e}")
            raise
    
    def get_user_embeddings(self, user_id: str) -> List[Tuple[FaceEmbedding, EmbeddingMetadata]]:
        """
        Get all embeddings for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of (embedding, metadata) tuples
        """
        try:
            user_dir = self.base_path / user_id / 'embeddings'
            if not user_dir.exists():
                return []
            
            embeddings = []
            
            # Find all embedding files
            for embedding_file in user_dir.glob('*.npy'):
                try:
                    embedding, metadata = self.load_embedding(embedding_file)
                    embeddings.append((embedding, metadata))
                except Exception as e:
                    logger.warning(f"Failed to load embedding {embedding_file}: {e}")
            
            # Sort by quality score (highest first)
            embeddings.sort(key=lambda x: x[1].quality_score, reverse=True)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings for user {user_id}: {e}")
            return []
    
    def get_best_embedding(self, user_id: str, model_name: str = None) -> Optional[Tuple[FaceEmbedding, EmbeddingMetadata]]:
        """
        Get the best quality embedding for a user.
        
        Args:
            user_id: User identifier
            model_name: Filter by model name (optional)
            
        Returns:
            Best embedding and metadata, or None
        """
        embeddings = self.get_user_embeddings(user_id)
        
        if model_name:
            embeddings = [(e, m) for e, m in embeddings if m.model_name == model_name]
        
        if embeddings:
            return embeddings[0]  # Already sorted by quality
        
        return None
    
    def delete_user_embeddings(self, user_id: str) -> bool:
        """
        Delete all embeddings for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Success status
        """
        try:
            user_dir = self.base_path / user_id / 'embeddings'
            if user_dir.exists():
                import shutil
                shutil.rmtree(user_dir)
                logger.info(f"Deleted embeddings for user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for user {user_id}: {e}")
            return False
    
    def get_all_users(self) -> List[str]:
        """Get list of all users with embeddings."""
        try:
            users = []
            for user_dir in self.base_path.iterdir():
                if user_dir.is_dir() and (user_dir / 'embeddings').exists():
                    if any((user_dir / 'embeddings').glob('*.npy')):
                        users.append(user_dir.name)
            return sorted(users)
            
        except Exception as e:
            logger.error(f"Failed to get users list: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {
                'total_users': 0,
                'total_embeddings': 0,
                'models_used': set(),
                'storage_size_mb': 0.0,
                'user_stats': {}
            }
            
            for user_dir in self.base_path.iterdir():
                if not user_dir.is_dir():
                    continue
                
                embeddings_dir = user_dir / 'embeddings'
                if not embeddings_dir.exists():
                    continue
                
                user_id = user_dir.name
                user_embeddings = list(embeddings_dir.glob('*.npy'))
                
                if user_embeddings:
                    stats['total_users'] += 1
                    stats['total_embeddings'] += len(user_embeddings)
                    
                    # Calculate user storage size
                    user_size = sum(f.stat().st_size for f in embeddings_dir.iterdir())
                    stats['storage_size_mb'] += user_size / (1024 * 1024)
                    
                    # Track models used
                    user_models = set()
                    for embedding_file in user_embeddings:
                        model_name = embedding_file.stem.split('_')[0]
                        stats['models_used'].add(model_name)
                        user_models.add(model_name)
                    
                    stats['user_stats'][user_id] = {
                        'embedding_count': len(user_embeddings),
                        'models': list(user_models),
                        'size_mb': user_size / (1024 * 1024)
                    }
            
            stats['models_used'] = list(stats['models_used'])
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}


class EmbeddingManager:
    """High-level embedding management interface."""
    
    def __init__(self, storage: EmbeddingStorage = None):
        """
        Initialize embedding manager.
        
        Args:
            storage: Embedding storage instance
        """
        self.storage = storage or EmbeddingStorage()
        self.similarity_threshold = 0.3
        self.quality_threshold = 0.5
    
    def add_user_embedding(self, user_id: str, embedding: FaceEmbedding, 
                          image_path: str = "", model_name: str = "") -> str:
        """
        Add new embedding for a user.
        
        Args:
            user_id: User identifier
            embedding: Face embedding
            image_path: Source image path
            model_name: Model name used
            
        Returns:
            Path to saved embedding
        """
        # Create metadata
        metadata = EmbeddingMetadata(
            user_id=user_id,
            image_path=image_path,
            model_name=model_name or embedding.model_name,
            quality_score=embedding.quality_score,
            confidence=embedding.confidence
        )
        
        # Save embedding
        return self.storage.save_embedding(user_id, embedding, metadata)
    
    def get_user_template(self, user_id: str, model_name: str = None) -> Optional[FaceEmbedding]:
        """
        Get representative template embedding for a user.
        
        Args:
            user_id: User identifier
            model_name: Filter by model name
            
        Returns:
            Template embedding or None
        """
        embeddings = self.storage.get_user_embeddings(user_id)
        
        if model_name:
            embeddings = [(e, m) for e, m in embeddings if m.model_name == model_name]
        
        if not embeddings:
            return None
        
        # Filter by quality threshold
        high_quality = [(e, m) for e, m in embeddings if m.quality_score >= self.quality_threshold]
        
        if high_quality:
            embeddings = high_quality
        
        # Return best quality embedding
        return embeddings[0][0] if embeddings else None
    
    def recognize_user(self, query_embedding: FaceEmbedding, 
                      candidate_users: List[str] = None) -> Tuple[Optional[str], float]:
        """
        Recognize user from query embedding.
        
        Args:
            query_embedding: Query face embedding
            candidate_users: List of candidate users (all users if None)
            
        Returns:
            Tuple of (user_id, similarity_score)
        """
        if candidate_users is None:
            candidate_users = self.storage.get_all_users()
        
        best_user = None
        best_similarity = 0.0
        
        for user_id in candidate_users:
            template = self.get_user_template(user_id, query_embedding.model_name)
            
            if template is None:
                continue
            
            similarity = query_embedding.similarity(template)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_user = user_id
        
        return best_user, best_similarity
    
    def update_user_embeddings(self, user_id: str, new_embeddings: List[FaceEmbedding], 
                              model_name: str = "", max_embeddings: int = 10) -> int:
        """
        Update user embeddings with new ones, keeping only the best quality.
        
        Args:
            user_id: User identifier
            new_embeddings: New embeddings to add
            model_name: Model name for new embeddings
            max_embeddings: Maximum embeddings to keep per user
            
        Returns:
            Number of embeddings added
        """
        added_count = 0
        
        for embedding in new_embeddings:
            if embedding.quality_score >= self.quality_threshold:
                try:
                    self.add_user_embedding(user_id, embedding, model_name=model_name)
                    added_count += 1
                except Exception as e:
                    logger.warning(f"Failed to add embedding for user {user_id}: {e}")
        
        # Clean up old embeddings if we exceed the limit
        self._cleanup_user_embeddings(user_id, max_embeddings)
        
        return added_count
    
    def _cleanup_user_embeddings(self, user_id: str, max_embeddings: int):
        """Remove old/low quality embeddings to stay within limit."""
        try:
            embeddings = self.storage.get_user_embeddings(user_id)
            
            if len(embeddings) <= max_embeddings:
                return
            
            # Keep only the best embeddings
            embeddings_to_remove = embeddings[max_embeddings:]
            
            user_dir = self.storage.base_path / user_id / 'embeddings'
            
            for _, metadata in embeddings_to_remove:
                # Find and remove embedding files
                for embedding_file in user_dir.glob(f"{metadata.model_name}_*.npy"):
                    if metadata.created_at in str(embedding_file):
                        embedding_file.unlink(missing_ok=True)
                        # Also remove metadata file
                        metadata_file = embedding_file.with_suffix('.json')
                        metadata_file.unlink(missing_ok=True)
                        break
            
            logger.info(f"Cleaned up {len(embeddings_to_remove)} old embeddings for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup embeddings for user {user_id}: {e}")
    
    def get_similarity_matrix(self, user_ids: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix between users.
        
        Args:
            user_ids: List of user identifiers
            
        Returns:
            Similarity matrix
        """
        n = len(user_ids)
        similarity_matrix = np.zeros((n, n))
        
        # Get templates for all users
        templates = {}
        for user_id in user_ids:
            template = self.get_user_template(user_id)
            if template:
                templates[user_id] = template
        
        # Calculate similarities
        for i, user_i in enumerate(user_ids):
            for j, user_j in enumerate(user_ids):
                if user_i in templates and user_j in templates:
                    similarity = templates[user_i].similarity(templates[user_j])
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def set_thresholds(self, similarity_threshold: float = None, 
                      quality_threshold: float = None):
        """Update recognition thresholds."""
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        if quality_threshold is not None:
            self.quality_threshold = quality_threshold
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics."""
        storage_stats = self.storage.get_storage_stats()
        
        return {
            'storage_stats': storage_stats,
            'similarity_threshold': self.similarity_threshold,
            'quality_threshold': self.quality_threshold,
            'total_users': storage_stats.get('total_users', 0),
            'total_embeddings': storage_stats.get('total_embeddings', 0)
        }