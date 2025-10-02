"""
Embedding manager for generating and managing text embeddings.
Uses sentence-transformers for creating semantic embeddings.
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from config import CONFIG

class EmbeddingManager:
    """Manages text embeddings using sentence-transformers."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name or CONFIG.models.embedding_model
        self.device = device or CONFIG.models.device
        self.model = None
        self.embedding_dim = None
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            self.embedding_dim = test_embedding.shape[1]
            
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, 
                    show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Encode texts
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            return embeddings
            
        except Exception as e:
            print(f"Error encoding texts: {e}")
            return np.array([]).reshape(0, self.embedding_dim)
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            numpy array embedding with shape (embedding_dim,)
        """
        embeddings = self.encode_texts([text], show_progress=False)
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.embedding_dim)
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                           candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarities between a query embedding and multiple candidates.
        
        Args:
            query_embedding: Query embedding with shape (embedding_dim,)
            candidate_embeddings: Candidate embeddings with shape (n_candidates, embedding_dim)
            
        Returns:
            Array of similarity scores with shape (n_candidates,)
        """
        try:
            if len(candidate_embeddings) == 0:
                return np.array([])
            
            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Compute dot product (assuming normalized embeddings)
            similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
            
            return similarities
            
        except Exception as e:
            print(f"Error computing similarities: {e}")
            return np.zeros(len(candidate_embeddings))
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        
        # Truncate if too long
        max_length = CONFIG.models.max_sequence_length
        if len(text) > max_length:
            text = text[:max_length-3] + "..."
        
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "max_sequence_length": self.model.max_seq_length if self.model else None
        }
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None



