"""
Semantic search engine using FAISS for efficient similarity search.
Provides fast retrieval of relevant evidence based on semantic similarity.
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
import pickle
import os
from dataclasses import asdict

from data_sources.base_connector import EvidenceItem
from .embedding_manager import EmbeddingManager
from config import CONFIG

class SemanticSearchEngine:
    """FAISS-based semantic search engine for evidence retrieval."""
    
    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        """
        Initialize the semantic search engine.
        
        Args:
            embedding_manager: EmbeddingManager instance for generating embeddings
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.index = None
        self.evidence_items = []
        self.embeddings = None
        
        # FAISS index configuration
        self.index_type = CONFIG.retrieval.faiss_index_type
        self.embedding_dim = self.embedding_manager.embedding_dim
        
        # Initialize empty index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize an empty FAISS index."""
        try:
            if self.index_type == "IndexFlatIP":
                # Inner product index (good for normalized embeddings/cosine similarity)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == "IndexFlatL2":
                # L2 distance index
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                # Default to inner product
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            print(f"Initialized FAISS index: {self.index_type}")
            
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
            raise
    
    def add_evidence(self, evidence_items: List[EvidenceItem], 
                    batch_size: int = 32) -> bool:
        """
        Add evidence items to the search index.
        
        Args:
            evidence_items: List of EvidenceItem objects to add
            batch_size: Batch size for embedding generation
            
        Returns:
            True if successful, False otherwise
        """
        if not evidence_items:
            return True
        
        try:
            print(f"Adding {len(evidence_items)} evidence items to search index...")
            
            # Extract texts for embedding
            texts = [item.text for item in evidence_items]
            
            # Generate embeddings
            new_embeddings = self.embedding_manager.encode_texts(
                texts, batch_size=batch_size, show_progress=True
            )
            
            if len(new_embeddings) == 0:
                print("No embeddings generated")
                return False
            
            # Add to FAISS index
            self.index.add(new_embeddings.astype(np.float32))
            
            # Store evidence items and embeddings
            self.evidence_items.extend(evidence_items)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            print(f"Successfully added {len(evidence_items)} items. Total items: {len(self.evidence_items)}")
            return True
            
        except Exception as e:
            print(f"Error adding evidence to index: {e}")
            return False
    
    def search(self, query: str, k: int = 5, 
              similarity_threshold: float = None) -> List[Tuple[EvidenceItem, float]]:
        """
        Search for evidence items similar to the query.
        
        Args:
            query: Search query string
            k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (EvidenceItem, similarity_score) tuples, sorted by similarity
        """
        if not self.evidence_items:
            print("No evidence items in index")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.encode_single_text(query)
            
            if query_embedding is None or len(query_embedding) == 0:
                print("Failed to generate query embedding")
                return []
            
            # Search in FAISS index
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            similarities, indices = self.index.search(query_embedding, min(k, len(self.evidence_items)))
            
            # Process results
            results = []
            threshold = similarity_threshold or CONFIG.retrieval.similarity_threshold
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and idx < len(self.evidence_items):
                    # Convert inner product back to cosine similarity if needed
                    if self.index_type == "IndexFlatIP":
                        similarity_score = float(similarity)
                    else:
                        # For L2 distance, convert to similarity
                        similarity_score = 1.0 / (1.0 + float(similarity))
                    
                    if similarity_score >= threshold:
                        results.append((self.evidence_items[idx], similarity_score))
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Found {len(results)} relevant evidence items for query")
            return results
            
        except Exception as e:
            print(f"Error searching index: {e}")
            return []
    
    def search_by_source(self, query: str, source_filter: str, k: int = 5) -> List[Tuple[EvidenceItem, float]]:
        """
        Search for evidence items from a specific source.
        
        Args:
            query: Search query string
            source_filter: Source name to filter by (e.g., "wikipedia", "arxiv")
            k: Number of top results to return
            
        Returns:
            List of (EvidenceItem, similarity_score) tuples from the specified source
        """
        # Get all results first
        all_results = self.search(query, k=len(self.evidence_items))
        
        # Filter by source
        filtered_results = [
            (item, score) for item, score in all_results 
            if item.source.lower() == source_filter.lower()
        ]
        
        return filtered_results[:k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the search index.
        
        Returns:
            Dictionary with index statistics
        """
        source_counts = {}
        if self.evidence_items:
            for item in self.evidence_items:
                source_counts[item.source] = source_counts.get(item.source, 0) + 1
        
        return {
            "total_items": len(self.evidence_items),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "sources": source_counts,
            "index_size": self.index.ntotal if self.index else 0
        }
    
    def clear_index(self):
        """Clear all evidence items and reset the index."""
        self.evidence_items = []
        self.embeddings = None
        self._initialize_index()
        print("Search index cleared")
    
    def save_index(self, filepath: str) -> bool:
        """
        Save the search index to disk.
        
        Args:
            filepath: Path to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save evidence items and metadata
            with open(f"{filepath}.pkl", "wb") as f:
                pickle.dump({
                    "evidence_items": [asdict(item) for item in self.evidence_items],
                    "embeddings": self.embeddings,
                    "index_type": self.index_type,
                    "embedding_dim": self.embedding_dim
                }, f)
            
            print(f"Index saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load a search index from disk.
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load evidence items and metadata
            with open(f"{filepath}.pkl", "rb") as f:
                data = pickle.load(f)
            
            # Reconstruct evidence items
            self.evidence_items = [
                EvidenceItem(**item_dict) for item_dict in data["evidence_items"]
            ]
            self.embeddings = data["embeddings"]
            self.index_type = data["index_type"]
            self.embedding_dim = data["embedding_dim"]
            
            print(f"Index loaded from {filepath}")
            print(f"Loaded {len(self.evidence_items)} evidence items")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False



