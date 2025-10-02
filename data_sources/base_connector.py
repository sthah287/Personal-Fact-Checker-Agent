"""
Base connector class for data sources.
Defines the interface that all data source connectors must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class EvidenceItem:
    """Represents a piece of evidence retrieved from a data source."""
    
    text: str
    source: str
    title: str
    url: Optional[str] = None
    publication_date: Optional[datetime] = None
    credibility_score: float = 0.5
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}
        
        # Truncate text if too long
        if len(self.text) > 1000:
            self.text = self.text[:997] + "..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "credibility_score": self.credibility_score,
            "metadata": self.metadata
        }

class BaseConnector(ABC):
    """Abstract base class for all data source connectors."""
    
    def __init__(self, source_name: str, base_url: str):
        """
        Initialize the connector.
        
        Args:
            source_name: Name of the data source (e.g., "wikipedia", "arxiv")
            base_url: Base URL for the API
        """
        self.source_name = source_name
        self.base_url = base_url
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[EvidenceItem]:
        """
        Search for evidence related to the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of EvidenceItem objects
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the data source is currently available.
        
        Returns:
            True if available, False otherwise
        """
        pass
    
    def get_source_info(self) -> Dict[str, str]:
        """
        Get information about this data source.
        
        Returns:
            Dictionary with source metadata
        """
        return {
            "name": self.source_name,
            "base_url": self.base_url,
            "type": self.__class__.__name__
        }



