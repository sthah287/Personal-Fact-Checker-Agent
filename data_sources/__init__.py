"""
Data sources package for retrieving evidence from various APIs.
"""

from .wikipedia_connector import WikipediaConnector
from .arxiv_connector import ArxivConnector
from .base_connector import BaseConnector, EvidenceItem

__all__ = [
    "BaseConnector",
    "EvidenceItem", 
    "WikipediaConnector",
    "ArxivConnector"
]



