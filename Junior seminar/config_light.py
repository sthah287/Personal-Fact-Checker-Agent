"""
Lightweight configuration for faster testing.
Uses smaller models for quicker startup.
"""

import os
from typing import Dict, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    embedding_model: str = "all-MiniLM-L12-v2"  # Smaller model
    nli_model_actual: str = "microsoft/DialoGPT-medium"  # Placeholder - we'll use simple logic
    device: str = "cpu"
    max_sequence_length: int = 256  # Shorter sequences

@dataclass
class RetrievalConfig:
    """Configuration for evidence retrieval."""
    max_results_per_source: int = 2  # Fewer results
    similarity_threshold: float = 0.2  # Lower threshold
    max_evidence_length: int = 200  # Shorter evidence
    faiss_index_type: str = "IndexFlatIP"

@dataclass
class APIConfig:
    """Configuration for external APIs."""
    wikipedia_api_url: str = "https://en.wikipedia.org/api/rest_v1"
    arxiv_api_url: str = "http://export.arxiv.org/api/query"
    news_rss_feeds: List[str] = field(default_factory=lambda: [])

@dataclass
class UIConfig:
    """Configuration for user interfaces."""
    gradio_port: int = 7860
    gradio_share: bool = False
    max_claim_length: int = 500
    show_confidence_threshold: float = 0.5

@dataclass
class SystemConfig:
    """Main system configuration."""
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    apis: APIConfig = field(default_factory=APIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # More lenient thresholds
    true_threshold: float = 0.6
    false_threshold: float = 0.6
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "wikipedia": 0.8,
        "arxiv": 0.9,
        "government": 0.85,
        "news": 0.6
    })

# Global configuration instance
CONFIG = SystemConfig()

# Source credibility mapping
SOURCE_CREDIBILITY = {
    "wikipedia": 0.8,
    "arxiv": 0.9,
    "cdc": 0.95,
    "data.gov": 0.9,
    "reuters": 0.8,
    "bbc": 0.8,
    "cnn": 0.7,
    "default": 0.5
}

def get_source_credibility(source_name: str) -> float:
    """Get credibility score for a given source."""
    source_lower = source_name.lower()
    for key, value in SOURCE_CREDIBILITY.items():
        if key in source_lower:
            return value
    return SOURCE_CREDIBILITY["default"]


