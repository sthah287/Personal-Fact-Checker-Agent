"""
Configuration module for the Personal Fact-Checker Agent.
Contains API endpoints, model configurations, and system settings.
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
    embedding_model: str = "all-MiniLM-L6-v2"
    nli_model: str = "microsoft/DialoGPT-medium"  # We'll use RoBERTa-MNLI instead
    nli_model_actual: str = "roberta-large-mnli"
    device: str = "cpu"  # Change to "cuda" if GPU available
    max_sequence_length: int = 512

@dataclass
class RetrievalConfig:
    """Configuration for evidence retrieval."""
    max_results_per_source: int = 5
    similarity_threshold: float = 0.3
    max_evidence_length: int = 500
    faiss_index_type: str = "IndexFlatIP"  # Inner product for cosine similarity

@dataclass
class APIConfig:
    """Configuration for external APIs."""
    wikipedia_api_url: str = "https://en.wikipedia.org/api/rest_v1"
    arxiv_api_url: str = "http://export.arxiv.org/api/query"
    news_rss_feeds: List[str] = field(default_factory=lambda: [
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.reuters.com/reuters/topNews",
        "https://rss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml"
    ])

@dataclass
class UIConfig:
    """Configuration for user interfaces."""
    gradio_port: int = 7860
    gradio_share: bool = False
    max_claim_length: int = 1000
    show_confidence_threshold: float = 0.6

@dataclass
class SystemConfig:
    """Main system configuration."""
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    apis: APIConfig = field(default_factory=APIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Verdict thresholds
    true_threshold: float = 0.7
    false_threshold: float = 0.7
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

