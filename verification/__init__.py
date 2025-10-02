"""
Verification package for claim verification using NLI models.
"""

from .nli_classifier import NLIClassifier
from .verdict_aggregator import VerdictAggregator
from .fact_checker import FactChecker

__all__ = [
    "NLIClassifier",
    "VerdictAggregator",
    "FactChecker"
]



