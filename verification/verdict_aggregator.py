"""
Verdict aggregator for combining NLI results and source credibility into final verdicts.
Implements logic for aggregating evidence and computing confidence scores.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from data_sources.base_connector import EvidenceItem
from .nli_classifier import NLIResult
from config import CONFIG, get_source_credibility

class VerdictLabel(Enum):
    """Possible verdict labels."""
    LIKELY_TRUE = "Likely True"
    LIKELY_FALSE = "Likely False"
    UNVERIFIED = "Unverified"

@dataclass
class EvidenceAssessment:
    """Assessment of a single piece of evidence."""
    evidence_item: EvidenceItem
    nli_result: NLIResult
    weighted_score: float
    supports_claim: bool
    refutes_claim: bool

@dataclass
class Verdict:
    """Final verdict for a claim."""
    label: VerdictLabel
    confidence: float
    supporting_evidence: List[EvidenceAssessment]
    refuting_evidence: List[EvidenceAssessment]
    neutral_evidence: List[EvidenceAssessment]
    reasoning: str
    metadata: Dict[str, Any]

class VerdictAggregator:
    """Aggregates NLI results and evidence credibility into final verdicts."""
    
    def __init__(self):
        """Initialize the verdict aggregator."""
        self.true_threshold = CONFIG.true_threshold
        self.false_threshold = CONFIG.false_threshold
        self.confidence_weights = CONFIG.confidence_weights
    
    def aggregate_verdict(self, claim: str, evidence_items: List[EvidenceItem], 
                         nli_results: List[NLIResult]) -> Verdict:
        """
        Aggregate evidence and NLI results into a final verdict.
        
        Args:
            claim: The original claim
            evidence_items: List of evidence items
            nli_results: List of NLI classification results
            
        Returns:
            Verdict object with final assessment
        """
        if len(evidence_items) != len(nli_results):
            raise ValueError("Number of evidence items must match number of NLI results")
        
        if not evidence_items:
            return self._create_unverified_verdict(claim, "No evidence found")
        
        # Step 1: Create evidence assessments
        assessments = self._create_evidence_assessments(evidence_items, nli_results)
        
        # Step 2: Categorize evidence
        supporting = [a for a in assessments if a.supports_claim]
        refuting = [a for a in assessments if a.refutes_claim]
        neutral = [a for a in assessments if not a.supports_claim and not a.refutes_claim]
        
        # Step 3: Calculate aggregate scores
        support_score = self._calculate_aggregate_score(supporting)
        refute_score = self._calculate_aggregate_score(refuting)
        
        # Step 4: Determine verdict
        verdict_label, confidence, reasoning = self._determine_verdict(
            support_score, refute_score, len(assessments)
        )
        
        # Step 5: Create metadata
        metadata = self._create_metadata(assessments, support_score, refute_score)
        
        return Verdict(
            label=verdict_label,
            confidence=confidence,
            supporting_evidence=supporting,
            refuting_evidence=refuting,
            neutral_evidence=neutral,
            reasoning=reasoning,
            metadata=metadata
        )
    
    def _create_evidence_assessments(self, evidence_items: List[EvidenceItem], 
                                   nli_results: List[NLIResult]) -> List[EvidenceAssessment]:
        """
        Create evidence assessments by combining evidence items with NLI results.
        
        Args:
            evidence_items: List of evidence items
            nli_results: List of NLI results
            
        Returns:
            List of EvidenceAssessment objects
        """
        assessments = []
        
        for evidence, nli_result in zip(evidence_items, nli_results):
            # Calculate weighted score based on NLI confidence and source credibility
            nli_confidence = nli_result.confidence
            source_credibility = evidence.credibility_score
            
            # Weighted combination of NLI confidence and source credibility
            weighted_score = (0.7 * nli_confidence + 0.3 * source_credibility)
            
            # Adjust score based on NLI label
            if nli_result.refutes_claim():
                weighted_score = -weighted_score  # Negative for refuting evidence
            elif nli_result.is_neutral():
                weighted_score = 0.0  # Neutral evidence doesn't contribute
            
            assessment = EvidenceAssessment(
                evidence_item=evidence,
                nli_result=nli_result,
                weighted_score=weighted_score,
                supports_claim=nli_result.supports_claim(),
                refutes_claim=nli_result.refutes_claim()
            )
            
            assessments.append(assessment)
        
        return assessments
    
    def _calculate_aggregate_score(self, assessments: List[EvidenceAssessment]) -> float:
        """
        Calculate aggregate score for a set of evidence assessments.
        
        Args:
            assessments: List of evidence assessments
            
        Returns:
            Aggregate score (0-1)
        """
        if not assessments:
            return 0.0
        
        # Use weighted average of scores
        total_weight = 0.0
        weighted_sum = 0.0
        
        for assessment in assessments:
            weight = assessment.evidence_item.credibility_score
            score = abs(assessment.weighted_score)  # Use absolute value for aggregation
            
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        aggregate_score = weighted_sum / total_weight
        
        # Apply diminishing returns for multiple pieces of evidence
        num_evidence = len(assessments)
        if num_evidence > 1:
            # Reduce impact of additional evidence
            diminishing_factor = 1.0 - (0.1 * (num_evidence - 1))
            diminishing_factor = max(0.5, diminishing_factor)  # Don't reduce below 50%
            aggregate_score *= diminishing_factor
        
        return min(1.0, aggregate_score)
    
    def _determine_verdict(self, support_score: float, refute_score: float, 
                          total_evidence: int) -> Tuple[VerdictLabel, float, str]:
        """
        Determine the final verdict based on support and refute scores.
        
        Args:
            support_score: Aggregate support score
            refute_score: Aggregate refute score
            total_evidence: Total number of evidence pieces
            
        Returns:
            Tuple of (verdict_label, confidence, reasoning)
        """
        # Calculate net score (positive = supporting, negative = refuting)
        net_score = support_score - refute_score
        
        # Determine verdict based on thresholds
        if support_score >= self.true_threshold and net_score > 0.2:
            verdict_label = VerdictLabel.LIKELY_TRUE
            confidence = min(0.95, support_score)
            reasoning = f"Strong supporting evidence found (support: {support_score:.2f}, refute: {refute_score:.2f})"
            
        elif refute_score >= self.false_threshold and net_score < -0.2:
            verdict_label = VerdictLabel.LIKELY_FALSE
            confidence = min(0.95, refute_score)
            reasoning = f"Strong refuting evidence found (support: {support_score:.2f}, refute: {refute_score:.2f})"
            
        else:
            verdict_label = VerdictLabel.UNVERIFIED
            
            # Calculate confidence for unverified verdict
            if total_evidence == 0:
                confidence = 0.0
                reasoning = "No evidence found to verify the claim"
            elif abs(net_score) < 0.1:
                confidence = 0.3 + (0.2 * total_evidence / 10)  # Low confidence due to conflicting evidence
                reasoning = f"Conflicting or insufficient evidence (support: {support_score:.2f}, refute: {refute_score:.2f})"
            else:
                confidence = 0.4
                reasoning = f"Insufficient evidence strength (support: {support_score:.2f}, refute: {refute_score:.2f})"
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return verdict_label, confidence, reasoning
    
    def _create_metadata(self, assessments: List[EvidenceAssessment], 
                        support_score: float, refute_score: float) -> Dict[str, Any]:
        """
        Create metadata dictionary with detailed information.
        
        Args:
            assessments: List of evidence assessments
            support_score: Aggregate support score
            refute_score: Aggregate refute score
            
        Returns:
            Metadata dictionary
        """
        # Source distribution
        source_counts = {}
        for assessment in assessments:
            source = assessment.evidence_item.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # NLI label distribution
        nli_labels = {}
        for assessment in assessments:
            label = assessment.nli_result.label
            nli_labels[label] = nli_labels.get(label, 0) + 1
        
        # Average confidence scores
        nli_confidences = [a.nli_result.confidence for a in assessments]
        source_credibilities = [a.evidence_item.credibility_score for a in assessments]
        
        return {
            "total_evidence": len(assessments),
            "support_score": support_score,
            "refute_score": refute_score,
            "net_score": support_score - refute_score,
            "source_distribution": source_counts,
            "nli_label_distribution": nli_labels,
            "avg_nli_confidence": np.mean(nli_confidences) if nli_confidences else 0.0,
            "avg_source_credibility": np.mean(source_credibilities) if source_credibilities else 0.0,
            "thresholds": {
                "true_threshold": self.true_threshold,
                "false_threshold": self.false_threshold
            }
        }
    
    def _create_unverified_verdict(self, claim: str, reason: str) -> Verdict:
        """
        Create an unverified verdict when no evidence is available.
        
        Args:
            claim: The original claim
            reason: Reason for unverified status
            
        Returns:
            Verdict object with unverified status
        """
        return Verdict(
            label=VerdictLabel.UNVERIFIED,
            confidence=0.0,
            supporting_evidence=[],
            refuting_evidence=[],
            neutral_evidence=[],
            reasoning=reason,
            metadata={
                "total_evidence": 0,
                "support_score": 0.0,
                "refute_score": 0.0,
                "net_score": 0.0,
                "source_distribution": {},
                "nli_label_distribution": {}
            }
        )
    
    def get_top_evidence(self, verdict: Verdict, max_items: int = 3) -> List[EvidenceAssessment]:
        """
        Get the top evidence items for display.
        
        Args:
            verdict: Verdict object
            max_items: Maximum number of evidence items to return
            
        Returns:
            List of top evidence assessments
        """
        all_evidence = (verdict.supporting_evidence + 
                       verdict.refuting_evidence + 
                       verdict.neutral_evidence)
        
        # Sort by absolute weighted score (strength of evidence)
        sorted_evidence = sorted(
            all_evidence, 
            key=lambda x: abs(x.weighted_score), 
            reverse=True
        )
        
        return sorted_evidence[:max_items]



