"""
Main fact checker that orchestrates the entire verification process.
Combines evidence retrieval, NLI classification, and verdict aggregation.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from data_sources.base_connector import EvidenceItem
from retrieval.evidence_retriever import EvidenceRetriever
from .nli_classifier import NLIClassifier, NLIResult
from .verdict_aggregator import VerdictAggregator, Verdict
from config import CONFIG

class FactChecker:
    """Main fact-checking orchestrator."""
    
    def __init__(self):
        """Initialize the fact checker with all components."""
        print("Initializing Fact Checker...")
        
        # Initialize components
        self.evidence_retriever = EvidenceRetriever()
        self.nli_classifier = NLIClassifier()
        self.verdict_aggregator = VerdictAggregator()
        
        print("Fact Checker initialized successfully")
    
    async def check_claim(self, claim: str, max_evidence_per_source: int = None,
                         show_steps: bool = False) -> Dict[str, Any]:
        """
        Check a claim and return a comprehensive fact-check result.
        
        Args:
            claim: The claim to fact-check
            max_evidence_per_source: Maximum evidence items per source
            show_steps: Whether to include step-by-step reasoning
            
        Returns:
            Dictionary with fact-check results
        """
        start_time = time.time()
        max_evidence = max_evidence_per_source or CONFIG.retrieval.max_results_per_source
        
        print(f"\n{'='*60}")
        print(f"FACT-CHECKING CLAIM: {claim}")
        print(f"{'='*60}")
        
        steps = [] if show_steps else None
        
        try:
            # Step 1: Retrieve evidence
            if show_steps:
                steps.append("Step 1: Retrieving evidence from multiple sources...")
            
            evidence_items = await self.evidence_retriever.retrieve_evidence(
                claim, max_evidence
            )
            
            if show_steps:
                steps.append(f"Found {len(evidence_items)} evidence items from {len(set(item.source for item in evidence_items))} sources")
            
            if not evidence_items:
                return self._create_no_evidence_result(claim, steps)
            
            # Step 2: NLI Classification
            if show_steps:
                steps.append("Step 2: Analyzing evidence using Natural Language Inference...")
            
            nli_results = self._classify_evidence(claim, evidence_items)
            
            if show_steps:
                supporting = sum(1 for r in nli_results if r.supports_claim())
                refuting = sum(1 for r in nli_results if r.refutes_claim())
                neutral = sum(1 for r in nli_results if r.is_neutral())
                steps.append(f"NLI Analysis: {supporting} supporting, {refuting} refuting, {neutral} neutral")
            
            # Step 3: Aggregate verdict
            if show_steps:
                steps.append("Step 3: Aggregating evidence and computing final verdict...")
            
            verdict = self.verdict_aggregator.aggregate_verdict(claim, evidence_items, nli_results)
            
            if show_steps:
                steps.append(f"Final verdict: {verdict.label.value} (confidence: {verdict.confidence:.1%})")
            
            # Step 4: Prepare result
            total_time = time.time() - start_time
            result = self._create_result(claim, verdict, total_time, steps)
            
            print(f"\nFACT-CHECK COMPLETE: {verdict.label.value} ({verdict.confidence:.1%} confidence)")
            print(f"Processing time: {total_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during fact-checking: {str(e)}"
            print(f"ERROR: {error_msg}")
            
            return {
                "claim": claim,
                "verdict": "Error",
                "confidence": 0.0,
                "evidence": [],
                "reasoning": error_msg,
                "processing_time": time.time() - start_time,
                "steps": steps,
                "error": True
            }
    
    def _classify_evidence(self, claim: str, evidence_items: List[EvidenceItem]) -> List[NLIResult]:
        """
        Classify evidence items using NLI model.
        
        Args:
            claim: The claim to verify
            evidence_items: List of evidence items
            
        Returns:
            List of NLI classification results
        """
        if not evidence_items:
            return []
        
        print(f"Classifying {len(evidence_items)} evidence items with NLI model...")
        
        # Extract evidence texts
        evidence_texts = [item.text for item in evidence_items]
        
        # Classify using NLI model
        nli_results = self.nli_classifier.verify_claim_against_evidence(claim, evidence_texts)
        
        # Log results
        for i, (evidence, nli_result) in enumerate(zip(evidence_items, nli_results)):
            print(f"Evidence {i+1} ({evidence.source}): {nli_result.label} ({nli_result.confidence:.2f})")
        
        return nli_results
    
    def _create_result(self, claim: str, verdict: Verdict, processing_time: float,
                      steps: Optional[List[str]]) -> Dict[str, Any]:
        """
        Create the final result dictionary.
        
        Args:
            claim: Original claim
            verdict: Verdict object
            processing_time: Time taken for processing
            steps: Optional list of processing steps
            
        Returns:
            Result dictionary
        """
        # Get top evidence for display
        top_evidence = self.verdict_aggregator.get_top_evidence(verdict, max_items=3)
        
        # Format evidence for display
        evidence_display = []
        for assessment in top_evidence:
            evidence_display.append({
                "text": assessment.evidence_item.text[:300] + "..." if len(assessment.evidence_item.text) > 300 else assessment.evidence_item.text,
                "source": assessment.evidence_item.source,
                "title": assessment.evidence_item.title,
                "url": assessment.evidence_item.url,
                "nli_label": assessment.nli_result.label,
                "nli_confidence": assessment.nli_result.confidence,
                "credibility_score": assessment.evidence_item.credibility_score,
                "supports_claim": assessment.supports_claim,
                "refutes_claim": assessment.refutes_claim
            })
        
        return {
            "claim": claim,
            "verdict": verdict.label.value,
            "confidence": verdict.confidence,
            "evidence": evidence_display,
            "reasoning": verdict.reasoning,
            "processing_time": processing_time,
            "steps": steps,
            "metadata": verdict.metadata,
            "error": False,
            "total_evidence_found": len(verdict.supporting_evidence) + len(verdict.refuting_evidence) + len(verdict.neutral_evidence)
        }
    
    def _create_no_evidence_result(self, claim: str, steps: Optional[List[str]]) -> Dict[str, Any]:
        """
        Create result when no evidence is found.
        
        Args:
            claim: Original claim
            steps: Optional list of processing steps
            
        Returns:
            Result dictionary for no evidence case
        """
        if steps:
            steps.append("No evidence found - returning unverified verdict")
        
        return {
            "claim": claim,
            "verdict": "Unverified",
            "confidence": 0.0,
            "evidence": [],
            "reasoning": "No evidence found from available sources",
            "processing_time": 0.0,
            "steps": steps,
            "metadata": {"total_evidence": 0},
            "error": False,
            "total_evidence_found": 0
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the fact-checking system.
        
        Returns:
            Dictionary with system information
        """
        return {
            "retrieval_stats": self.evidence_retriever.get_retrieval_stats(),
            "nli_model_info": self.nli_classifier.get_model_info(),
            "available_sources": self.evidence_retriever.get_available_sources(),
            "configuration": {
                "true_threshold": CONFIG.true_threshold,
                "false_threshold": CONFIG.false_threshold,
                "max_results_per_source": CONFIG.retrieval.max_results_per_source,
                "similarity_threshold": CONFIG.retrieval.similarity_threshold
            }
        }
    
    async def close(self):
        """Close all components and clean up resources."""
        print("Shutting down Fact Checker...")
        await self.evidence_retriever.close()
        print("Fact Checker shutdown complete")



