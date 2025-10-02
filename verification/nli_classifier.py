"""
Natural Language Inference (NLI) classifier for claim verification.
Uses RoBERTa-MNLI model to classify claim-evidence pairs as supporting/refuting/unrelated.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

from config import CONFIG

@dataclass
class NLIResult:
    """Result of NLI classification."""
    label: str  # "ENTAILMENT", "CONTRADICTION", "NEUTRAL"
    confidence: float  # Confidence score (0-1)
    raw_scores: Dict[str, float]  # Raw logits for each class
    
    def supports_claim(self) -> bool:
        """Check if this result supports the claim."""
        return self.label == "ENTAILMENT"
    
    def refutes_claim(self) -> bool:
        """Check if this result refutes the claim."""
        return self.label == "CONTRADICTION"
    
    def is_neutral(self) -> bool:
        """Check if this result is neutral/unrelated."""
        return self.label == "NEUTRAL"

class NLIClassifier:
    """RoBERTa-MNLI based classifier for natural language inference."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the NLI classifier.
        
        Args:
            model_name: Name of the NLI model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name or CONFIG.models.nli_model_actual
        self.device = device or CONFIG.models.device
        self.tokenizer = None
        self.model = None
        
        # Label mapping for RoBERTa-MNLI
        self.label_mapping = {
            0: "CONTRADICTION",  # Refutes the claim
            1: "NEUTRAL",        # Unrelated/insufficient evidence
            2: "ENTAILMENT"      # Supports the claim
        }
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the RoBERTa-MNLI model and tokenizer."""
        try:
            print(f"Loading NLI model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"NLI model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading NLI model: {e}")
            raise
    
    def classify_single(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Classify a single premise-hypothesis pair.
        
        Args:
            premise: The evidence text (premise)
            hypothesis: The claim to verify (hypothesis)
            
        Returns:
            NLIResult with classification and confidence
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                padding=True,
                max_length=CONFIG.models.max_sequence_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=-1)
            probs_np = probabilities.cpu().numpy()[0]
            
            # Get predicted label and confidence
            predicted_idx = np.argmax(probs_np)
            predicted_label = self.label_mapping[predicted_idx]
            confidence = float(probs_np[predicted_idx])
            
            # Create raw scores dictionary
            raw_scores = {
                self.label_mapping[i]: float(probs_np[i]) 
                for i in range(len(probs_np))
            }
            
            return NLIResult(
                label=predicted_label,
                confidence=confidence,
                raw_scores=raw_scores
            )
            
        except Exception as e:
            print(f"Error in NLI classification: {e}")
            # Return neutral result on error
            return NLIResult(
                label="NEUTRAL",
                confidence=0.0,
                raw_scores={"CONTRADICTION": 0.33, "NEUTRAL": 0.34, "ENTAILMENT": 0.33}
            )
    
    def classify_batch(self, premise_hypothesis_pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """
        Classify multiple premise-hypothesis pairs in batch.
        
        Args:
            premise_hypothesis_pairs: List of (premise, hypothesis) tuples
            
        Returns:
            List of NLIResult objects
        """
        if not premise_hypothesis_pairs:
            return []
        
        try:
            premises, hypotheses = zip(*premise_hypothesis_pairs)
            
            # Tokenize all pairs
            inputs = self.tokenizer(
                list(premises),
                list(hypotheses),
                truncation=True,
                padding=True,
                max_length=CONFIG.models.max_sequence_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=-1)
            probs_np = probabilities.cpu().numpy()
            
            # Process results
            results = []
            for i in range(len(probs_np)):
                predicted_idx = np.argmax(probs_np[i])
                predicted_label = self.label_mapping[predicted_idx]
                confidence = float(probs_np[i][predicted_idx])
                
                raw_scores = {
                    self.label_mapping[j]: float(probs_np[i][j]) 
                    for j in range(len(probs_np[i]))
                }
                
                results.append(NLIResult(
                    label=predicted_label,
                    confidence=confidence,
                    raw_scores=raw_scores
                ))
            
            return results
            
        except Exception as e:
            print(f"Error in batch NLI classification: {e}")
            # Return neutral results on error
            return [
                NLIResult(
                    label="NEUTRAL",
                    confidence=0.0,
                    raw_scores={"CONTRADICTION": 0.33, "NEUTRAL": 0.34, "ENTAILMENT": 0.33}
                )
                for _ in premise_hypothesis_pairs
            ]
    
    def verify_claim_against_evidence(self, claim: str, evidence_texts: List[str]) -> List[NLIResult]:
        """
        Verify a claim against multiple pieces of evidence.
        
        Args:
            claim: The claim to verify
            evidence_texts: List of evidence texts
            
        Returns:
            List of NLIResult objects, one for each evidence text
        """
        if not evidence_texts:
            return []
        
        # Create premise-hypothesis pairs (evidence as premise, claim as hypothesis)
        pairs = [(evidence, claim) for evidence in evidence_texts]
        
        return self.classify_batch(pairs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "label_mapping": self.label_mapping,
            "max_sequence_length": CONFIG.models.max_sequence_length
        }
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.tokenizer is not None



