"""
DriftShield BioBERT Classifier.

This module defines the classification engine for semantic drift detection using BioBERT.
It processes a query and a retrieved guideline context to predict the probability of temporal drift.
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

@dataclass
class DriftPrediction:
    """Dataclass holding the prediction result for a semantic drift query.

    Attributes:
        drift_score: Probability that the query contains an outdated premise (RISKY).
        verdict: 'RISKY' or 'SAFE' depending on the score and threshold.
        confidence: The confidence score associated with the prediction.
        label: Binary classification label (1 for RISKY, 0 for SAFE).
    """
    drift_score: float
    verdict: str
    confidence: float
    label: int


class DriftShieldClassifier:
    """Classifier wrapper for the BioBERT-based semantic drift detection model.

    This class provides initialization, prediction, batch prediction, and save/load
    utilities for the sequence classification task.
    """
    MODEL_NAME: str = "dmis-lab/biobert-base-cased-v1.1"
    MAX_LENGTH: int = 256
    THRESHOLD: float = 0.50

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = "auto") -> None:
        """Initializes the DriftShieldClassifier.

        Args:
            checkpoint_path: Path to the directory containing a saved model checkpoint.
                             If None, loads the default pre-trained BioBERT model.
            device: Computing device to run the model on. 'auto' selects CUDA if available.
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        model_source = str(checkpoint_path) if checkpoint_path else self.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source, num_labels=2
        ).to(self.device)
        self.model.eval()

    def prepare_inputs(self, query: str, context: str) -> Dict[str, torch.Tensor]:
        """Prepares tokenized inputs for the sequence classification model.

        Args:
            query: The clinical query containing a potential outdated premise.
            context: The retrieved guideline document segment.

        Returns:
            Dict mapping tokenizer keys to PyTorch tensors.
        """
        return self.tokenizer(
            query,
            context,
            max_length=self.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict(self, query: str, context: str, threshold: Optional[float] = None) -> DriftPrediction:
        """Runs inference on a single query and guideline context.

        Args:
            query: The user clinical query.
            context: Retrieved guideline context text.
            threshold: Custom threshold for classifying as RISKY. Defaults to THRESHOLD.

        Returns:
            DriftPrediction containing prediction score, verdict, confidence, and label.
        """
        thresh = threshold if threshold is not None else self.THRESHOLD
        inputs = {k: v.to(self.device) for k, v in self.prepare_inputs(query, context).items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        drift_score = float(probs[1])
        label = int(drift_score >= thresh)
        return DriftPrediction(
            drift_score=drift_score,
            verdict="RISKY" if label == 1 else "SAFE",
            confidence=float(max(probs)),
            label=label,
        )

    @torch.no_grad()
    def predict_batch(self, pairs: List[Tuple[str, str]], threshold: Optional[float] = None) -> List[DriftPrediction]:
        """Runs batch inference on a list of query-context pairs.

        Args:
            pairs: A list of (query, context) string tuples.
            threshold: Custom threshold for classification. Defaults to THRESHOLD.

        Returns:
            List of DriftPrediction objects.
        """
        thresh = threshold if threshold is not None else self.THRESHOLD
        queries, contexts = zip(*pairs)
        inputs = self.tokenizer(
            list(queries), list(contexts),
            max_length=self.MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results = []
        for p in probs:
            drift_score = float(p[1])
            label = int(drift_score >= thresh)
            results.append(DriftPrediction(
                drift_score=drift_score,
                verdict="RISKY" if label == 1 else "SAFE",
                confidence=float(max(p)),
                label=label,
            ))
        return results

    def save(self, output_dir: Path) -> None:
        """Saves model weights and tokenizer to the output directory.

        Args:
            output_dir: Directory path to save model checkpoints.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

    @classmethod
    def load(cls, checkpoint_path: Path, device: str = "auto") -> "DriftShieldClassifier":
        """Loads a model classifier from a checkpoint directory.

        Args:
            checkpoint_path: Path to the directory containing a saved checkpoint.
            device: Device to place the loaded model on.

        Returns:
            DriftShieldClassifier loaded with the saved weights.
        """
        return cls(checkpoint_path=checkpoint_path, device=device)
