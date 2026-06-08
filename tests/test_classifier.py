"""
DriftShield Classifier Unit Tests.

Verifies the preparation of tokenized tensors and checks classifier instantiations.
"""

import pytest
import torch
from pathlib import Path
from models.classifier import DriftShieldClassifier, DriftPrediction

def test_classifier_prepare_inputs() -> None:
    """Verifies that prepare_inputs tokenizes the query and context into the expected keys."""
    # Use CPU for testing
    classifier = DriftShieldClassifier(device="cpu")
    assert classifier.device == "cpu"
    
    inputs = classifier.prepare_inputs("Query text", "Context guideline text")
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert isinstance(inputs["input_ids"], torch.Tensor)
    assert inputs["input_ids"].shape[1] == classifier.MAX_LENGTH
