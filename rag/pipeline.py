"""
DriftShield End-to-End Inference Pipeline Module.

Combines retrieval (Semantic Retrieval via BioBERT + FAISS) and classification
(Drift Classifier via BioBERT) with zero-shot Qwen classification,
production logging, and multimodal fusion.
"""

import json
import time
import requests
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from .chunker import SemanticChunker
from .embedder import BioBERTEmbedder
from .retriever import FAISSRetriever, RetrievedChunk
from models.classifier import DriftShieldClassifier, DriftPrediction
from monitoring.query_logger import ProductionQueryLogger
from models.multimodal import DriftShieldMultimodalEngine

@dataclass
class PipelineResult:
    """Dataclass holding aggregate results of a single query pipeline execution."""
    query: str
    drift_score: float  # The hybrid score
    biobert_score: float
    qwen_score: float
    verdict: str
    confidence: float
    semantic_shift: str
    retrieved_guidelines: List[RetrievedChunk]
    individual_scores: List[float]
    threshold_used: float
    latency_ms: float
    is_multimodal: bool = False
    image_path: Optional[str] = None

class DriftShieldPipeline:
    """End-to-End Hybrid & Multimodal Concept Drift Detection Pipeline."""
    DEFAULT_THRESHOLD: float = 0.50
    SAFETY_THRESHOLD: float = 0.30

    def __init__(
        self,
        classifier: DriftShieldClassifier,
        retriever: FAISSRetriever,
        embedder: BioBERTEmbedder,
        top_k: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.classifier = classifier
        self.retriever = retriever
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold
        # Initialize singleton logger
        self.query_logger = ProductionQueryLogger()
        # Initialize multimodal engine
        self.multimodal_engine = DriftShieldMultimodalEngine()

    def _query_qwen_classifier(self, query: str, context: str) -> Tuple[float, str]:
        """Queries the local Qwen model to perform zero-shot concept drift classification."""
        payload = {
            "model": "qwen2.5-coder:7b",
            "prompt": (
                f"Assess whether the patient's premise conflicts with current guidelines.\n"
                f"Patient Query: \"{query}\"\n"
                f"Current Guideline: \"{context}\"\n"
                f"Output a JSON object with these exact keys: "
                f"'drift_score' (float 0.0 to 1.0), 'verdict' ('RISKY' or 'SAFE'), 'explanation' (string)."
            ),
            "system": (
                "You are an elite clinical NLP classifier analyzing temporal concept drift. "
                "Output ONLY a raw valid JSON block."
            ),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 150
            }
        }
        try:
            resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=8)
            resp.raise_for_status()
            res_data = resp.json().get("response", "{}")
            data = json.loads(res_data)
            score = float(data.get("drift_score", 0.0))
            explanation = data.get("explanation", "Grounded assessment by Qwen.")
            return score, explanation
        except Exception:
            return 0.0, "Qwen offline or timed out."

    def __call__(self, query: str) -> PipelineResult:
        """Processes query through the complete retrieve, classify, and ensemble pipeline."""
        start_time = time.perf_counter()
        
        query_embedding = self.embedder.embed(query)
        retrieved = self.retriever.retrieve(query_embedding, top_k=self.top_k)

        if not retrieved:
            elapsed = (time.perf_counter() - start_time) * 1000
            result = PipelineResult(
                query=query,
                drift_score=0.0,
                biobert_score=0.0,
                qwen_score=0.0,
                verdict="UNKNOWN",
                confidence=0.0,
                semantic_shift="No relevant guidelines found in corpus.",
                retrieved_guidelines=[],
                individual_scores=[],
                threshold_used=self.threshold,
                latency_ms=elapsed
            )
            # Log query metadata
            self.query_logger.log_query(
                query=query, biobert_score=0.0, qwen_score=0.0, hybrid_score=0.0,
                verdict="UNKNOWN", latency_ms=elapsed, retrieved_sources=[], confidence=0.0
            )
            return result

        # 1. BioBERT sequence classification over retrieved guidelines
        pairs = [(query, chunk.text) for chunk in retrieved]
        predictions: List[DriftPrediction] = self.classifier.predict_batch(
            pairs, threshold=self.threshold
        )
        individual_scores = [p.drift_score for p in predictions]
        biobert_score = max(individual_scores)
        top_idx = individual_scores.index(biobert_score)
        top_chunk = retrieved[top_idx]

        # 2. Qwen zero-shot classification on the top retrieved context chunk
        qwen_score, qwen_explanation = self._query_qwen_classifier(query, top_chunk.text)

        # 3. Hybrid ensemble aggregation (safety-first max approach)
        hybrid_score = max(biobert_score, qwen_score)
        verdict = "RISKY" if hybrid_score >= self.threshold else "SAFE"
        confidence = float(abs(hybrid_score - 0.5) * 2)

        # Create explanation summary
        if verdict == "RISKY":
            semantic_shift = (
                f"**Conflict Detected with {top_chunk.source_name} ({top_chunk.year})**: "
                f"{qwen_explanation}"
            )
        else:
            semantic_shift = "No clinical concept drift detected. Query matches current consensus."

        elapsed = (time.perf_counter() - start_time) * 1000

        result = PipelineResult(
            query=query,
            drift_score=hybrid_score,
            biobert_score=biobert_score,
            qwen_score=qwen_score,
            verdict=verdict,
            confidence=confidence,
            semantic_shift=semantic_shift,
            retrieved_guidelines=retrieved,
            individual_scores=individual_scores,
            threshold_used=self.threshold,
            latency_ms=elapsed
        )

        # Format retrieved chunks for the log
        log_sources = [
            {"source_name": c.source_name, "year": c.year, "domain": c.domain, "score": c.score}
            for c in retrieved
        ]
        # Log query telemetry
        self.query_logger.log_query(
            query=query, biobert_score=biobert_score, qwen_score=qwen_score,
            hybrid_score=hybrid_score, verdict=verdict, latency_ms=elapsed,
            retrieved_sources=log_sources, confidence=confidence
        )

        return result

    def predict_multimodal(self, query: str, image_path: str) -> PipelineResult:
        """Processes clinical query text alongside a visual diagnostic image path.

        Fuses BioBERT and mock CLIP embeddings to predict joint concept drift.
        """
        start_time = time.perf_counter()
        
        # Resolve text path of pipeline first
        text_result = self(query)
        text_embedding = self.embedder.embed(query)
        
        # Simulate CLIP image embedding
        image_embedding = self.multimodal_engine.generate_simulated_image_embedding(image_path).numpy()
        
        # Perform multimodal fusion inference
        mm_res = self.multimodal_engine.predict_multimodal(text_embedding, image_embedding, self.threshold)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        # Aggregated result
        hybrid_score = max(text_result.drift_score, mm_res["drift_score"])
        verdict = "RISKY" if hybrid_score >= self.threshold else "SAFE"
        confidence = float(abs(hybrid_score - 0.5) * 2)
        
        result = PipelineResult(
            query=query,
            drift_score=hybrid_score,
            biobert_score=text_result.biobert_score,
            qwen_score=text_result.qwen_score,
            verdict=verdict,
            confidence=confidence,
            semantic_shift=f"[Multimodal Fusion (Text+CLIP)] {text_result.semantic_shift}",
            retrieved_guidelines=text_result.retrieved_guidelines,
            individual_scores=text_result.individual_scores,
            threshold_used=self.threshold,
            latency_ms=elapsed,
            is_multimodal=True,
            image_path=image_path
        )
        
        # Format sources
        log_sources = [
            {"source_name": c.source_name, "year": c.year, "domain": c.domain, "score": c.score}
            for c in text_result.retrieved_guidelines
        ]
        # Log to telemetry
        self.query_logger.log_query(
            query=query, biobert_score=text_result.biobert_score, qwen_score=text_result.qwen_score,
            hybrid_score=hybrid_score, verdict=verdict, latency_ms=elapsed,
            retrieved_sources=log_sources, confidence=confidence, modalities=["text", "image"]
        )
        
        return result

    @classmethod
    def from_checkpoints(
        cls,
        classifier_checkpoint: Path,
        index_dir: Path,
        top_k: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> "DriftShieldPipeline":
        embedder = BioBERTEmbedder()
        retriever = FAISSRetriever()
        retriever.load(index_dir)
        classifier = DriftShieldClassifier.load(classifier_checkpoint)
        return cls(classifier, retriever, embedder, top_k=top_k, threshold=threshold)
