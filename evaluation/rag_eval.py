"""
DriftShield RAG Evaluation Suite (RAGAS-Lite).

This module defines evaluation metrics for the clinical RAG pipeline using
the local Qwen model as a judge. It evaluates context relevance,
faithfulness (groundedness), and answer relevance.
"""

import json
import logging
import requests
from typing import Dict, Any, List

logger = logging.getLogger("rag_evaluator")

class QwenRAGEvaluator:
    """Uses a local Qwen model via Ollama to evaluate medical RAG pipeline quality."""
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b", ollama_url: str = "http://localhost:11434/api/generate") -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url

    def _query_qwen(self, prompt: str, system_prompt: str = "") -> str:
        """Sends a request to local Ollama and returns the text response."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 150
            }
        }
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=15)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            logger.error(f"Error querying Qwen for RAG evaluation: {e}")
            return "{}"

    def evaluate_faithfulness(self, semantic_shift_explanation: str, retrieved_context: str) -> float:
        """Evaluates whether the system's explanation is grounded in the retrieved guideline (no hallucination).

        Returns:
            A score between 0.0 (not grounded) and 1.0 (fully grounded).
        """
        system = "You are a clinical peer reviewer validating medical claims. Output ONLY valid JSON."
        prompt = f"""
        Analyze if the Explanation is fully grounded in the retrieved Source Guideline text.
        Every factual claim in the Explanation must be directly derived from the Source Guideline.

        Source Guideline:
        "{retrieved_context}"

        Explanation:
        "{semantic_shift_explanation}"

        Output a JSON object with these exact keys:
        {{
          "reasoning": "Brief explanation of groundedness status",
          "grounded": true,  // true if every claim is supported, else false
          "faithfulness_score": 1.0  // Float value between 0.0 and 1.0
        }}
        """
        raw_resp = self._query_qwen(prompt, system)
        try:
            data = json.loads(raw_resp)
            return float(data.get("faithfulness_score", 0.0))
        except Exception:
            # Fallback parsing
            if '"grounded": true' in raw_resp or '"grounded":true' in raw_resp:
                return 1.0
            return 0.0

    def evaluate_context_relevance(self, query: str, retrieved_context: str) -> float:
        """Evaluates if the retrieved guideline context is relevant to the user's clinical query.

        Returns:
            A relevance score between 0.0 (irrelevant) and 1.0 (highly relevant).
        """
        system = "You are a clinical information retrieval evaluator. Output ONLY valid JSON."
        prompt = f"""
        Rate the clinical relevance of the retrieved Guideline Context in addressing the User Query.
        Does this guideline cover the exact clinical guidelines/reversals queried?

        User Query:
        "{query}"

        Guideline Context:
        "{retrieved_context}"

        Output a JSON object with these exact keys:
        {{
          "reasoning": "Brief analysis of topic alignment",
          "relevance_score": 0.85  // Float value between 0.0 (completely irrelevant) and 1.0 (perfectly relevant)
        }}
        """
        raw_resp = self._query_qwen(prompt, system)
        try:
            data = json.loads(raw_resp)
            return float(data.get("relevance_score", 0.0))
        except Exception:
            return 0.5

    def evaluate_context_precision(self, query: str, retrieved_chunks: List[str], ground_truth_sources: List[str]) -> float:
        """Evaluates context precision: are relevant documents placed higher in the retrieved list?

        Returns:
            Mean Precision score (Precision@K).
        """
        precisions = []
        hits = 0
        for i, chunk in enumerate(retrieved_chunks):
            # Check if chunk references any ground truth source organization (e.g. 'ADA', 'USPSTF')
            is_relevant = any(source.lower() in chunk.lower() for source in ground_truth_sources)
            if is_relevant:
                hits += 1
                precisions.append(hits / (i + 1))
        
        if not precisions:
            return 0.0
        return float(sum(precisions) / len(precisions))

    def run_pipeline_evaluation(
        self,
        query: str,
        semantic_shift: str,
        retrieved_texts: List[str],
        source_names: List[str]
    ) -> Dict[str, float]:
        """Runs the complete RAG evaluation suite for a single pipeline run.

        Args:
            query: The user clinical query.
            semantic_shift: The generated explanation or shift verdict description.
            retrieved_texts: List of retrieved context chunks.
            source_names: Ground truth sources/organizations targeted.

        Returns:
            A dictionary containing:
              - faithfulness
              - context_relevance
              - context_precision
              - joint_rag_score (mean of the three)
        """
        full_context = "\n\n".join(retrieved_texts)
        
        faithfulness = self.evaluate_faithfulness(semantic_shift, full_context)
        context_relevance = self.evaluate_context_relevance(query, full_context)
        context_precision = self.evaluate_context_precision(query, retrieved_texts, source_names)
        
        mean_score = float((faithfulness + context_relevance + context_precision) / 3.0)
        
        return {
            "faithfulness": faithfulness,
            "context_relevance": context_relevance,
            "context_precision": context_precision,
            "joint_rag_score": mean_score
        }
