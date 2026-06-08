"""
DriftShield Production Query Logger.

This module implements a thread-safe logger that records incoming query metadata,
inference latency, drift scores, and retriever hits to a JSON-Lines file
to support MLOps monitoring and statistical calculations.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List, Optional

logger = logging.getLogger("query_logger")

class ProductionQueryLogger:
    """Thread-safe JSONL query logger for production telemetry."""
    
    _instance: Optional["ProductionQueryLogger"] = None
    _lock: Lock = Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "ProductionQueryLogger":
        """Singleton pattern to ensure thread safety across application components."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProductionQueryLogger, cls).__new__(cls)
            return cls._instance

    def __init__(self, log_dir: Path = Path("monitoring"), log_filename: str = "query_logs.jsonl") -> None:
        """Initializes the query logger.

        Args:
            log_dir: Directory where logs will be stored.
            log_filename: Name of the log file (.jsonl extension recommended).
        """
        # Ensure initialization runs only once
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_filename
        self.write_lock = Lock()
        self._initialized = True

    def log_query(
        self,
        query: str,
        biobert_score: float,
        qwen_score: float,
        hybrid_score: float,
        verdict: str,
        latency_ms: float,
        retrieved_sources: List[Dict[str, Any]],
        confidence: float,
        modalities: List[str] = ["text"]
    ) -> Dict[str, Any]:
        """Appends a new query telemetry record to the JSON-Lines file.

        Args:
            query: The user clinical query text.
            biobert_score: Semantic drift score from BioBERT.
            qwen_score: Semantic drift score from Qwen.
            hybrid_score: Combined ensemble drift score.
            verdict: The final decision ('SAFE' or 'RISKY').
            latency_ms: Inference pipeline latency in milliseconds.
            retrieved_sources: List of dictionaries representing retrieved chunks.
            confidence: Decision confidence.
            modalities: Modalities utilized (default ['text']).

        Returns:
            The logged dictionary record.
        """
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "biobert_score": float(biobert_score),
            "qwen_score": float(qwen_score),
            "hybrid_score": float(hybrid_score),
            "verdict": verdict,
            "latency_ms": float(latency_ms),
            "retrieved_sources": [
                {
                    "source_name": s.get("source_name", "Unknown"),
                    "year": int(s.get("year", 0)),
                    "domain": s.get("domain", "general"),
                    "score": float(s.get("score", 0.0))
                }
                for s in retrieved_sources
            ],
            "confidence": float(confidence),
            "modalities": modalities
        }

        # Write to JSONL in a thread-safe manner
        with self.write_lock:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                logger.error(f"Failed to write query telemetry record: {e}")
                
        return record

    def read_logs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Reads query logs from the persistent file.

        Args:
            limit: Maximum number of recent log records to return.

        Returns:
            List of dictionary telemetry records (ordered oldest to newest).
        """
        if not self.log_path.exists():
            return []
            
        records = []
        with self.write_lock:
            try:
                with open(self.log_path, "r") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to read query telemetry records: {e}")
                return []
                
        if limit is not None:
            return records[-limit:]
        return records
