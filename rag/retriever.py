"""
DriftShield FAISS Retriever Module.

Sets up vector indices using Meta's FAISS library to retrieve relevant guideline context chunks.
Uses inner product (cosine similarity) metrics on normalized embeddings.
"""

import json
import numpy as np
import faiss
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

@dataclass
class RetrievedChunk:
    """Dataclass holding retrieved chunk information and its similarity score.

    Attributes:
        text: Segment text content.
        score: Cosine similarity score.
        domain: Medical domain.
        year: Year of guidelines.
        source_name: Organization name.
        source_doc_id: Source document ID.
        chunk_index: Index of chunk in source document.
    """
    text: str
    score: float
    domain: str
    year: int
    source_name: str
    source_doc_id: str
    chunk_index: int


class FAISSRetriever:
    """Retriever engine that manages building, loading, saving, and searching a FAISS Index."""

    def __init__(self, embedding_dim: int = 768) -> None:
        """Initializes the FAISSRetriever.

        Args:
            embedding_dim: Dimension of embeddings (default 768 for BioBERT).
        """
        self.embedding_dim: int = embedding_dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Constructs a FAISS index from document embeddings and stores metadata.

        Args:
            embeddings: Numpy array of shape (N, 768) containing document vectors.
            metadata: List of metadata dictionaries mapping directly to indexed items.
        """
        assert embeddings.shape[1] == self.embedding_dim, f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
        normalized = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(normalized)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(normalized)
        self.metadata = metadata

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        """Performs semantic search to find top_k closest segments.

        Args:
            query_embedding: Numpy array of shape (768,) representing the query.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of RetrievedChunk instances.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        q = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(RetrievedChunk(
                text=meta["text"],
                score=float(score),
                domain=meta["domain"],
                year=meta["year"],
                source_name=meta["source_name"],
                source_doc_id=meta["source_doc_id"],
                chunk_index=meta["chunk_index"],
            ))
        return results

    def save(self, index_dir: Path) -> None:
        """Saves index binary and JSON metadata to disk.

        Args:
            index_dir: Directory path where output will be saved.
        """
        if self.index is None:
            raise RuntimeError("Index is not initialized. Cannot save.")
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "guidelines.faiss"))
        with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, index_dir: Path) -> None:
        """Loads index binary and JSON metadata from disk.

        Args:
            index_dir: Directory path containing guidelines.faiss and metadata.json.
        """
        self.index = faiss.read_index(str(index_dir / "guidelines.faiss"))
        with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        # Verify dimension match
        assert self.index.d == self.embedding_dim, f"Loaded index dimension mismatch: index={self.index.d}, expected={self.embedding_dim}"
