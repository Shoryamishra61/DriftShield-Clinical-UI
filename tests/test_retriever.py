"""
DriftShield FAISS Retriever Unit Tests.

Verifies indexing, normalization, nearest-neighbor searching, and save/load logic.
"""

import numpy as np
import pytest
from pathlib import Path
from rag.retriever import FAISSRetriever, RetrievedChunk

def test_retriever_build_and_retrieve(tmp_path: Path) -> None:
    """Tests that the FAISSRetriever indexes correctly, searches, and saves/loads state.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    retriever = FAISSRetriever(embedding_dim=4)
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    metadata = [
        {"text": "Guideline A", "domain": "cardiology", "year": 2022, "source_name": "AHA", "source_doc_id": "CARD_001", "chunk_index": 0},
        {"text": "Guideline B", "domain": "oncology", "year": 2023, "source_name": "NCCN", "source_doc_id": "ONCO_001", "chunk_index": 0},
        {"text": "Guideline C", "domain": "diabetes", "year": 2024, "source_name": "ADA", "source_doc_id": "DIAB_001", "chunk_index": 0},
    ]
    retriever.build_index(embeddings, metadata)
    assert retriever.index is not None
    assert retriever.index.ntotal == 3
    
    # Retrieve top 1 closest to first document
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = retriever.retrieve(query, top_k=1)
    assert len(results) == 1
    assert results[0].text == "Guideline A"
    assert results[0].domain == "cardiology"
    
    # Save and load verification
    retriever.save(tmp_path)
    new_retriever = FAISSRetriever(embedding_dim=4)
    new_retriever.load(tmp_path)
    assert new_retriever.index is not None
    assert new_retriever.index.ntotal == 3
    
    # Run retrieval on loaded index
    loaded_results = new_retriever.retrieve(query, top_k=1)
    assert len(loaded_results) == 1
    assert loaded_results[0].text == "Guideline A"
