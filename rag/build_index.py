"""
DriftShield FAISS Index Builder.

Loads a guideline corpus file, chunks its items semantically,
generates BioBERT embeddings, and builds a FAISS search index stored on disk.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from .chunker import SemanticChunker
from .embedder import BioBERTEmbedder
from .retriever import FAISSRetriever

def build_faiss_index(
    corpus_path: Path = Path("data/guideline_corpus.json"),
    index_dir: Path = Path("rag/index"),
) -> None:
    """Builds and saves the FAISS search index for semantic retrieval.

    Args:
        corpus_path: Path to the JSON guideline corpus file.
        index_dir: Directory where the generated index and metadata will be saved.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    chunker = SemanticChunker()
    embedder = BioBERTEmbedder()
    retriever = FAISSRetriever()

    all_chunks = []
    for doc in corpus:
        chunks = chunker.chunk_document(doc["text"], doc)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_batch(texts)

    metadata: List[Dict[str, Any]] = [
        {
            "text": c.text,
            "domain": c.domain,
            "year": c.year,
            "source_name": c.source_name,
            "source_doc_id": c.source_doc_id,
            "chunk_index": c.chunk_index,
        }
        for c in all_chunks
    ]

    retriever.build_index(embeddings, metadata)
    retriever.save(index_dir)
    print(f"FAISS index saved to {index_dir}")

if __name__ == "__main__":
    build_faiss_index()
