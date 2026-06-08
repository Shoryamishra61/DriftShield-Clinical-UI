"""
DriftShield BioBERT Embedder Module.

Handles dense vector embedding generation for queries and guideline documents using BioBERT encoder.
Ensures L2-normalized embeddings for fast cosine similarity operations.
"""

import torch
import numpy as np
from typing import List, Union
from transformers import AutoModel, AutoTokenizer

class BioBERTEmbedder:
    """Embedder class that maps clinical text to 768-dimensional normalized dense vectors.

    Utilizes dmis-lab/biobert-base-cased-v1.1 as the encoder.
    """
    MODEL_NAME: str = "dmis-lab/biobert-base-cased-v1.1"
    MAX_LENGTH: int = 512
    EMBEDDING_DIM: int = 768

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32, device: str = "auto") -> None:
        """Initializes the BioBERTEmbedder.

        Args:
            model_name: HuggingFace model reference.
            batch_size: Batch size used for encoding collections of texts.
            device: Computing device ('cuda', 'cpu', or 'auto').
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encodes a single batch of texts to normalized embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            np.ndarray of shape (batch_size, 768) with L2-normalized values.
        """
        inputs = self.tokenizer(
            texts,
            max_length=self.MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Pool by taking the CLS token representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # L2-normalize
        norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return cls_embeddings / norms

    def embed(self, text: str) -> np.ndarray:
        """Encodes a single text string into a dense vector.

        Args:
            text: Input string.

        Returns:
            np.ndarray of shape (768,).
        """
        return self._encode_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Encodes multiple text strings into a stacked 2D array of vectors.

        Args:
            texts: List of input strings.

        Returns:
            np.ndarray of shape (len(texts), 768).
        """
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.append(self._encode_batch(batch))
        return np.vstack(all_embeddings)
