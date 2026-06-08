"""
DriftShield Multimodal Medical Drift Fusion.

This module implements a multimodal fusion model in PyTorch that integrates
clinical text embeddings (from BioBERT) and medical imaging embeddings (from CLIP)
using a cross-attention projection mechanism to detect multimodal concept drift.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple


class MultimodalProjection(nn.Module):
    """Projects disparate text and image embeddings into a shared dimension."""
    def __init__(self, text_dim: int = 768, image_dim: int = 512, joint_dim: int = 256):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text_proj(text_emb), self.image_proj(image_emb)


class CrossAttentionFusion(nn.Module):
    """Applies cross-attention between visual and text modalities to capture joint context."""
    def __init__(self, joint_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=joint_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(joint_dim)
        
    def forward(self, text_proj: torch.Tensor, image_proj: torch.Tensor) -> torch.Tensor:
        # Reshape to (batch, seq_len=1, dim) for MultiheadAttention
        q = text_proj.unsqueeze(1)
        k = image_proj.unsqueeze(1)
        v = image_proj.unsqueeze(1)
        
        attn_out, _ = self.mha(q, k, v)
        fused = self.layernorm(text_proj + attn_out.squeeze(1))
        return fused


class MultimodalDriftFusionModel(nn.Module):
    """The complete multimodal drift classifier integrating BioBERT and CLIP."""
    def __init__(self, text_dim: int = 768, image_dim: int = 512, joint_dim: int = 256):
        super().__init__()
        self.projection = MultimodalProjection(text_dim, image_dim, joint_dim)
        self.fusion = CrossAttentionFusion(joint_dim)
        self.classification_head = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multimodal network.

        Args:
            text_emb: BioBERT text embeddings shape (batch_size, 768)
            image_emb: CLIP image embeddings shape (batch_size, 512)

        Returns:
            Logits tensor of shape (batch_size, 2)
        """
        t_proj, i_proj = self.projection(text_emb, image_emb)
        fused = self.fusion(t_proj, i_proj)
        logits = self.classification_head(fused)
        return logits


class DriftShieldMultimodalEngine:
    """Wrapper class handling data preparation, inference, and state-management for multimodal drift."""
    def __init__(self, model_path: Optional[Path] = None, device: str = "auto") -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = MultimodalDriftFusionModel().to(self.device)
        if model_path and model_path.exists():
            self.load_model(model_path)
        self.model.eval()

    def generate_simulated_image_embedding(self, seed_text: str) -> torch.Tensor:
        """Simulates a CLIP-like image embedding vector for research and demonstration.

        Fills vector with reproducible values derived from hash of the seed string.
        """
        rng = torch.Generator()
        # Derive a stable numeric hash from the seed text
        stable_hash = sum(ord(c) * (i + 1) for i, c in enumerate(seed_text)) % 999999
        rng.manual_seed(stable_hash)
        
        # Generate a standard normal vector of size 512
        vector = torch.randn(512, generator=rng)
        # Normalize to unit sphere (like CLIP embeddings)
        vector = vector / torch.norm(vector, p=2)
        return vector

    @torch.no_grad()
    def predict_multimodal(self, text_emb: np.ndarray, image_emb: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
        """Runs joint inference on text and image feature vectors.

        Args:
            text_emb: Numpy array of shape (768,)
            image_emb: Numpy array of shape (512,)
            threshold: Drift alert score threshold (default 0.50).

        Returns:
            Dictionary containing prediction scores, verdicts, and modalities.
        """
        t_tensor = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        i_tensor = torch.tensor(image_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        logits = self.model(t_tensor, i_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        drift_score = float(probs[1])
        label = int(drift_score >= threshold)
        
        return {
            "drift_score": drift_score,
            "verdict": "RISKY" if label == 1 else "SAFE",
            "confidence": float(max(probs)),
            "modalities_fused": ["text", "image"],
            "label": label
        }

    def save_model(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(save_path))

    def load_model(self, model_path: Path) -> None:
        self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.model.eval()
