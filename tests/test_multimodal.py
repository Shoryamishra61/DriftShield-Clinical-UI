"""
DriftShield Test Suite — Multimodal Fusion Model.

Tests the cross-attention multimodal fusion architecture for dimensionality,
forward pass correctness, and simulated image embedding generation.
"""

import pytest
import torch
import numpy as np
from models.multimodal import (
    MultimodalProjection,
    CrossAttentionFusion,
    MultimodalDriftFusionModel,
    DriftShieldMultimodalEngine,
)


class TestMultimodalProjection:
    """Tests for the projection modules."""

    def test_output_dimensions(self):
        """Projection should map text (768) and image (512) to joint_dim (256)."""
        proj = MultimodalProjection(text_dim=768, image_dim=512, joint_dim=256)
        text = torch.randn(4, 768)
        image = torch.randn(4, 512)
        t_proj, i_proj = proj(text, image)
        assert t_proj.shape == (4, 256)
        assert i_proj.shape == (4, 256)

    def test_custom_dimensions(self):
        """Projection should work with custom dimensions."""
        proj = MultimodalProjection(text_dim=384, image_dim=256, joint_dim=128)
        text = torch.randn(2, 384)
        image = torch.randn(2, 256)
        t_proj, i_proj = proj(text, image)
        assert t_proj.shape == (2, 128)
        assert i_proj.shape == (2, 128)


class TestCrossAttentionFusion:
    """Tests for the cross-attention fusion module."""

    def test_fusion_output_shape(self):
        """Fusion should produce output of joint_dim."""
        fusion = CrossAttentionFusion(joint_dim=256, num_heads=4)
        text_proj = torch.randn(4, 256)
        image_proj = torch.randn(4, 256)
        fused = fusion(text_proj, image_proj)
        assert fused.shape == (4, 256)


class TestMultimodalDriftFusionModel:
    """Tests for the complete fusion model."""

    def test_forward_pass(self):
        """Complete forward pass should produce logits of shape (batch, 2)."""
        model = MultimodalDriftFusionModel()
        text = torch.randn(4, 768)
        image = torch.randn(4, 512)
        logits = model(text, image)
        assert logits.shape == (4, 2)

    def test_single_sample(self):
        """Single sample should work correctly."""
        model = MultimodalDriftFusionModel()
        text = torch.randn(1, 768)
        image = torch.randn(1, 512)
        logits = model(text, image)
        assert logits.shape == (1, 2)


class TestDriftShieldMultimodalEngine:
    """Tests for the engine wrapper."""

    def test_simulated_image_embedding(self):
        """Simulated embedding should be 512-dim unit vector."""
        engine = DriftShieldMultimodalEngine()
        emb = engine.generate_simulated_image_embedding("chest x-ray showing pneumonia")
        assert emb.shape == (512,)
        norm = torch.norm(emb, p=2).item()
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_deterministic_embeddings(self):
        """Same seed text should produce same embedding."""
        engine = DriftShieldMultimodalEngine()
        emb1 = engine.generate_simulated_image_embedding("CT scan of lungs")
        emb2 = engine.generate_simulated_image_embedding("CT scan of lungs")
        assert torch.allclose(emb1, emb2)

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        engine = DriftShieldMultimodalEngine()
        emb1 = engine.generate_simulated_image_embedding("chest x-ray")
        emb2 = engine.generate_simulated_image_embedding("brain MRI")
        assert not torch.allclose(emb1, emb2)

    def test_predict_multimodal_output(self):
        """Prediction should return correct dict structure."""
        engine = DriftShieldMultimodalEngine()
        text_emb = np.random.randn(768).astype(np.float32)
        image_emb = np.random.randn(512).astype(np.float32)
        result = engine.predict_multimodal(text_emb, image_emb)

        assert "drift_score" in result
        assert "verdict" in result
        assert "confidence" in result
        assert "modalities_fused" in result
        assert result["verdict"] in ("SAFE", "RISKY")
        assert 0.0 <= result["drift_score"] <= 1.0
