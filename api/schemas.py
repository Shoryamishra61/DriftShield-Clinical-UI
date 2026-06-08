"""
DriftShield API Schemas.

Defines Pydantic models for inputs and outputs of the FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    """Pydantic request schema for single clinical query drift prediction."""
    question: str = Field(..., min_length=10, max_length=2048, description="The clinical query containing potential outdated premises.")
    threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Drift classification decision threshold.")

class MultimodalPredictRequest(BaseModel):
    """Pydantic request schema for joint text and visual image drift prediction."""
    question: str = Field(..., min_length=10, max_length=2048, description="The symptom description clinical query text.")
    image_path: str = Field(..., description="Path to the visual diagnostic image (e.g. chest X-ray).")
    threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Drift classification threshold.")

class GuidelineResult(BaseModel):
    """Pydantic sub-schema representing a retrieved guideline chunk."""
    text: str = Field(..., description="Guideline text excerpt.")
    score: float = Field(..., description="Cosine similarity retrieval score.")
    domain: str = Field(..., description="Medical domain specialty.")
    year: int = Field(..., description="Year of guidelines publication.")
    source_name: str = Field(..., description="Publishing organization.")

class PredictResponse(BaseModel):
    """Pydantic response schema for single query drift prediction."""
    drift_score: float = Field(..., description="Max predicted drift probability.")
    verdict: str = Field(..., description="Risk label ('RISKY' or 'SAFE').")
    confidence: float = Field(..., description="Confidence probability score.")
    semantic_shift: str = Field(..., description="Explanation string summarizing the drift premise differences.")
    retrieved_guidelines: List[GuidelineResult] = Field(..., description="Top relevant retrieved guideline segments.")
    threshold_used: float = Field(..., description="The classification threshold applied.")
    processing_time_ms: float = Field(..., description="Inference latency in milliseconds.")
    is_multimodal: bool = Field(default=False, description="True if image modality was fused.")
    image_path: Optional[str] = Field(default=None, description="Path of the visual query modality.")

class BatchPredictRequest(BaseModel):
    """Pydantic request schema for batch queries drift prediction."""
    questions: List[str] = Field(..., min_length=1, max_length=50, description="List of clinical query strings to predict.")
    threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Threshold used across predictions.")

class HealthResponse(BaseModel):
    """Pydantic health check response model."""
    status: str = Field(..., description="API state ('healthy' or 'degraded').")
    model_loaded: bool = Field(..., description="Indicates if BioBERT classifier is loaded.")
    index_loaded: bool = Field(..., description="Indicates if FAISS retrieval index is loaded.")
    index_vector_count: int = Field(..., description="Count of guideline vectors indexed in FAISS.")
    version: str = Field(default="1.0.0", description="API version string.")

