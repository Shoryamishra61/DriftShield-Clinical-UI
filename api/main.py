"""
DriftShield FastAPI Backend.

Serves semantic drift prediction endpoints and handles async model evaluation executors.
"""

import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from functools import partial
from typing import List, Dict, Any, Generator, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    PredictRequest, PredictResponse, MultimodalPredictRequest, BatchPredictRequest, HealthResponse, GuidelineResult
)
from rag.pipeline import DriftShieldPipeline

CLASSIFIER_CHECKPOINT: Path = Path("checkpoints/best_model")
INDEX_DIR: Path = Path("rag/index")
pipeline: Optional[DriftShieldPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> Generator[None, None, None]:
    """Manages system startup and shutdown lifespan events for resource loading.

    Loads the pipeline checkpoint model weights and FAISS vector databases.
    """
    global pipeline
    try:
        pipeline = DriftShieldPipeline.from_checkpoints(CLASSIFIER_CHECKPOINT, INDEX_DIR)
    except Exception as e:
        print(f"Warning: Components could not be loaded at startup: {e}")
        pipeline = None
    yield
    pipeline = None

app: FastAPI = FastAPI(
    title="DriftShield API",
    description="Semantic drift detection for medical LLM inputs. Detects outdated clinical premises before they reach an LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Classifies a single clinical query for outdated medical concept drift.

    Args:
        request: PredictRequest instance holding query question and threshold.

    Returns:
        PredictResponse containing classification scores and retrieved guidelines.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    # Execute cpu/gpu bound model call in a separate thread executor
    result = await loop.run_in_executor(None, partial(pipeline, request.question))
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return PredictResponse(
        drift_score=result.drift_score,
        verdict=result.verdict,
        confidence=result.confidence,
        semantic_shift=result.semantic_shift,
        retrieved_guidelines=[
            GuidelineResult(
                text=c.text[:400],
                score=c.score,
                domain=c.domain,
                year=c.year,
                source_name=c.source_name,
            )
            for c in result.retrieved_guidelines
        ],
        threshold_used=result.threshold_used,
        processing_time_ms=elapsed_ms,
        is_multimodal=result.is_multimodal,
        image_path=result.image_path,
    )

@app.post("/v1/predict_multimodal", response_model=PredictResponse)
async def predict_multimodal(request: MultimodalPredictRequest) -> PredictResponse:
    """Fuses clinical text symptom descriptions and raw medical image paths to classify drift."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(pipeline.predict_multimodal, request.question, request.image_path)
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return PredictResponse(
        drift_score=result.drift_score,
        verdict=result.verdict,
        confidence=result.confidence,
        semantic_shift=result.semantic_shift,
        retrieved_guidelines=[
            GuidelineResult(
                text=c.text[:400],
                score=c.score,
                domain=c.domain,
                year=c.year,
                source_name=c.source_name,
            )
            for c in result.retrieved_guidelines
        ],
        threshold_used=result.threshold_used,
        processing_time_ms=elapsed_ms,
        is_multimodal=result.is_multimodal,
        image_path=result.image_path,
    )

@app.post("/v1/batch_predict")
async def batch_predict(request: BatchPredictRequest) -> List[Dict[str, Any]]:
    """Runs batch classification for a list of clinical statements.

    Args:
        request: BatchPredictRequest instance containing questions list.

    Returns:
        List of dict summaries for each query prediction.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    loop = asyncio.get_event_loop()
    results = await asyncio.gather(*[
        loop.run_in_executor(None, partial(pipeline, q))
        for q in request.questions
    ])
    return [
        {
            "question": q,
            "drift_score": r.drift_score,
            "verdict": r.verdict,
            "confidence": r.confidence
        }
        for q, r in zip(request.questions, results)
    ]

@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Performs system checks for loaded models and vector databases.

    Returns:
        HealthResponse holding service state.
    """
    loaded = pipeline is not None
    index_ok = loaded and pipeline.retriever.index is not None
    vector_count = pipeline.retriever.index.ntotal if loaded and pipeline.retriever.index else 0
    
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        index_loaded=index_ok,
        index_vector_count=vector_count,
    )

