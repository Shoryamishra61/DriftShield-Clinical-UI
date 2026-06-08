"""
DriftShield API Unit Tests.

Uses FastAPI's TestClient to verify health check status and route validation logic.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
import api.main

def test_health_endpoint_degraded() -> None:
    """Verifies the health endpoint state when the underlying model is not loaded."""
    api.main.pipeline = None
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False
    assert data["index_loaded"] is False
    assert data["index_vector_count"] == 0
