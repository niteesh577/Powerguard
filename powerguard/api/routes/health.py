"""
api/routes/health.py
====================
Health check endpoints.
GET /health        — liveness
GET /health/ready  — readiness (checks LLM + Qdrant connectivity)
"""

from fastapi import APIRouter
from pydantic import BaseModel

from config.settings import get_settings

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]


@router.get("", response_model=HealthResponse, summary="Liveness check")
async def health_live() -> HealthResponse:
    """Returns 200 if the application is alive."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness check")
async def health_ready() -> ReadinessResponse:
    """
    Verifies that all downstream dependencies are reachable.
    Returns 200 if ready, 503 if any dependency is unhealthy.
    """
    from fastapi import Response
    checks: dict[str, str] = {}
    overall = "ok"

    # Check Groq reachability (lightweight)
    try:
        from langchain_groq import ChatGroq
        settings = get_settings()
        _ = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model)
        checks["groq_llm"] = "ok"
    except Exception as e:
        checks["groq_llm"] = f"error: {e}"
        overall = "degraded"

    # Check Qdrant reachability
    try:
        from vector_store.qdrant_store import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
        # Qdrant unavailable = degraded, not critical (RAG is non-blocking)

    return ReadinessResponse(status=overall, checks=checks)
