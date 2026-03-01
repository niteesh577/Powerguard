"""
api/main.py
===========
FastAPI application factory.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.request_logger import RequestLoggerMiddleware
from api.routes.health import router as health_router
from api.routes.analysis import router as analysis_router
from config.settings import get_settings
from observability.logger import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("app")
    logger.info("powerguard.startup", version=settings.app_version, env=settings.app_env)

    # Pre-warm the LangGraph graph at startup (avoids cold compile on first request)
    from agents.graph import get_graph
    _ = get_graph()
    logger.info("powerguard.graph_ready")

    yield

    logger.info("powerguard.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade multi-agent powerlifting injury prevention system. "
            "Combines computer vision biomechanics analysis with LLM-powered risk synthesis."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware (order matters: outermost first) ────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggerMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(analysis_router)

    return app


app = create_app()
