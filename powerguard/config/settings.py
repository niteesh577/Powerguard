"""
config/settings.py
==================
Central configuration via Pydantic Settings.
Reads from powerguard/.env automatically.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────────────────────
    app_name: str = "PowerGuard AI"
    app_version: str = "1.0.0"
    app_env: str = "development"
    log_level: str = "INFO"

    # ── LLM (Groq) ──────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    groq_max_tokens: int = Field(default=1024, ge=256, le=4096)

    # ── Qdrant ──────────────────────────────────────────────────────────────
    qdrant_api_key: str = ""
    qdrant_url: str = ""
    qdrant_collection: str = "powerguard_literature"
    qdrant_embedding_dim: int = 384   # all-MiniLM-L6-v2

    # ── CV Layer ─────────────────────────────────────────────────────────────
    # Absolute path to biomechanics-sbd so we can import analyzers
    biomechanics_module_path: str = "../biomechanics-sbd"

    # ── Risk Thresholds (per experience level) ───────────────────────────────
    # These scale the master risk equation: Risk = (Stress × Fatigue × Progression) / Recovery
    beginner_risk_multiplier: float = 1.3   # beginners have lower tissue capacity
    intermediate_risk_multiplier: float = 1.0
    advanced_risk_multiplier: float = 0.8

    # Weekly progression rate: above this = red flag
    safe_weekly_progression_rate: float = 0.10   # 10%
    warning_weekly_progression_rate: float = 0.15  # 15%


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
