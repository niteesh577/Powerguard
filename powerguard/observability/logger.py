"""
observability/logger.py
=======================
Structured JSON logging via structlog.
Every request gets a trace_id injected into all log lines.
"""

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

# ── Context variable: trace_id is injected per-request ───────────────────────
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def generate_trace_id() -> str:
    return str(uuid.uuid4())


def _add_trace_id(logger, method, event_dict):  # noqa: ARG001
    tid = trace_id_var.get("")
    if tid:
        event_dict["trace_id"] = tid
    return event_dict


def _add_severity(logger, method, event_dict):  # noqa: ARG001
    """Map structlog level names to GCP/standard severity labels."""
    level = event_dict.get("level", "info").upper()
    event_dict["severity"] = level
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Call once at application startup."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_trace_id,
        _add_severity,
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str = "powerguard") -> structlog.BoundLogger:
    return structlog.get_logger(name)
