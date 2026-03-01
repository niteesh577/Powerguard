"""
api/middleware/request_logger.py
=================================
Request/response logging middleware.
Injects a trace_id into every request and logs:
  - Incoming request (method, path, user-agent)
  - Outgoing response (status, latency)
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from observability.logger import get_logger, trace_id_var

logger = get_logger("http")


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = str(uuid.uuid4())
        token = trace_id_var.set(trace_id)

        start = time.perf_counter()

        logger.info(
            "request.received",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error("request.unhandled_error", error=str(exc))
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            trace_id_var.reset(token)

        response.headers["X-Trace-Id"] = trace_id

        logger.info(
            "request.completed",
            status=response.status_code,
            duration_ms=duration_ms,
        )

        return response
