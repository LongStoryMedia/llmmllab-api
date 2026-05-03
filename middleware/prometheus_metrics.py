"""Prometheus metrics middleware for the API service."""

import time
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

# Registry — use default so /metrics picks it up automatically
_registry = CollectorRegistry()

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=_registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=_registry,
)

active_http_connections = Gauge(
    "http_active_connections",
    "Number of active HTTP connections",
    registry=_registry,
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Instrument every request with Prometheus counters and histograms."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip metrics scraping itself
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        endpoint = request.url.path

        active_http_connections.inc()
        start = time.monotonic()
        status = 500

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            duration = time.monotonic() - start
            http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
            active_http_connections.dec()

        return response


def get_metrics_registry() -> CollectorRegistry:
    return _registry
