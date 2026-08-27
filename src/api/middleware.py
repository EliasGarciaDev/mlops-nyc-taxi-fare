from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.core.log import get_logger, set_correlation_id

CORRELATION_HEADER: str = "X-Request-ID"
MILLISECONDS_PER_SECOND: int = 1000

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Tag every request with a correlation id and record how long it took.

    O identificador vem do cabeçalho quando o cliente já enviou um, de modo que a
    correlação sobreviva a um proxy à frente da aplicação, e é devolvido na resposta para
    que o usuário consiga citar a requisição exata ao relatar um problema.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid4())
        set_correlation_id(correlation_id)

        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "http_request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": elapsed_ms(started_at),
                },
            )
            raise

        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": elapsed_ms(started_at),
            },
        )
        response.headers[CORRELATION_HEADER] = correlation_id
        return response


def elapsed_ms(started_at: float) -> float:
    """Milliseconds since a `perf_counter` mark, rounded to what a latency budget cares about."""
    return round((perf_counter() - started_at) * MILLISECONDS_PER_SECOND, 3)
