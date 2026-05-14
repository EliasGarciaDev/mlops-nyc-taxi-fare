from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from src.api import routes
from src.api.middleware import RequestLoggingMiddleware
from src.core.config import ENVIRONMENT, WEB_DIR
from src.core.log import configure_logging, get_logger

logger = get_logger(__name__)

PRODUCTION_ENVIRONMENT = "production"


class NoCacheStaticFiles(StaticFiles):
    """Serve static assets that the browser must never reuse from its cache.

    Sem isso o navegador segura a versão anterior de `app.js` e `zones.js` mesmo depois de
    o processo ser reiniciado, e o sintoma - código corrigido no disco, comportamento antigo
    na tela - é caro justamente numa demonstração. Os validadores saem junto: um ETag ou
    Last-Modified sozinho ainda autoriza um 304, que devolveria a cópia velha.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["cache-control"] = "no-store, must-revalidate"
        for validator in ("etag", "last-modified"):
            if validator in response.headers:
                del response.headers[validator]
        return response


def build_static_files(directory: Path, environment: str) -> StaticFiles:
    """Choose the caching policy of the web assets for the given environment.

    Em produção o cache é desejável e quem serve decide a validade; em desenvolvimento ele
    só atrapalha, porque o arquivo muda a cada edição.
    """
    if environment == PRODUCTION_ENVIRONMENT:
        return StaticFiles(directory=directory, html=True)
    return NoCacheStaticFiles(directory=directory, html=True)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger.info("application_starting", extra={"environment": ENVIRONMENT})
    routes.load_models()
    yield


app = FastAPI(
    title="NYC Taxi Fare Predictor",
    description="MLOps Fare Prediction Service for New York City Yellow and Green Taxis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(routes.router)

if WEB_DIR.exists():
    app.mount("/app", build_static_files(WEB_DIR, ENVIRONMENT), name="web")
