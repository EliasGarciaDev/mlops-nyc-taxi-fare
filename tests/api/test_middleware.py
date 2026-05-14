import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware import RequestLoggingMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ok")
    def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/boom")
    def boom() -> dict[str, bool]:
        raise RuntimeError("falha proposital")

    return app


@pytest.fixture
def client_and_log(log_capture):
    """O cliente HTTP dos testes também loga em INFO, por isso os registros são filtrados."""
    with TestClient(build_app(), raise_server_exceptions=False) as client:
        yield client, log_capture.records


# ---------------------------------------------------------------------------
# 1. Toda requisição vira um registro
# ---------------------------------------------------------------------------


class TestRequestLogging:
    def test_logs_method_path_and_status(self, client_and_log):
        client, records = client_and_log
        client.get("/ok")
        record = records("http_request")[-1]
        assert record["method"] == "GET"
        assert record["path"] == "/ok"
        assert record["status_code"] == 200
