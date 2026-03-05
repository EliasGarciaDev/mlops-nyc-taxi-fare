import logging

import pytest

from src.core.log import (
    get_logger,
    set_correlation_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fail_with_value_error() -> None:
    raise ValueError("falha de exemplo")


@pytest.fixture(autouse=True)
def clean_correlation_id():
    set_correlation_id(None)
    yield
    set_correlation_id(None)


@pytest.fixture
def captured(log_capture):
    """Devolve o buffer do log e a função que lê o último registro como dict."""
    logging.getLogger().setLevel("DEBUG")
    return log_capture.buffer, lambda: log_capture.records()[-1]


# ---------------------------------------------------------------------------
# 1. Cada registro é uma linha de JSON
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_record_is_a_single_json_line(self, captured):
        buffer, _ = captured
        get_logger("teste").info("algo_aconteceu")
        assert len(buffer.getvalue().strip().splitlines()) == 1
