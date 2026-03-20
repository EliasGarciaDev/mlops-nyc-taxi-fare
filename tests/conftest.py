import io
import json
import logging
from dataclasses import dataclass

import pandas as pd
import pytest

from src.core.log import configure_logging


@dataclass(frozen=True, slots=True)
class LogCapture:
    """Acesso aos registros estruturados emitidos durante um teste."""

    buffer: io.StringIO

    def records(self, event: str | None = None) -> list[dict]:
        parsed = [
            json.loads(line) for line in self.buffer.getvalue().splitlines() if line.strip()
        ]
        return [r for r in parsed if event is None or r.get("event") == event]


@pytest.fixture
def log_capture():
    """Redireciona o log estruturado para memória durante o teste.

    Os handlers são zerados antes de configurar porque a subida da API também configura o
    log, e a ordem de criação das fixtures não é garantida.
    """
    root = logging.getLogger()
    root.handlers = []
    capture = LogCapture(io.StringIO())
    configure_logging("INFO", capture.buffer)
    yield capture
    root.handlers = []


@pytest.fixture
def sample_yellow_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2024-01-15 08:30:00"]),
        "tpep_dropoff_datetime": pd.to_datetime(["2024-01-15 08:45:00"]),
        "fare_amount": [18.5],
        "trip_distance": [3.5],
        "passenger_count": [1],
        "PULocationID": [161],
        "DOLocationID": [237],
        "RatecodeID": [1],
        "total_amount": [21.5],
    })


@pytest.fixture
def sample_green_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "lpep_pickup_datetime": pd.to_datetime(["2024-01-15 09:00:00"]),
        "lpep_dropoff_datetime": pd.to_datetime(["2024-01-15 09:20:00"]),
        "fare_amount": [15.0],
        "trip_distance": [2.8],
        "passenger_count": [2],
        "PULocationID": [74],
        "DOLocationID": [41],
        "RatecodeID": [1],
        "total_amount": [21.5],
    })
