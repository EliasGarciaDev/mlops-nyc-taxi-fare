import io
import json
import logging
import urllib.request
from dataclasses import dataclass

import pandas as pd
import pytest

from src.core.log import configure_logging


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Faz qualquer chamada de rede não simulada falhar alto em vez de sair do laptop.

    Três testes do extractor baixavam o parquet real da TLC a cada execução: simulavam
    `pd.read_parquet` e esqueciam do `urlopen` uma linha antes. A suíte ficava verde, levava
    dezenas de segundos e não passava sem internet - o pior tipo de defeito, porque o sinal
    que deveria acusá-lo é justamente o que estava sendo enganado.

    `urllib.request` é a única borda de rede do projeto, então travá-la cobre todas.
    """

    def refuse(url, *args, **kwargs):
        target = getattr(url, "full_url", url)
        raise RuntimeError(
            f"Teste tentou alcançar a rede em {target}. Simule "
            f"`urllib.request.urlopen` no módulo sob teste."
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


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
        "tip_amount": [3.0],
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
        "tip_amount": [3.0],
    })
