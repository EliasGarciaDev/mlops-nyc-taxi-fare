import urllib.error
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.exceptions import DataExtractionError
from src.pipeline.extractor import (
    DOWNLOAD_MAX_ATTEMPTS,
    extract_trip_data,
    fetch_parquet,
)

# A janela de meses disponíveis acompanha o calendário. Os testes fixam "hoje" para não
# passarem a falhar sozinhos com a passagem do tempo.
FROZEN_TODAY = date(2026, 8, 25)


def frozen_today() -> date:
    return FROZEN_TODAY

# ---------------------------------------------------------------------------
# Constante com as colunas obrigatórias no DataFrame de saída
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_COLUMNS = {
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
    "fare_amount",
    "total_amount",
    "payment_type",
    "taxi_type",
}


# ---------------------------------------------------------------------------
# Helpers - DataFrames mínimos que simulam os arquivos parquet da NYC TLC
# ---------------------------------------------------------------------------

def make_yellow_raw(**extra_cols) -> pd.DataFrame:
    """Simula o parquet bruto do Yellow Taxi (usa tpep_* para datetime)."""
    data = {
        "tpep_pickup_datetime": [datetime(2024, 6, 1, 10, 0, 0)],
        "tpep_dropoff_datetime": [datetime(2024, 6, 1, 10, 30, 0)],
        "passenger_count": [2],
        "trip_distance": [3.5],
        "PULocationID": [100],
        "DOLocationID": [200],
        "RatecodeID": [1],
        "fare_amount": [15.0],
        "total_amount": [18.5],
        "payment_type": [1],
    }
    data.update(extra_cols)
    return pd.DataFrame(data)


def make_green_raw(**extra_cols) -> pd.DataFrame:
    """Simula o parquet bruto do Green Taxi (usa lpep_* para datetime)."""
    data = {
        "lpep_pickup_datetime": [datetime(2024, 6, 1, 10, 0, 0)],
        "lpep_dropoff_datetime": [datetime(2024, 6, 1, 10, 30, 0)],
        "passenger_count": [1],
        "trip_distance": [2.0],
        "PULocationID": [50],
        "DOLocationID": [80],
        "RatecodeID": [1],
        "fare_amount": [10.0],
        "total_amount": [12.0],
        "payment_type": [2],
    }
    data.update(extra_cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Colunas obrigatórias no DataFrame de saída
# ---------------------------------------------------------------------------
PARQUET_URL = "https://exemplo.invalido/yellow_tripdata_2024-01.parquet"
READ_PARQUET = "src.pipeline.extractor.pd.read_parquet"
SLEEP = "src.pipeline.extractor.time.sleep"


def make_http_response(payload: bytes = b"parquet-bytes") -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = payload
    return response


def make_http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(PARQUET_URL, status, "erro simulado", None, None)


class TestDatetimeNormalization:
    def test_yellow_columns_are_renamed_to_the_common_contract(self):
        """`tpep_` no yellow e `lpep_` no green viram o mesmo nome para o pipeline."""
        with patch("src.pipeline.extractor.pd.read_parquet", return_value=make_yellow_raw()), \
             patch("src.pipeline.extractor.today_utc", frozen_today):
            resultado = extract_trip_data(2024, 1, "yellow")
        assert "pickup_datetime" in resultado.columns
        assert "dropoff_datetime" in resultado.columns
        assert "tpep_pickup_datetime" not in resultado.columns

    def test_green_columns_are_renamed_too(self):
        with patch("src.pipeline.extractor.pd.read_parquet", return_value=make_green_raw()), \
             patch("src.pipeline.extractor.today_utc", frozen_today):
            resultado = extract_trip_data(2024, 1, "green")
        assert "pickup_datetime" in resultado.columns
        assert "lpep_pickup_datetime" not in resultado.columns

    def test_the_fleet_is_recorded_in_the_frame(self):
        with patch("src.pipeline.extractor.pd.read_parquet", return_value=make_yellow_raw()), \
             patch("src.pipeline.extractor.today_utc", frozen_today):
            resultado = extract_trip_data(2024, 1, "yellow")
        assert set(resultado["taxi_type"]) == {"yellow"}


class TestRequestValidation:
    def test_an_unknown_fleet_is_rejected(self):
        with pytest.raises(Exception, match="taxi_type"):
            extract_trip_data(2024, 1, "fhv")

    def test_a_month_not_yet_published_is_rejected(self):
        """A TLC publica com atraso; pedir o mês corrente encontraria arquivo inexistente."""
        with patch("src.pipeline.extractor.today_utc", frozen_today), \
             pytest.raises(ValueError, match="disponível"):
            extract_trip_data(2026, 8, "yellow")


class TestDownloadResilience:
    def test_a_transient_failure_is_retried(self):
        """A TLC é servida por CDN e falha de forma intermitente."""
        respostas = [urllib.error.URLError("timeout"), make_http_response()]
        with patch("src.pipeline.extractor.urllib.request.urlopen", side_effect=respostas), \
             patch("src.pipeline.extractor.pd.read_parquet", return_value=make_yellow_raw()), \
             patch("src.pipeline.extractor.time.sleep"):
            assert fetch_parquet("http://exemplo/x.parquet") is not None

    def test_it_gives_up_after_the_configured_attempts(self):
        falha = urllib.error.URLError("sem rede")
        with patch(
            "src.pipeline.extractor.urllib.request.urlopen", side_effect=[falha] * 10
        ) as chamada, patch("src.pipeline.extractor.time.sleep"), \
             pytest.raises(DataExtractionError):
            fetch_parquet("http://exemplo/x.parquet")
        assert chamada.call_count == DOWNLOAD_MAX_ATTEMPTS

    def test_a_missing_month_is_not_retried(self):
        """404 não melhora com tentativa: o arquivo não existe. Só erro de servidor é retentado."""
        with patch(
            "src.pipeline.extractor.urllib.request.urlopen", side_effect=make_http_error(404)
        ) as chamada, patch(SLEEP), pytest.raises(urllib.error.HTTPError):
            fetch_parquet(PARQUET_URL)
        assert chamada.call_count == 1
