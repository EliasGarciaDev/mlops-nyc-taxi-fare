import urllib.error
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.constants import MAX_DISCOVERY_MONTHS
from src.core.exceptions import DataExtractionError, InvalidTaxiTypeError
from src.core.months import YearMonth
from src.pipeline.extractor import (
    DOWNLOAD_MAX_ATTEMPTS,
    discover_latest_published_month,
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
        "tip_amount": [3.0],
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
        "tip_amount": [2.0],
        "payment_type": [2],
    }
    data.update(extra_cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Colunas obrigatórias no DataFrame de saída
# ---------------------------------------------------------------------------
PARQUET_URL = "https://exemplo.invalido/yellow_tripdata_2024-01.parquet"
READ_PARQUET = "src.pipeline.extractor.pd.read_parquet"
URLOPEN = "src.pipeline.extractor.urllib.request.urlopen"
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
        with patch(URLOPEN, return_value=make_http_response()), \
             patch(READ_PARQUET, return_value=make_yellow_raw()), \
             patch("src.pipeline.extractor.today_utc", frozen_today):
            resultado = extract_trip_data(2024, 1, "yellow")
        assert "pickup_datetime" in resultado.columns
        assert "dropoff_datetime" in resultado.columns
        assert "tpep_pickup_datetime" not in resultado.columns

    def test_green_columns_are_renamed_too(self):
        with patch(URLOPEN, return_value=make_http_response()), \
             patch(READ_PARQUET, return_value=make_green_raw()), \
             patch("src.pipeline.extractor.today_utc", frozen_today):
            resultado = extract_trip_data(2024, 1, "green")
        assert "pickup_datetime" in resultado.columns
        assert "lpep_pickup_datetime" not in resultado.columns

    def test_the_fleet_is_recorded_in_the_frame(self):
        with patch(URLOPEN, return_value=make_http_response()), \
             patch(READ_PARQUET, return_value=make_yellow_raw()), \
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


# ---------------------------------------------------------------------------
# 4. Descoberta do mês publicado - a TLC não tem atraso constante
# ---------------------------------------------------------------------------


def make_head_response(status: int = 200) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.status = status
    return response


class TestPublishedMonthDiscovery:
    def test_the_newest_existing_month_wins(self):
        """Os dois primeiros candidatos não saíram; o terceiro sim."""
        respostas = [make_http_error(403), make_http_error(403), make_head_response()]
        with patch(URLOPEN, side_effect=respostas), patch(SLEEP):
            encontrado = discover_latest_published_month("yellow", FROZEN_TODAY)
        assert encontrado == YearMonth(2026, 5)

    def test_the_current_month_is_never_probed(self):
        """O mês em curso não terminou, então não pode estar publicado."""
        with patch(URLOPEN, return_value=make_head_response()) as chamada, patch(SLEEP):
            discover_latest_published_month("yellow", FROZEN_TODAY)
        url = chamada.call_args.args[0].full_url
        assert "2026-07" in url

    def test_a_dropped_connection_is_not_read_as_a_missing_month(self):
        """Ler queda de conexão como ausência faria o ciclo parar de ver dado que existe."""
        respostas = [urllib.error.URLError("reset"), make_head_response()]
        with patch(URLOPEN, side_effect=respostas), patch(SLEEP):
            encontrado = discover_latest_published_month("yellow", FROZEN_TODAY)
        assert encontrado == YearMonth(2026, 7)

    def test_it_gives_up_instead_of_walking_back_forever(self):
        falhas = [make_http_error(403)] * (MAX_DISCOVERY_MONTHS + 5)
        with patch(URLOPEN, side_effect=falhas), patch(SLEEP), \
             pytest.raises(DataExtractionError, match="Nenhum mês publicado"):
            discover_latest_published_month("yellow", FROZEN_TODAY)

    def test_an_unknown_fleet_is_rejected_before_any_request(self):
        with patch(URLOPEN) as chamada, pytest.raises(InvalidTaxiTypeError, match="taxi_type"):
            discover_latest_published_month("fhv", FROZEN_TODAY)
        assert chamada.call_count == 0


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
