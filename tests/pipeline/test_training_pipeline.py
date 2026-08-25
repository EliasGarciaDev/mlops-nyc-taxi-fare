from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from src.core.constants import FEATURE_ORDER
from src.core.exceptions import InvalidTaxiTypeError
from src.core.months import YearMonth
from src.ml.promotion import default_segments
from src.pipeline.training_pipeline import MODEL_COLUMNS, load_dataset, run_training_pipeline

# ---------------------------------------------------------------------------
# Helpers - parquet mensal sintético no formato bruto da NYC TLC
# ---------------------------------------------------------------------------

ROWS_PER_MONTH = 150


def make_month_frame(year: int, month: int, rows: int = ROWS_PER_MONTH, **extra) -> pd.DataFrame:
    """Simula um parquet mensal já com os nomes de coluna normalizados pelo extractor."""
    pickups = [datetime(year, month, 1) + timedelta(hours=index) for index in range(rows)]
    distances = [(index % 20) * 0.5 + 0.5 for index in range(rows)]

    data = {
        "pickup_datetime": pickups,
        "dropoff_datetime": [pickup + timedelta(minutes=15) for pickup in pickups],
        "fare_amount": [2.5 * distance + 3.0 for distance in distances],
        # O alvo é o total pago; a diferença fixa representa as taxas determinísticas.
        "total_amount": [2.5 * distance + 9.5 for distance in distances],
        "trip_distance": distances,
        "passenger_count": [(index % 4) + 1 for index in range(rows)],
        "PULocationID": [(index % 200) + 1 for index in range(rows)],
        "DOLocationID": [(index % 150) + 1 for index in range(rows)],
        "RatecodeID": [(index % 6) + 1 for index in range(rows)],
        "taxi_type": ["yellow"] * rows,
    }
    data.update(extra)
    return pd.DataFrame(data)


def fake_extract(year: int, month: int, taxi_type: str) -> pd.DataFrame:
    return make_month_frame(year, month)


JANUARY_TO_MARCH = [YearMonth(2024, 1), YearMonth(2024, 2), YearMonth(2024, 3)]


# ---------------------------------------------------------------------------
# 1. load_dataset percorre extração, limpeza e features
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_it_concatenates_the_months_of_the_window(self):
        with patch("src.pipeline.training_pipeline.extract_trip_data", side_effect=fake_extract):
            dataset = load_dataset("yellow", JANUARY_TO_MARCH)
        assert len(dataset) > 0
        assert list(dataset.index) == list(range(len(dataset)))

    def test_only_the_model_columns_survive_the_projection(self):
        """Um mês de yellow tem milhões de linhas: carregar tudo estoura a memória."""
        with patch("src.pipeline.training_pipeline.extract_trip_data", side_effect=fake_extract):
            dataset = load_dataset("yellow", JANUARY_TO_MARCH)
        assert set(dataset.columns) == set(MODEL_COLUMNS)
        for descartada in ("dropoff_datetime", "passenger_count", "fare_amount"):
            assert descartada not in dataset.columns

    def test_the_zone_columns_survive_but_are_not_model_features(self):
        """As zonas entram só para calibrar a camada de regras, nunca como preditor."""
        with patch("src.pipeline.training_pipeline.extract_trip_data", side_effect=fake_extract):
            dataset = load_dataset("yellow", JANUARY_TO_MARCH)
        assert "DOLocationID" in dataset.columns
        assert "DOLocationID" not in FEATURE_ORDER

    def test_the_projection_feeds_every_column_the_gate_needs(self):
        """Sem este teste, remover uma coluna faria o veto geográfico virar decorativo."""
        exigidas = {
            coluna
            for segmento in default_segments().values()
            for coluna in segmento.required_columns
        }
        assert exigidas
        assert exigidas <= set(MODEL_COLUMNS)


class TestRequestValidation:
    def test_an_unknown_fleet_is_rejected(self):
        with pytest.raises(InvalidTaxiTypeError):
            run_training_pipeline("fhv", JANUARY_TO_MARCH)

    def test_months_out_of_order_are_rejected(self):
        with pytest.raises(ValueError, match="ordem"):
            run_training_pipeline("yellow", list(reversed(JANUARY_TO_MARCH)))

    def test_a_window_without_room_for_validation_is_rejected(self):
        """Corte temporal exige pelo menos um mês de treino e um de validação."""
        with pytest.raises(ValueError, match="validation_months"):
            run_training_pipeline("yellow", JANUARY_TO_MARCH[:1])
