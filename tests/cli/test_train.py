from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from src.cli.train import main, parse_year_month
from src.core.months import YearMonth

ROWS_PER_MONTH = 150


def make_month_frame(year: int, month: int) -> pd.DataFrame:
    pickups = [datetime(year, month, 1) + timedelta(hours=index) for index in range(ROWS_PER_MONTH)]
    distances = [(index % 20) * 0.5 + 0.5 for index in range(ROWS_PER_MONTH)]
    return pd.DataFrame(
        {
            "pickup_datetime": pickups,
            "dropoff_datetime": [pickup + timedelta(minutes=15) for pickup in pickups],
            "fare_amount": [2.5 * distance + 3.0 for distance in distances],
            # O alvo é o total pago; a diferença fixa representa as taxas determinísticas.
            "total_amount": [2.5 * distance + 9.5 for distance in distances],
            "trip_distance": distances,
            "passenger_count": [(index % 4) + 1 for index in range(ROWS_PER_MONTH)],
            "PULocationID": [(index % 200) + 1 for index in range(ROWS_PER_MONTH)],
            "DOLocationID": [(index % 150) + 1 for index in range(ROWS_PER_MONTH)],
            "RatecodeID": [(index % 6) + 1 for index in range(ROWS_PER_MONTH)],
        }
    )


def fake_extract(year: int, month: int, taxi_type: str) -> pd.DataFrame:
    return make_month_frame(year, month)


def run_cli(tmp_path, *args: str) -> int:
    with patch("src.pipeline.training_pipeline.extract_trip_data", side_effect=fake_extract):
        return main(
            [
                "--taxi-type",
                "yellow",
                "--from",
                "2024-01",
                "--to",
                "2024-03",
                "--models-dir",
                str(tmp_path),
                *args,
            ]
        )


# ---------------------------------------------------------------------------
# 1. Leitura do formato AAAA-MM
# ---------------------------------------------------------------------------


class TestParseYearMonth:
    def test_parses_a_valid_month(self):
        assert parse_year_month("2024-07") == YearMonth(2024, 7)
