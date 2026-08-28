import pandas as pd
import pytest

from src.core.constants import FEATURE_ORDER, LOCATION_IDS_BY_BOROUGH, TARGET_COLUMN
from src.ml.segment_metrics import (
    MIN_SEGMENT_SAMPLES_FOR_MARGIN,
    measure_segment_errors,
)

MANHATTAN_ZONE = 230
BRONX_ZONE = sorted(LOCATION_IDS_BY_BOROUGH["Bronx"])[0]
STATEN_ISLAND_ZONE = sorted(LOCATION_IDS_BY_BOROUGH["Staten Island"])[0]


class ConstantErrorModel:
    """Prediz o alvo com um desvio que depende da região, para o RMSE ser conhecido."""

    def __init__(self, frame: pd.DataFrame, bias_by_zone: dict[int, float]):
        self._values = frame[TARGET_COLUMN] + frame["DOLocationID"].map(bias_by_zone).fillna(0.0)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return self._values.loc[features.index]


def make_frame(zone_counts: dict[int, int]) -> pd.DataFrame:
    """Monta uma janela de validação com a quantidade de corridas pedida por zona."""
    rows: list[dict[str, float]] = []
    for zone, count in zone_counts.items():
        for index in range(count):
            row = {name: float(index % 5) for name in FEATURE_ORDER}
            row["trip_distance"] = 1.0 + (index % 20)
            row[TARGET_COLUMN] = 3.0 + 2.5 * row["trip_distance"]
            row["DOLocationID"] = float(zone)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. O erro é medido por borough de desembarque
# ---------------------------------------------------------------------------


class TestMeasureSegmentErrors:
    def test_each_region_gets_its_own_error(self):
        frame = make_frame({
            MANHATTAN_ZONE: MIN_SEGMENT_SAMPLES_FOR_MARGIN,
            BRONX_ZONE: MIN_SEGMENT_SAMPLES_FOR_MARGIN,
        })
        model = ConstantErrorModel(frame, {MANHATTAN_ZONE: 2.0, BRONX_ZONE: 10.0})

        errors = measure_segment_errors(model, frame)

        assert errors is not None
        assert errors.rmse_by_borough["Manhattan"] == pytest.approx(2.0)
        assert errors.rmse_by_borough["Bronx"] == pytest.approx(10.0)

    def test_a_region_without_enough_trips_is_left_out(self):
        """Abaixo do mínimo o RMSE oscila mais que a diferença que ele deveria comunicar,
        e trocar um número errado por outro não é ganho."""
        frame = make_frame({
            MANHATTAN_ZONE: MIN_SEGMENT_SAMPLES_FOR_MARGIN,
            STATEN_ISLAND_ZONE: MIN_SEGMENT_SAMPLES_FOR_MARGIN - 1,
        })
        model = ConstantErrorModel(frame, {})

        errors = measure_segment_errors(model, frame)

        assert errors is not None
        assert "Manhattan" in errors.rmse_by_borough
        assert "Staten Island" not in errors.rmse_by_borough
