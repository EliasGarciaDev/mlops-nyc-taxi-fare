from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.core.constants import FEATURE_ORDER, PICKUP_DATETIME_COLUMN, TARGET_COLUMN
from src.core.months import YearMonth
from src.ml.promotion import PromotionOutcome
from src.ml.registry import promote_version, resolve_current_metadata, save_model
from src.ml.retraining import RetrainingTrigger, run_retraining_cycle
from src.ml.trainer import ModelMetrics, TrainingResult

JAN, FEB, MAR, APR = (YearMonth(2024, month) for month in range(1, 5))


def make_month_frame(month: YearMonth, rows: int = 400, slope: float = 2.5) -> pd.DataFrame:
    distance = pd.Series([(index % 20) * 0.5 + 0.5 for index in range(rows)])
    frame = pd.DataFrame(
        {
            TARGET_COLUMN: slope * distance,
            PICKUP_DATETIME_COLUMN: pd.date_range(month.first_day(), periods=rows, freq="min"),
            "trip_distance": distance,
            "hour_of_day": pd.Series([index % 24 for index in range(rows)]),
            "trip_duration_minutes": pd.Series([(index % 30) + 5.0 for index in range(rows)]),
            "is_airport_trip": pd.Series([1 if index % 8 == 0 else 0 for index in range(rows)]),
            "is_congestion_zone": pd.Series([1 if index % 4 == 0 else 0 for index in range(rows)]),
        }
    )
    for name in FEATURE_ORDER:
        if name not in frame.columns:
            frame[name] = pd.Series([float(index % 5) for index in range(rows)])
    return frame


def make_loader(slopes: dict[YearMonth, float] | None = None):
    def load(taxi_type: str, month: YearMonth) -> pd.DataFrame:
        return make_month_frame(month, slope=(slopes or {}).get(month, 2.5))

    return load


def seed_champion(models_dir, slope: float = 2.5) -> str:
    """Grava e promove um campeão treinado com a relação indicada."""
    frame = make_month_frame(JAN, slope=slope)
    model = LinearRegression().fit(frame[FEATURE_ORDER], frame[TARGET_COLUMN])
    metadata = save_model(
        TrainingResult(
            model=model,
            taxi_type="yellow",
            trained_at=datetime(2024, 2, 1, tzinfo=UTC),
            train_metrics=ModelMetrics(rmse=1.0, mae=0.5, r2=0.9, sample_count=400),
            validation_metrics=ModelMetrics(rmse=1.0, mae=0.5, r2=0.9, sample_count=400),
        ),
        ["2024-01"],
        datetime(2024, 2, 1),
        models_dir,
    )
    promote_version("yellow", metadata.model_version, models_dir)
    return metadata.model_version


def run(models_dir, months=None, slopes=None, criteria=None, trigger=RetrainingTrigger.SCHEDULED):
    with patch("src.ml.retraining.load_month_frame", make_loader(slopes)):
        return run_retraining_cycle(
            taxi_type="yellow",
            months=months or [JAN, FEB, MAR],
            models_dir=models_dir,
            trigger=trigger,
            criteria=criteria,
        )


# ---------------------------------------------------------------------------
# 1. O ciclo treina, avalia e só então decide
# ---------------------------------------------------------------------------


class TestRetrainingCycle:
    def test_first_cycle_promotes_because_there_is_no_champion(self, tmp_path):
        cycle = run(tmp_path)
        assert cycle.decision.outcome is PromotionOutcome.PROMOTED_FIRST
        assert cycle.promoted is True

    def test_the_promoted_version_becomes_active(self, tmp_path):
        cycle = run(tmp_path)
        active = resolve_current_metadata("yellow", tmp_path)
        assert active is not None
        assert active.model_version == cycle.challenger.model_version
