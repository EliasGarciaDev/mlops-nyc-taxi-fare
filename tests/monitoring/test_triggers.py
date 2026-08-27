from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.core.constants import FEATURE_ORDER, PICKUP_DATETIME_COLUMN, TARGET_COLUMN
from src.core.months import YearMonth
from src.ml.registry import ArtifactContext, promote_version, save_model
from src.ml.retraining import RetrainingTrigger
from src.ml.trainer import ModelMetrics, TrainingResult
from src.monitoring.drift import build_reference_profile
from src.monitoring.triggers import (
    evaluate_retraining_need,
)

JAN, FEB, MAR, APR, MAY = (YearMonth(2024, month) for month in range(1, 6))


def make_month_frame(month: YearMonth, rows: int = 400, slope: float = 2.5, shift: float = 0.0):
    distance = pd.Series([(index % 20) * 0.5 + 0.5 + shift for index in range(rows)])
    frame = pd.DataFrame(
        {
            TARGET_COLUMN: slope * distance,
            PICKUP_DATETIME_COLUMN: pd.date_range(month.first_day(), periods=rows, freq="min"),
            "trip_distance": distance,
            "hour_of_day": pd.Series([index % 24 for index in range(rows)]),
            "trip_duration_minutes": pd.Series([(index % 30) + 5.0 for index in range(rows)]),
        }
    )
    for name in FEATURE_ORDER:
        if name not in frame.columns:
            frame[name] = pd.Series([float(index % 5) for index in range(rows)])
    return frame


def loader(slope: float = 2.5, shift: float = 0.0):
    def load(taxi_type: str, month: YearMonth) -> pd.DataFrame:
        return make_month_frame(month, slope=slope, shift=shift)

    return load


def seed_champion(models_dir, trained_months: list[YearMonth], slope: float = 2.5):
    """Grava e promove um campeão, com baseline de drift da própria janela de treino."""
    train = pd.concat([make_month_frame(m, slope=slope) for m in trained_months], ignore_index=True)
    model = LinearRegression().fit(train[FEATURE_ORDER], train[TARGET_COLUMN])
    metrics = ModelMetrics(rmse=0.001, mae=0.001, r2=0.99, sample_count=len(train))
    metadata = save_model(
        TrainingResult(
            model=model,
            taxi_type="yellow",
            trained_at=datetime(2024, 3, 1, tzinfo=UTC),
            train_metrics=metrics,
            validation_metrics=metrics,
        ),
        [str(m) for m in trained_months],
        trained_months[-1].first_day(),
        models_dir,
        context=ArtifactContext(
            reference_profile=build_reference_profile(train, list(FEATURE_ORDER)),
            trigger=RetrainingTrigger.SCHEDULED.value,
        ),
    )
    promote_version("yellow", metadata.model_version, models_dir)
    return metadata


def evaluate(models_dir, latest=MAR, load=None, **kwargs):
    with patch("src.monitoring.triggers.load_month_frame", load or loader()):
        return evaluate_retraining_need(
            taxi_type="yellow", latest_published=latest, models_dir=models_dir, **kwargs
        )


# ---------------------------------------------------------------------------
# 1. Sem campeão não há o que avaliar - treinar é a única saída
# ---------------------------------------------------------------------------


class TestNoChampion:
    def test_recommends_retraining_when_nothing_was_ever_trained(self, tmp_path):
        decision = evaluate(tmp_path)
        assert decision.should_retrain is True
        assert decision.trigger is RetrainingTrigger.MANUAL

    def test_explains_that_there_is_no_model(self, tmp_path):
        assert "nenhum modelo" in evaluate(tmp_path).reason.lower()


# ---------------------------------------------------------------------------
# 2. Calendário: chegou dado novo o suficiente
# ---------------------------------------------------------------------------


class TestCalendarTrigger:
    def test_holds_while_the_model_is_up_to_date(self, tmp_path):
        seed_champion(tmp_path, [JAN, FEB])
        decision = evaluate(tmp_path, latest=FEB)
        assert decision.should_retrain is False
        assert decision.months_behind == 0
