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


def seed_champion(
    models_dir,
    trained_months: list[YearMonth],
    slope: float = 2.5,
    *,
    bias: float | None = None,
    rmse: float = 0.001,
):
    """Grava e promove um campeão, com baseline de drift da própria janela de treino."""
    train = pd.concat([make_month_frame(m, slope=slope) for m in trained_months], ignore_index=True)
    model = LinearRegression().fit(train[FEATURE_ORDER], train[TARGET_COLUMN])
    metrics = ModelMetrics(rmse=rmse, mae=rmse, r2=0.99, sample_count=len(train), bias=bias)
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


# ---------------------------------------------------------------------------
# 5. Viés: o erro passou a cair sempre para o mesmo lado
# ---------------------------------------------------------------------------


def shifted_target(month: YearMonth, surcharge: float):
    """Simula uma sobretaxa nova: todo valor sobe o mesmo tanto, e nada mais muda."""

    def load(taxi_type: str, month_asked: YearMonth) -> pd.DataFrame:
        frame = make_month_frame(month_asked)
        frame[TARGET_COLUMN] = frame[TARGET_COLUMN] + surcharge
        return frame

    return load


# O baseline precisa ser realista para o cenário significar alguma coisa: com RMSE de validação
# próximo de zero qualquer deslocamento vira degradação de milhares por cento e o ramo de erro
# dispara antes. É com erro na casa dos cinco dólares - o do modelo real - que os US$ 2,00 se
# escondem no RMSE, que é exatamente o que o ramo de viés existe para pegar.
REALISTIC_RMSE = 5.40


class TestBiasTrigger:
    def test_a_flat_surcharge_is_caught(self, tmp_path):
        seed_champion(tmp_path, [JAN, FEB], bias=0.0, rmse=REALISTIC_RMSE)
        decision = evaluate(tmp_path, load=shifted_target(MAR, 2.0), bias_tolerance=1.5)
        assert decision.should_retrain is True
        assert decision.trigger is RetrainingTrigger.BIAS_SHIFT

    def test_the_same_surcharge_does_not_move_the_error_branch(self, tmp_path):
        """A razão de ser do ramo: o RMSE não acusa o que o viés acusa."""
        seed_champion(tmp_path, [JAN, FEB], bias=0.0, rmse=REALISTIC_RMSE)
        decision = evaluate(tmp_path, load=shifted_target(MAR, 2.0), bias_tolerance=99.0)
        assert decision.trigger is not RetrainingTrigger.ERROR_THRESHOLD

    def test_a_surcharge_below_the_threshold_is_not_a_trigger(self, tmp_path):
        """Limiar testado dos dois lados: 1,00 fica dentro da variação natural medida."""
        seed_champion(tmp_path, [JAN, FEB], bias=0.0, rmse=REALISTIC_RMSE)
        decision = evaluate(tmp_path, load=shifted_target(MAR, 1.0), bias_tolerance=1.5)
        assert decision.trigger is not RetrainingTrigger.BIAS_SHIFT

    def test_an_artifact_without_a_recorded_bias_abstains(self, tmp_path):
        """Sem referência não há deslocamento a medir, e assumir zero alarmaria à toa."""
        seed_champion(tmp_path, [JAN, FEB], bias=None, rmse=REALISTIC_RMSE)
        decision = evaluate(tmp_path, load=shifted_target(MAR, 5.0), bias_tolerance=1.5)
        assert decision.trigger is not RetrainingTrigger.BIAS_SHIFT

    def test_the_reason_names_the_shift_in_dollars(self, tmp_path):
        seed_champion(tmp_path, [JAN, FEB], bias=0.0, rmse=REALISTIC_RMSE)
        decision = evaluate(tmp_path, load=shifted_target(MAR, 2.0), bias_tolerance=1.5)
        assert "US$" in decision.reason
        assert "patamar" in decision.reason
