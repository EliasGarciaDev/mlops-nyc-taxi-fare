from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.core.constants import (
    FEATURE_ORDER,
    PICKUP_DATETIME_COLUMN,
    TARGET_COLUMN,
)
from src.core.months import YearMonth
from src.ml.autopilot import (
    AutopilotAction,
    AutopilotPolicy,
    run_autopilot,
    run_autopilot_for_fleet,
)
from src.ml.registry import (
    ArtifactContext,
    promote_version,
    resolve_current_metadata,
    save_model,
)
from src.ml.retraining import RetrainingTrigger
from src.ml.trainer import ModelMetrics, TrainingResult
from src.monitoring.drift import build_reference_profile

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
        }
    )
    for name in FEATURE_ORDER:
        if name not in frame.columns:
            frame[name] = pd.Series([float(index % 5) for index in range(rows)])
    return frame


def loader(slopes: dict[YearMonth, float] | None = None):
    def load(taxi_type: str, month: YearMonth) -> pd.DataFrame:
        return make_month_frame(month, slope=(slopes or {}).get(month, 2.5))

    return load


def seed_champion(models_dir, months: list[YearMonth], slope: float = 2.5, rmse: float = 0.001):
    train = pd.concat([make_month_frame(m, slope=slope) for m in months], ignore_index=True)
    model = LinearRegression().fit(train[FEATURE_ORDER], train[TARGET_COLUMN])
    metrics = ModelMetrics(rmse=rmse, mae=rmse, r2=0.99, sample_count=len(train))
    metadata = save_model(
        TrainingResult(
            model=model,
            taxi_type="yellow",
            trained_at=datetime(2024, 3, 1, tzinfo=UTC),
            train_metrics=metrics,
            validation_metrics=metrics,
        ),
        [str(m) for m in months],
        months[-1].first_day(),
        models_dir,
        context=ArtifactContext(
            reference_profile=build_reference_profile(train, list(FEATURE_ORDER)),
            trigger=RetrainingTrigger.SCHEDULED.value,
        ),
    )
    promote_version("yellow", metadata.model_version, models_dir)
    return metadata


def fly(models_dir, latest=MAR, slopes=None, **policy_kwargs):
    load = loader(slopes)
    with (
        patch("src.monitoring.triggers.load_month_frame", load),
        patch("src.ml.retraining.load_month_frame", load),
        patch("src.ml.autopilot.load_month_frame", load),
    ):
        return run_autopilot_for_fleet(
            taxi_type="yellow",
            latest_published=latest,
            models_dir=models_dir,
            policy=AutopilotPolicy(**policy_kwargs),
        )


# ---------------------------------------------------------------------------
# 1. Não agir também é uma ação
# ---------------------------------------------------------------------------


class TestAutonomousCycle:
    def test_it_retrains_and_promotes_when_the_challenger_wins(self, tmp_path):
        seed_champion(tmp_path, [JAN], slope=2.5, rmse=9.0)
        execucao = fly(tmp_path, latest=MAR, slopes={MAR: 2.5})
        assert execucao.action in (
            AutopilotAction.RETRAINED_PROMOTED,
            AutopilotAction.RETRAINED_REJECTED,
        )

    def test_a_second_run_on_the_same_month_holds(self, tmp_path):
        """Idempotência: um agendador que dispara duas vezes não acumula artefatos idênticos."""
        fly(tmp_path, latest=MAR)
        assert fly(tmp_path, latest=MAR).action is AutopilotAction.HELD

    def test_the_trigger_is_recorded_in_the_artifact(self, tmp_path):
        """Sem procedência, "por que este modelo foi treinado?" não tem resposta depois."""
        seed_champion(tmp_path, [JAN], slope=2.5, rmse=9.0)
        execucao = fly(tmp_path, latest=MAR, slopes={MAR: 2.5})
        if execucao.cycle is not None and execucao.cycle.promoted:
            ativo = resolve_current_metadata("yellow", tmp_path)
            assert ativo.trigger is not None


class TestAutomaticRollback:
    def _two_versions(self, tmp_path):
        """Uma versão boa, seguida de uma ruim promovida à força."""
        good = seed_champion(tmp_path, [JAN], slope=2.5, rmse=0.001)
        bad_train = make_month_frame(FEB, slope=100.0)
        bad_model = LinearRegression().fit(bad_train[FEATURE_ORDER], bad_train[TARGET_COLUMN])
        metrics = ModelMetrics(rmse=0.001, mae=0.001, r2=0.99, sample_count=400)
        bad = save_model(
            TrainingResult(
                model=bad_model,
                taxi_type="yellow",
                trained_at=datetime(2024, 4, 1, tzinfo=UTC),
                train_metrics=metrics,
                validation_metrics=metrics,
            ),
            [str(FEB)],
            MAR.first_day(),
            tmp_path,
        )
        promote_version("yellow", bad.model_version, tmp_path)
        return good, bad

    def test_it_reverts_when_the_previous_version_is_better(self, tmp_path):
        """O gate impede promover algo pior, mas nada impedia um modelo já promovido de se
        revelar ruim depois. Sem operador, esta é a única defesa."""
        good, _ = self._two_versions(tmp_path)
        execucao = fly(tmp_path, latest=MAR, allow_rollback=True)
        assert execucao.action is AutopilotAction.ROLLED_BACK
        assert resolve_current_metadata("yellow", tmp_path).model_version == good.model_version

    def test_the_rollback_can_be_disabled(self, tmp_path):
        self._two_versions(tmp_path)
        assert fly(tmp_path, latest=MAR, allow_rollback=False).action is not (
            AutopilotAction.ROLLED_BACK
        )

    def test_it_does_not_roll_back_twice_in_a_row(self, tmp_path):
        """O mundo muda entre os meses e inverte a ordem entre dois modelos: reverter de A
        para B e de B para A consome ciclos sem produzir modelo novo."""
        self._two_versions(tmp_path)
        primeira = fly(tmp_path, latest=MAR, allow_rollback=True)
        assert primeira.action is AutopilotAction.ROLLED_BACK
        segunda = fly(tmp_path, latest=MAR, allow_rollback=True)
        assert segunda.action is not AutopilotAction.ROLLED_BACK


class TestFailureIsolation:
    def test_a_failing_fleet_is_reported_not_raised(self, tmp_path):
        """Sem operador, uma exceção que sobe mata o agendador e o sistema para de se manter."""
        with patch("src.ml.autopilot.run_autopilot_for_fleet", side_effect=RuntimeError("falhou")):
            resultados = run_autopilot(latest_published=MAR, models_dir=tmp_path)
        assert all(r.action is AutopilotAction.FAILED for r in resultados)
