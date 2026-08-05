import pandas as pd
import pytest

from src.core.constants import FEATURE_ORDER, PICKUP_DATETIME_COLUMN, TARGET_COLUMN
from src.core.exceptions import InvalidTaxiTypeError
from src.core.months import YearMonth
from src.ml.backtest import (
    ExpandingWindow,
    FrozenModel,
    SlidingWindow,
    run_replay_backtest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JAN, FEB, MAR, APR, MAY = (YearMonth(2024, month) for month in range(1, 6))
HISTORY = [JAN, FEB, MAR, APR]


def make_month_frame(month: YearMonth, rows: int = 300, slope: float = 2.5) -> pd.DataFrame:
    """Dados sintéticos de um mês, com relação linear conhecida entre distância e tarifa.

    O `slope` permite mudar a relação de um mês para o outro, que é exatamente o que o
    backtest existe para detectar.
    """
    distance = pd.Series([(index % 20) * 0.5 + 0.5 for index in range(rows)])
    duration = pd.Series([(index % 30) * 1.0 + 5.0 for index in range(rows)])
    frame = pd.DataFrame(
        {
            # A relação depende só da distância: a duração saiu do contrato de features,
            # e mantê-la no alvo tornaria o erro irredutível por construção.
            TARGET_COLUMN: slope * distance,
            PICKUP_DATETIME_COLUMN: pd.date_range(
                month.first_day(), periods=rows, freq="min"
            ),
            "trip_distance": distance,
            "hour_of_day": pd.Series([index % 24 for index in range(rows)]),
            "day_of_week": pd.Series([index % 7 for index in range(rows)]),
            "is_weekend": pd.Series([1 if index % 7 >= 5 else 0 for index in range(rows)]),
            "trip_duration_minutes": duration,
            "is_airport_trip": pd.Series([1 if index % 10 == 0 else 0 for index in range(rows)]),
            "is_congestion_zone": pd.Series([1 if index % 5 == 0 else 0 for index in range(rows)]),
        }
    )
    for name in FEATURE_ORDER:
        if name not in frame.columns:
            frame[name] = 0
    return frame


def make_loader(slopes: dict[YearMonth, float] | None = None):
    """Loader injetável que não toca a rede - o backtest recebe os meses já materializados."""
    calls: list[YearMonth] = []

    def load(taxi_type: str, month: YearMonth) -> pd.DataFrame:
        calls.append(month)
        return make_month_frame(month, slope=(slopes or {}).get(month, 2.5))

    load.calls = calls  # type: ignore[attr-defined]
    return load


# ---------------------------------------------------------------------------
# 1. Políticas de retreino: que meses treinam para prever um mês alvo
# ---------------------------------------------------------------------------


class TestExpandingWindow:
    def test_uses_every_month_before_the_target(self):
        assert ExpandingWindow().training_months(HISTORY, MAY) == [JAN, FEB, MAR, APR]

    def test_never_includes_the_target_month(self):
        """Incluir o mês previsto seria vazamento - o modelo teria visto o que vai prever."""
        assert MAR not in ExpandingWindow().training_months(HISTORY, MAR)


class TestSlidingWindow:
    def test_keeps_only_the_most_recent_months(self):
        assert SlidingWindow(size=2).training_months(HISTORY, MAY) == [MAR, APR]

    def test_never_includes_the_target_month(self):
        assert MAR not in SlidingWindow(size=2).training_months(HISTORY, MAR)

    def test_a_window_smaller_than_one_month_is_rejected(self):
        with pytest.raises(ValueError, match="size inválido"):
            SlidingWindow(size=0)


class TestFrozenModel:
    def test_always_trains_on_the_same_first_months(self):
        """É o contrafactual do trabalho: se ele se mover, o ganho do retreino vira ficção."""
        congelado = FrozenModel(initial_months=2)
        assert congelado.training_months(HISTORY, MAR) == [JAN, FEB]
        assert congelado.training_months(HISTORY, APR) == [JAN, FEB]
        assert congelado.training_months(HISTORY, MAY) == [JAN, FEB]

    def test_zero_initial_months_is_rejected(self):
        with pytest.raises(ValueError, match="initial_months inválido"):
            FrozenModel(initial_months=0)


# ---------------------------------------------------------------------------
# 2. O replay: a propriedade que sustenta toda a evidência do ADR 0013
# ---------------------------------------------------------------------------


class TestReplayHasNoLeakage:
    """Se o modelo que avalia um mês tiver visto esse mês, a curva de degradação é ficção.

    É a única propriedade do backtest que não dá para conferir olhando o resultado: uma
    curva com vazamento parece boa, e é justamente por parecer boa que engana.
    """

    def test_no_evaluated_month_appears_in_its_own_training_window(self):
        resultado = run_replay_backtest(
            "yellow", [JAN, FEB, MAR, APR, MAY], ExpandingWindow(), load_month=make_loader()
        )
        for janela in resultado.windows:
            assert janela.month not in janela.train_months

    def test_no_future_month_is_used_to_predict_the_past(self):
        resultado = run_replay_backtest(
            "yellow", [JAN, FEB, MAR, APR, MAY], ExpandingWindow(), load_month=make_loader()
        )
        for janela in resultado.windows:
            assert all(treino < janela.month for treino in janela.train_months)

    def test_the_frozen_policy_never_grows_past_its_initial_window(self):
        """`min_train_months=2` porque prever fevereiro só tem janeiro de histórico."""
        resultado = run_replay_backtest(
            "yellow",
            [JAN, FEB, MAR, APR, MAY],
            FrozenModel(initial_months=2),
            load_month=make_loader(),
            min_train_months=2,
        )
        assert [janela.month for janela in resultado.windows] == [MAR, APR, MAY]
        for janela in resultado.windows:
            assert janela.train_months == [JAN, FEB]


class TestReplayDetectsRegimeChange:
    def test_a_changed_relationship_raises_the_error_of_a_frozen_model(self):
        """O motivo de o backtest existir: a tarifa muda de patamar e o congelado não sabe."""
        virada = {APR: 4.0, MAY: 4.0}
        resultado = run_replay_backtest(
            "yellow",
            [JAN, FEB, MAR, APR, MAY],
            FrozenModel(initial_months=2),
            load_month=make_loader(virada),
        )
        por_mes = {janela.month: janela.metrics.rmse for janela in resultado.windows}
        assert por_mes[MAR] < por_mes[APR]

    def test_retraining_recovers_what_the_frozen_model_loses(self):
        virada = {APR: 4.0, MAY: 4.0}
        congelado = run_replay_backtest(
            "yellow", [JAN, FEB, MAR, APR, MAY], FrozenModel(2), load_month=make_loader(virada)
        )
        deslizante = run_replay_backtest(
            "yellow", [JAN, FEB, MAR, APR, MAY], SlidingWindow(2), load_month=make_loader(virada)
        )
        assert deslizante.mean_rmse() < congelado.mean_rmse()


class TestReplayContract:
    def test_each_month_is_materialized_only_once(self):
        """Um mês de Yellow Taxi custa dezenas de megabytes e é reusado em várias janelas."""
        carregador = make_loader()
        run_replay_backtest(
            "yellow", [JAN, FEB, MAR, APR, MAY], ExpandingWindow(), load_month=carregador
        )
        assert len(carregador.calls) == len(set(carregador.calls))

    def test_a_window_too_short_to_leave_an_evaluation_month_is_rejected(self):
        with pytest.raises(ValueError, match="min_train_months"):
            run_replay_backtest(
                "yellow", [JAN, FEB], ExpandingWindow(), min_train_months=2,
                load_month=make_loader(),
            )

    def test_an_unknown_fleet_is_rejected(self):
        with pytest.raises(InvalidTaxiTypeError, match="taxi_type"):
            run_replay_backtest("fhv", HISTORY, ExpandingWindow(), load_month=make_loader())
