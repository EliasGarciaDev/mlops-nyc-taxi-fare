import pandas as pd

from src.core.constants import FEATURE_ORDER, PICKUP_DATETIME_COLUMN, TARGET_COLUMN
from src.core.months import YearMonth
from src.ml.backtest import (
    ExpandingWindow,
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
