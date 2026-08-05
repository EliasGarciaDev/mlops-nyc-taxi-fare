

from src.cli.backtest import build_policy
from src.core.months import YearMonth
from src.ml.backtest import BacktestResult, ExpandingWindow, ReplayWindow
from src.ml.trainer import ModelMetrics

JAN, FEB, MAR = (YearMonth(2024, month) for month in range(1, 4))


def make_result(policy: str = "ExpandingWindow") -> BacktestResult:
    return BacktestResult(
        taxi_type="yellow",
        policy=policy,
        windows=[
            ReplayWindow(
                month=FEB,
                train_months=[JAN],
                train_sample_count=1000,
                metrics=ModelMetrics(rmse=4.0, mae=1.4, r2=0.94, sample_count=500),
            ),
            ReplayWindow(
                month=MAR,
                train_months=[JAN, FEB],
                train_sample_count=2000,
                metrics=ModelMetrics(rmse=5.0, mae=1.6, r2=0.91, sample_count=600),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Seleção da política pela linha de comando
# ---------------------------------------------------------------------------


class TestBuildPolicy:
    def test_expanding_ignores_the_window_size(self):
        assert isinstance(build_policy("expanding", 3), ExpandingWindow)
