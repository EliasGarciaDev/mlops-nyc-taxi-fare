import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.cli.train import parse_year_month
from src.core.constants import VALID_TAXI_TYPES
from src.core.exceptions import AppBaseException
from src.core.log import configure_logging
from src.core.months import YearMonth, latest_published_month, month_range, today_utc
from src.ml.backtest import (
    BacktestResult,
    ExpandingWindow,
    FrozenModel,
    RetrainingPolicy,
    SlidingWindow,
    run_replay_backtest,
)
from src.pipeline.training_pipeline import load_month_frame

FAILURE_EXIT_CODE = 1

POLICY_CHOICES = ("expanding", "sliding", "frozen")


def build_policy(name: str, window_size: int) -> RetrainingPolicy:
    """Instantiate the retraining strategy the run will replay."""
    if name == "expanding":
        return ExpandingWindow()
    if name == "sliding":
        return SlidingWindow(size=window_size)
    return FrozenModel(initial_months=window_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.backtest",
        description=(
            "Reproduz mês a mês a posição de quem está em produção: treina com o passado, "
            "prevê o mês seguinte e mede o erro. A série resultante é a curva de degradação."
        ),
    )
    parser.add_argument("--taxi-type", required=True, choices=VALID_TAXI_TYPES)
    parser.add_argument(
        "--from", dest="start", required=True, type=parse_year_month, metavar="AAAA-MM"
    )
    parser.add_argument("--to", dest="end", type=parse_year_month, metavar="AAAA-MM")
    parser.add_argument(
        "--policy",
        choices=POLICY_CHOICES,
        default="expanding",
        help=(
            "expanding: retreina com todo o histórico. sliding: retreina com os últimos "
            "--window meses. frozen: treina uma vez com os primeiros --window meses e nunca "
            "mais, que é o contrafactual de não retreinar."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Meses da janela deslizante, ou do treino inicial da política congelada.",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=1,
        help="Quantos meses do início da janela servem apenas de treino, sem serem avaliados.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Grava o resultado em JSON, para comparar políticas sem repetir o download.",
    )
    return parser


def format_report(result: BacktestResult) -> str:
    lines = [
        f"Backtest {result.taxi_type} - política {result.policy}",
        f"  {'mês':<9} {'treino':>8} {'corridas':>12} {'RMSE':>8} {'MAE':>8} {'R²':>8}",
    ]
    for window in result.windows:
        lines.append(
            f"  {window.month!s:<9} {len(window.train_months):>8} "
            f"{window.train_sample_count:>12,} {window.metrics.rmse:>8.4f} "
            f"{window.metrics.mae:>8.4f} {window.metrics.r2:>8.4f}"
        )

    if result.windows:
        first, last = result.windows[0].metrics.rmse, result.windows[-1].metrics.rmse
        lines.append(f"  RMSE médio: {result.mean_rmse():.4f}")
        # A variação entre o primeiro e o último mês é a leitura direta da degradação:
        # positiva significa que o erro cresceu ao longo da janela replicada.
        lines.append(f"  Primeiro mês {first:.4f} → último {last:.4f} ({last - first:+.4f})")
    return "\n".join(lines)


def to_json(result: BacktestResult) -> dict[str, object]:
    return {
        "taxi_type": result.taxi_type,
        "policy": result.policy,
        "mean_rmse": result.mean_rmse(),
        "windows": [
            {
                "month": str(window.month),
                "train_months": [str(month) for month in window.train_months],
                "train_sample_count": window.train_sample_count,
                "rmse": window.metrics.rmse,
                "mae": window.metrics.mae,
                "r2": window.metrics.r2,
                "sample_count": window.metrics.sample_count,
            }
            for window in result.windows
        ],
    }


def resolve_months(start: YearMonth, end: YearMonth | None) -> list[YearMonth]:
    return month_range(start, end if end is not None else latest_published_month(today_utc()))


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)

    try:
        result = run_replay_backtest(
            taxi_type=args.taxi_type,
            months=resolve_months(args.start, args.end),
            policy=build_policy(args.policy, args.window),
            load_month=load_month_frame,
            min_train_months=args.min_train_months,
        )
    except (AppBaseException, ValueError) as failure:
        sys.stderr.write(f"Falha no backtest: {failure}\n")
        return FAILURE_EXIT_CODE

    print(format_report(result))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(to_json(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nResultado gravado em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
