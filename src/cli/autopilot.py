import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.cli.train import parse_year_month
from src.core.config import MODELS_DIR
from src.core.constants import VALID_TAXI_TYPES
from src.core.log import configure_logging
from src.core.months import latest_published_month, today_utc
from src.ml.autopilot import AutopilotAction, AutopilotPolicy, AutopilotRun, run_autopilot
from src.monitoring.triggers import DEFAULT_ERROR_TOLERANCE, DEFAULT_MONTHS_BEHIND

FAILURE_EXIT_CODE = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.autopilot",
        description=(
            "Ciclo autônomo: avalia a necessidade de retreino, reverte se a versão anterior "
            "for melhor, treina o challenger e o promove apenas se ele vencer o campeão. "
            "Desenhado para rodar agendado, sem ninguém no comando."
        ),
    )
    parser.add_argument(
        "--month",
        dest="month",
        type=parse_year_month,
        metavar="AAAA-MM",
        help="Mês publicado a considerar. Omitido, usa o mais recente segundo o calendário.",
    )
    parser.add_argument("--taxi-type", action="append", choices=VALID_TAXI_TYPES, dest="fleets")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--months-behind",
        type=int,
        default=DEFAULT_MONTHS_BEHIND,
        help="Quantos meses publicados o modelo pode não ter visto antes de valer um ciclo.",
    )
    parser.add_argument(
        "--error-tolerance",
        type=float,
        default=DEFAULT_ERROR_TOLERANCE,
        help="Fração de piora do RMSE sobre o baseline que dispara o retreino.",
    )
    parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="Desliga a reversão automática para a versão anterior quando ela é melhor.",
    )
    return parser


def format_report(runs: list[AutopilotRun]) -> str:
    lines = ["Ciclo autônomo:"]
    for run in runs:
        lines.append(f"  [{run.taxi_type}] {run.action.value}")
        lines.append(f"      {run.reason}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)

    runs = run_autopilot(
        latest_published=args.month or latest_published_month(today_utc()),
        taxi_types=args.fleets,
        models_dir=args.models_dir,
        policy=AutopilotPolicy(
            months_behind_threshold=args.months_behind,
            error_tolerance=args.error_tolerance,
            allow_rollback=not args.no_rollback,
        ),
    )

    print(format_report(runs))

    # Falha de uma frota vira código de saída para que o agendador consiga alarmar - sem
    # operador, um erro que não sai pelo código de retorno não é visto por ninguém.
    if any(run.action is AutopilotAction.FAILED for run in runs):
        sys.stderr.write("Ao menos uma frota falhou; ver o log estruturado.\n")
        return FAILURE_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
