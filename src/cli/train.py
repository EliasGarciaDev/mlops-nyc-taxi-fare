import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.core.config import MODELS_DIR
from src.core.constants import VALID_TAXI_TYPES
from src.core.exceptions import AppBaseException
from src.core.log import configure_logging
from src.core.months import YearMonth, latest_published_month, month_range, today_utc
from src.ml.registry import ModelMetadata
from src.pipeline.training_pipeline import run_training_pipeline

YEAR_MONTH_PARTS = 2
FAILURE_EXIT_CODE = 1


def parse_year_month(value: str) -> YearMonth:
    """Parse the ``AAAA-MM`` form used by the TLC monthly datasets."""
    parts = value.split("-")
    if len(parts) != YEAR_MONTH_PARTS or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(f"Mês inválido: '{value}'. Use o formato AAAA-MM.")

    year, month = int(parts[0]), int(parts[1])
    if not 1 <= month <= 12:  # noqa: PLR2004
        raise argparse.ArgumentTypeError(f"Mês inválido: '{value}'. O mês deve estar entre 01 e 12.")
    return YearMonth(year, month)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.train",
        description="Treina um modelo de tarifa a partir dos dados públicos da NYC TLC.",
    )
    parser.add_argument("--taxi-type", required=True, choices=VALID_TAXI_TYPES)
    parser.add_argument(
        "--from", dest="start", required=True, type=parse_year_month, metavar="AAAA-MM"
    )
    parser.add_argument(
        "--to",
        dest="end",
        type=parse_year_month,
        metavar="AAAA-MM",
        help=(
            "Último mês da janela. Omitido, usa o mês mais recente publicado pela TLC, "
            "de modo que um retreino agendado sempre alcance os dados novos."
        ),
    )
    parser.add_argument(
        "--validation-months",
        type=int,
        default=1,
        help="Quantos meses finais da janela ficam reservados para validação (padrão: 1).",
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    return parser


def format_summary(metadata: ModelMetadata) -> str:
    train, validation = metadata.train_metrics, metadata.validation_metrics
    return "\n".join(
        [
            f"Modelo treinado: {metadata.model_version}",
            f"  Janela        : {metadata.training_months[0]} a {metadata.training_months[-1]}",
            f"  Corte         : {metadata.validation_cutoff.date()}",
            f"  Treino        : {train.sample_count} corridas | "
            f"RMSE {train.rmse:.4f} | MAE {train.mae:.4f} | R² {train.r2:.4f}",
            f"  Validação     : {validation.sample_count} corridas | "
            f"RMSE {validation.rmse:.4f} | MAE {validation.mae:.4f} | R² {validation.r2:.4f}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    end = args.end if args.end is not None else latest_published_month(today_utc())

    try:
        metadata = run_training_pipeline(
            taxi_type=args.taxi_type,
            months=month_range(args.start, end),
            validation_months=args.validation_months,
            models_dir=args.models_dir,
        )
    except (AppBaseException, ValueError) as failure:
        # A fronteira do CLI é onde erro de domínio vira código de saída. O contexto da
        # exceção é preservado na mensagem, que é o que o operador precisa ler.
        sys.stderr.write(f"Falha no treinamento: {failure}\n")
        return FAILURE_EXIT_CODE

    print(format_summary(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
