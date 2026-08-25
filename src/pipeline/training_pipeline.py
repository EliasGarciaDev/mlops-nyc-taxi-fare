from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.core.config import MODELS_DIR
from src.core.constants import (
    FEATURE_ORDER,
    PICKUP_DATETIME_COLUMN,
    TARGET_COLUMN,
    VALID_TAXI_TYPES,
)
from src.core.exceptions import InvalidTaxiTypeError
from src.core.log import get_logger
from src.core.months import YearMonth
from src.ml.fare_rules import calibrate_flat_fare
from src.ml.registry import ArtifactContext, ModelMetadata, promote_version, save_model
from src.ml.segment_metrics import measure_segment_errors
from src.ml.splitter import split_by_cutoff
from src.ml.trainer import train_and_evaluate
from src.monitoring.drift import build_reference_profile
from src.pipeline.cleaner import clean_trip_data
from src.pipeline.extractor import extract_trip_data
from src.pipeline.feature_engineer import engineer_features

logger = get_logger(__name__)

# Só estas colunas seguem para o treino. Um mês de Yellow Taxi tem milhões de linhas e
MODEL_COLUMNS = [
    TARGET_COLUMN,
    PICKUP_DATETIME_COLUMN,
    *FEATURE_ORDER,
    "PULocationID",
    "DOLocationID",
]


def _validate_request(taxi_type: str, months: Sequence[YearMonth], validation_months: int) -> None:
    if taxi_type not in VALID_TAXI_TYPES:
        raise InvalidTaxiTypeError(f"taxi_type inválido: '{taxi_type}'. Use 'yellow' ou 'green'.")

    if not months:
        raise ValueError("Nenhum mês informado para treinamento.")

    if list(months) != sorted(months):
        raise ValueError(f"Meses fora de ordem cronológica: {[str(m) for m in months]}.")

    if len(set(months)) != len(months):
        raise ValueError(f"Meses repetidos na janela: {[str(m) for m in months]}.")

    if not 1 <= validation_months < len(months):
        raise ValueError(
            f"validation_months inválido: {validation_months}. "
            f"Use um valor entre 1 e {len(months) - 1} para uma janela de {len(months)} meses."
        )


def load_month_frame(taxi_type: str, month: YearMonth) -> pd.DataFrame:
    """Extract, clean and engineer a single month, projected to the model columns.

    Unidade de carga do pipeline e do backtest: os dois precisam exatamente do mesmo
    tratamento por mês, e ter duas implementações seria convite a divergirem.
    """
    raw = extract_trip_data(month.year, month.month, taxi_type)
    cleaned = clean_trip_data(raw)
    logger.info(
        "month_ingested",
        extra={
            "taxi_type": taxi_type,
            "month": str(month),
            "raw_rows": len(raw),
            "clean_rows": len(cleaned),
            "discarded_rows": len(raw) - len(cleaned),
        },
    )
    return engineer_features(cleaned)[MODEL_COLUMNS]


def load_dataset(taxi_type: str, months: Sequence[YearMonth]) -> pd.DataFrame:
    """Extract, clean and engineer each monthly dataset before concatenating them.

    O processamento é feito mês a mês porque o esquema da TLC muda ao longo do tempo:
    `cbd_congestion_fee` só existe a partir de 2025, e concatenar antes de derivar as
    features faria os meses antigos herdarem uma coluna que eles nunca tiveram.

    Cada mês é projetado para as colunas do modelo antes de entrar na concatenação, o que
    mantém o pico de memória proporcional ao maior mês e não à janela inteira.
    """
    frames = [load_month_frame(taxi_type, month) for month in months]
    return pd.concat(frames, ignore_index=True)


def run_training_pipeline(
    taxi_type: str,
    months: Sequence[YearMonth],
    validation_months: int = 1,
    models_dir: Path = MODELS_DIR,
) -> ModelMetadata:
    """Run the full training cycle and persist a versioned, self-describing artifact."""
    _validate_request(taxi_type, months, validation_months)

    logger.info(
        "training_started",
        extra={
            "taxi_type": taxi_type,
            "months": [str(month) for month in months],
            "validation_months": validation_months,
        },
    )

    dataset = load_dataset(taxi_type, months)
    cutoff = months[-validation_months].first_day()
    train_df, validation_df = split_by_cutoff(dataset, cutoff)
    result = train_and_evaluate(train_df, validation_df, taxi_type)

 # O baseline de drift descreve a partição de treino, e não a janela inteira: é contra o
 # que o modelo de fato viu que uma janela de produção precisa ser comparada.
    reference_profile = build_reference_profile(train_df, list(FEATURE_ORDER))
    metadata = save_model(
        result,
        [str(month) for month in months],
        cutoff,
        models_dir,
        context=ArtifactContext(
            reference_profile=reference_profile,
            flat_fare_calibration=calibrate_flat_fare(train_df),
            segment_errors=measure_segment_errors(result.model, validation_df),
        ),
    )
 # O treino direto promove sempre: quem quer o gate no meio usa o ciclo de retreino.
    promote_version(taxi_type, metadata.model_version, models_dir)

    logger.info(
        "training_finished",
        extra={
            "taxi_type": taxi_type,
            "model_version": metadata.model_version,
            "train_rmse": result.train_metrics.rmse,
            "validation_rmse": result.validation_metrics.rmse,
            "validation_r2": result.validation_metrics.r2,
        },
    )
    return metadata
