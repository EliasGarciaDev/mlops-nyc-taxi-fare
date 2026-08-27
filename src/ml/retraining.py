from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from src.core.config import MODELS_DIR
from src.core.constants import FEATURE_ORDER, VALID_TAXI_TYPES
from src.core.exceptions import InvalidTaxiTypeError
from src.core.log import get_logger
from src.core.months import YearMonth
from src.ml.fare_rules import calibrate_flat_fare
from src.ml.promotion import PromotionCriteria, PromotionDecision, evaluate_promotion
from src.ml.protocols import Predictor
from src.ml.registry import (
    ArtifactContext,
    ModelMetadata,
    load_model,
    promote_version,
    resolve_current_metadata,
    save_model,
)
from src.ml.segment_metrics import measure_segment_errors
from src.ml.trainer import train_and_evaluate
from src.monitoring.drift import build_reference_profile
from src.pipeline.training_pipeline import load_month_frame

logger = get_logger(__name__)

# O último mês da janela é reservado para a disputa: é sobre ele que campeão e challenger
# são pontuados, e por isso ele não pode entrar no treino de nenhum dos dois.
MIN_CYCLE_MONTHS: int = 2


class RetrainingTrigger(StrEnum):
    """Motivo do disparo do ciclo de retreinamento.

    Registrado junto do artefato porque "por que este modelo foi treinado?" é uma pergunta
    que aparece meses depois, quando ninguém lembra o que estava acontecendo na época.
    """

    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    ERROR_THRESHOLD = "error_threshold"
    MANUAL = "manual"


@dataclass
class RetrainingCycle:
    """Resultado de um ciclo completo de retreino e promoção."""

    taxi_type: str
    trigger: RetrainingTrigger
    validation_month: YearMonth
    challenger: ModelMetadata
    champion_version: str | None
    decision: PromotionDecision

    @property
    def promoted(self) -> bool:
        return self.decision.promoted


def _load_champion(taxi_type: str, models_dir: Path) -> tuple[str | None, Predictor | None]:
    """Retorna a versão e o modelo atualmente em produção, se houver."""
    metadata = resolve_current_metadata(taxi_type, models_dir)
    if metadata is None:
        return None, None
    return metadata.model_version, load_model(metadata, models_dir)


def run_retraining_cycle(
    taxi_type: str,
    months: Sequence[YearMonth],
    trigger: RetrainingTrigger = RetrainingTrigger.SCHEDULED,
    models_dir: Path = MODELS_DIR,
    criteria: PromotionCriteria | None = None,
) -> RetrainingCycle:
    """Executa o ciclo de retreino e submete o challenger ao gate de promoção.

    É aqui que as peças da Fase 5 se encontram: o treino produz o candidato, o gate compara
    os dois sobre o **mesmo** mês de validação, e o ponteiro só é reescrito se o challenger
    vencer. O artefato reprovado permanece em disco - "por que este modelo não entrou?"
    precisa poder ser respondido olhando o que foi treinado.
    """
    if taxi_type not in VALID_TAXI_TYPES:
        raise InvalidTaxiTypeError(f"taxi_type inválido: '{taxi_type}'. Use 'yellow' ou 'green'.")

    ordered = sorted(months)
    if len(ordered) < MIN_CYCLE_MONTHS:
        raise ValueError(
            f"Janela de {len(ordered)} meses insuficiente: o ciclo precisa de ao menos "
            f"{MIN_CYCLE_MONTHS} meses, sendo o último reservado para a validação."
        )

    train_months, validation_month = ordered[:-1], ordered[-1]
    logger.info(
        "retraining_started",
        extra={
            "taxi_type": taxi_type,
            "trigger": trigger.value,
            "train_months": [str(month) for month in train_months],
            "validation_month": str(validation_month),
        },
    )

    train_df = pd.concat(
        [load_month_frame(taxi_type, month) for month in train_months], ignore_index=True
    )
    validation_df = load_month_frame(taxi_type, validation_month)

    result = train_and_evaluate(train_df, validation_df, taxi_type)
    challenger = save_model(
        result,
        [str(month) for month in train_months],
        validation_month.first_day(),
        models_dir,
        context=ArtifactContext(
            reference_profile=build_reference_profile(train_df, list(FEATURE_ORDER)),
            trigger=trigger.value,
            flat_fare_calibration=calibrate_flat_fare(train_df),
            segment_errors=measure_segment_errors(result.model, validation_df),
        ),
    )

    champion_version, champion = _load_champion(taxi_type, models_dir)
    decision = evaluate_promotion(
        champion=champion,
        challenger=result.model,
        validation_df=validation_df,
        criteria=criteria,
    )

    if decision.promoted:
        promote_version(taxi_type, challenger.model_version, models_dir)

    logger.info(
        "retraining_finished",
        extra={
            "taxi_type": taxi_type,
            "trigger": trigger.value,
            "challenger_version": challenger.model_version,
            "champion_version": champion_version,
            "promoted": decision.promoted,
            "outcome": decision.outcome.value,
            "reason": decision.reason,
        },
    )

    return RetrainingCycle(
        taxi_type=taxi_type,
        trigger=trigger,
        validation_month=validation_month,
        challenger=challenger,
        champion_version=champion_version,
        decision=decision,
    )
