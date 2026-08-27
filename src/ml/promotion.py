from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from src.core.constants import MIN_EVALUATION_SAMPLES, OUTER_BOROUGH_LOCATION_IDS
from src.core.exceptions import InsufficientDataError
from src.core.log import get_logger
from src.ml.protocols import Predictor
from src.ml.trainer import ModelMetrics, evaluate_model

logger = get_logger(__name__)

# Margem mínima de melhora para valer uma troca. Melhora abaixo disso está dentro do ruído
DEFAULT_MIN_IMPROVEMENT: float = 0.01

# Quanto um segmento pode piorar sem barrar a promoção. Zero reprovaria qualquer challenger,
# porque sempre há algum recorte que oscila; o limiar separa ruído de dano concentrado.
DEFAULT_MAX_SEGMENT_REGRESSION: float = 0.05

# Abaixo disso o RMSE do segmento é instável demais para sustentar um veto.
MIN_SEGMENT_SAMPLES: int = 50

# Geografia precisa de um mínimo próprio, e muito maior. Medido em 2026-08-27 com o mesmo
MIN_GEOGRAPHIC_SEGMENT_SAMPLES: int = 10_000

NIGHT_START_HOUR: int = 22
NIGHT_END_HOUR: int = 5
SHORT_TRIP_MILES: float = 2.0
LONG_TRIP_MILES: float = 10.0

SegmentPredicate = Callable[[pd.DataFrame], "pd.Series[bool]"]


@dataclass
class Segment:
    """Recorte da janela de validação utilizado para avaliação de regressão."""

    predicate: SegmentPredicate
    min_samples: int = MIN_SEGMENT_SAMPLES
 # Colunas que o predicado exige. Um segmento cujas colunas não chegaram é pulado e
 # registrado no log - fingir que foi verificado é pior que não verificar.
    required_columns: tuple[str, ...] = ()


def _dropoff_in(borough: str) -> SegmentPredicate:
    zone_ids = OUTER_BOROUGH_LOCATION_IDS[borough]
    return lambda df: df["DOLocationID"].isin(zone_ids)


class PromotionOutcome(StrEnum):
    """Resultado da avaliação de promoção do modelo desafiante."""

    PROMOTED = "promoted"
    PROMOTED_FIRST = "promoted_first"
    REJECTED_WORSE = "rejected_worse"
    REJECTED_INSUFFICIENT_MARGIN = "rejected_insufficient_margin"
    REJECTED_SEGMENT_REGRESSION = "rejected_segment_regression"


def default_segments() -> dict[str, Segment]:
    """Slices where a regression would hurt even with a better average.

    Um modelo que melhora na média e piora só no aeroporto é exatamente o tipo de regressão
    que passa despercebida até o usuário reclamar.

    Os recortes por borough de desembarque respondem à análise de equidade da Fase 5, que
    mediu 82% de acerto dentro de US$ 5 em Manhattan contra 46% no Bronx: sem eles, o gate
    aprova um challenger que melhora a média às custas de uma região inteira, porque a média é
    91% Manhattan. Manhattan não entra - ela domina a média, então a média já a protege.
    """
    return {
        "airport": Segment(lambda df: df["is_airport_trip"] == 1),
        "congestion_zone": Segment(lambda df: df["is_congestion_zone"] == 1),
        "short_trip": Segment(lambda df: df["trip_distance"] <= SHORT_TRIP_MILES),
        "long_trip": Segment(lambda df: df["trip_distance"] >= LONG_TRIP_MILES),
        "night": Segment(
            lambda df: (df["hour_of_day"] >= NIGHT_START_HOUR)
            | (df["hour_of_day"] <= NIGHT_END_HOUR)
        ),
        **{
            f"dropoff_{borough.lower().replace(' ', '_')}": Segment(
                _dropoff_in(borough),
                min_samples=MIN_GEOGRAPHIC_SEGMENT_SAMPLES,
                required_columns=("DOLocationID",),
            )
            for borough in OUTER_BOROUGH_LOCATION_IDS
        },
    }


@dataclass
class PromotionCriteria:
    """Critérios mínimos de melhora global e tolerância máxima por segmento."""

    min_improvement: float = DEFAULT_MIN_IMPROVEMENT
    max_segment_regression: float = DEFAULT_MAX_SEGMENT_REGRESSION
    segments: dict[str, Segment] | None = None

    def slices(self) -> dict[str, Segment]:
        return self.segments if self.segments is not None else default_segments()


@dataclass
class SegmentComparison:
    """Comparativo de desempenho entre champion e challenger em um recorte específico."""

    segment: str
    champion_rmse: float
    challenger_rmse: float
    sample_count: int
    regressed: bool

    @property
    def relative_change(self) -> float:
        """Variação percentual do RMSE: positivo indica piora do challenger."""
        if self.champion_rmse == 0:
            return 0.0
        return (self.challenger_rmse - self.champion_rmse) / self.champion_rmse


@dataclass
class PromotionDecision:
    """Decisão final do gate de promoção acompanhada das métricas comparativas."""

    outcome: PromotionOutcome
    reason: str
    challenger_metrics: ModelMetrics
    champion_metrics: ModelMetrics | None
    segments: list[SegmentComparison]

    @property
    def promoted(self) -> bool:
        return self.outcome in (PromotionOutcome.PROMOTED, PromotionOutcome.PROMOTED_FIRST)

    @property
    def relative_improvement(self) -> float:
        """Percentual de redução do RMSE pelo challenger (positivo indica melhora)."""
        if self.champion_metrics is None or self.champion_metrics.rmse == 0:
            return 0.0
        return (
            self.champion_metrics.rmse - self.challenger_metrics.rmse
        ) / self.champion_metrics.rmse


def _compare_segments(
    champion: Predictor,
    challenger: Predictor,
    validation_df: pd.DataFrame,
    segments: dict[str, Segment],
    max_regression: float,
) -> list[SegmentComparison]:
    comparisons: list[SegmentComparison] = []
    for name, segment in segments.items():
        missing = [
            column for column in segment.required_columns if column not in validation_df.columns
        ]
        if missing:
            logger.warning(
                "segment_skipped_missing_columns",
                extra={"segment": name, "missing_columns": missing},
            )
            continue

        slice_df = validation_df[segment.predicate(validation_df)]
        if len(slice_df) < segment.min_samples:
            continue

        champion_rmse = evaluate_model(champion, slice_df).rmse
        challenger_rmse = evaluate_model(challenger, slice_df).rmse
        change = (
            (challenger_rmse - champion_rmse) / champion_rmse if champion_rmse else 0.0
        )
        comparisons.append(
            SegmentComparison(
                segment=name,
                champion_rmse=champion_rmse,
                challenger_rmse=challenger_rmse,
                sample_count=len(slice_df),
                regressed=change > max_regression,
            )
        )
    return comparisons


def evaluate_promotion(
    champion: Predictor | None,
    challenger: Predictor,
    validation_df: pd.DataFrame,
    criteria: PromotionCriteria | None = None,
) -> PromotionDecision:
    """Avalia se o modelo challenger substitui o champion na mesma janela de validação.

    O gate é a peça que impede o sistema de degradar sozinho: um modelo recém-treinado não
    substitui o que está em produção por ser mais novo, e sim por vencer. Reprovar é
    resultado válido - um gate que nunca reprova não está medindo nada.

    Os dois modelos são avaliados sobre **as mesmas linhas**. Comparar o challenger na janela
    dele contra o campeão na janela antiga é comparar coisas diferentes, e é o erro que faz
    um modelo pior parecer melhor.
    """
    bar = criteria if criteria is not None else PromotionCriteria()

    if len(validation_df) < MIN_EVALUATION_SAMPLES:
        raise InsufficientDataError(
            f"Janela de validação insuficiente para decidir promoção: {len(validation_df)} "
            f"linhas. Mínimo de {MIN_EVALUATION_SAMPLES}."
        )

    challenger_metrics = evaluate_model(challenger, validation_df)

    if champion is None:
 # Primeira versão da frota: não há contra o que competir, e recusar deixaria o
 # sistema sem modelo nenhum.
        decision = PromotionDecision(
            outcome=PromotionOutcome.PROMOTED_FIRST,
            reason=(
                f"Nenhum campeão em produção; challenger promovido com "
                f"RMSE {challenger_metrics.rmse:.4f}."
            ),
            challenger_metrics=challenger_metrics,
            champion_metrics=None,
            segments=[],
        )
        _log_decision(decision)
        return decision

    champion_metrics = evaluate_model(champion, validation_df)
    improvement = (champion_metrics.rmse - challenger_metrics.rmse) / champion_metrics.rmse

    comparisons = _compare_segments(
        champion, challenger, validation_df, bar.slices(), bar.max_segment_regression
    )
    regressed = [comparison for comparison in comparisons if comparison.regressed]

    if improvement < 0:
        outcome = PromotionOutcome.REJECTED_WORSE
        reason = (
            f"Challenger pior: RMSE {challenger_metrics.rmse:.4f} contra "
            f"{champion_metrics.rmse:.4f} do campeão."
        )
    elif improvement < bar.min_improvement:
        outcome = PromotionOutcome.REJECTED_INSUFFICIENT_MARGIN
        reason = (
            f"Melhora de {improvement:.2%} abaixo da margem de {bar.min_improvement:.2%}: "
            f"RMSE {challenger_metrics.rmse:.4f} contra {champion_metrics.rmse:.4f}."
        )
    elif regressed:
        names = ", ".join(
            f"{comparison.segment} ({comparison.relative_change:+.2%})" for comparison in regressed
        )
        outcome = PromotionOutcome.REJECTED_SEGMENT_REGRESSION
        reason = (
            f"Melhora de {improvement:.2%} no RMSE agregado, mas com regressão acima de "
            f"{bar.max_segment_regression:.2%} em: {names}."
        )
    else:
        outcome = PromotionOutcome.PROMOTED
        reason = (
            f"Melhora de {improvement:.2%}: RMSE {challenger_metrics.rmse:.4f} contra "
            f"{champion_metrics.rmse:.4f}, sem regressão por segmento."
        )

    decision = PromotionDecision(
        outcome=outcome,
        reason=reason,
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
        segments=comparisons,
    )
    _log_decision(decision)
    return decision


def _log_decision(decision: PromotionDecision) -> None:
    """Registra em log a decisão do gate para rastreabilidade de governança."""
    logger.info(
        "promotion_decision",
        extra={
            "outcome": decision.outcome.value,
            "promoted": decision.promoted,
            "reason": decision.reason,
            "challenger_rmse": decision.challenger_metrics.rmse,
            "champion_rmse": (
                decision.champion_metrics.rmse if decision.champion_metrics is not None else None
            ),
            "relative_improvement": decision.relative_improvement,
            "regressed_segments": [
                comparison.segment for comparison in decision.segments if comparison.regressed
            ],
        },
    )
