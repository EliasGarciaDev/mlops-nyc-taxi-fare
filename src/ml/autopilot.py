import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from src.core.config import MODELS_DIR
from src.core.constants import FIRST_SUPPORTED_MONTH, VALID_TAXI_TYPES
from src.core.log import get_logger
from src.core.months import YearMonth
from src.ml.promotion import PromotionCriteria
from src.ml.registry import (
    METADATA_SUFFIX,
    ModelMetadata,
    PromotionReason,
    last_promotion_reason,
    load_model,
    previous_champion,
    promote_version,
)
from src.ml.retraining import RetrainingCycle, RetrainingTrigger, run_retraining_cycle
from src.ml.trainer import evaluate_model
from src.monitoring.triggers import (
    DEFAULT_ERROR_TOLERANCE,
    DEFAULT_MONTHS_BEHIND,
    RetrainingNeed,
    evaluate_retraining_need,
)
from src.pipeline.training_pipeline import load_month_frame

logger = get_logger(__name__)


class AutopilotAction(StrEnum):
    """Ação executada pelo ciclo autônomo."""

    HELD = "held"
    RETRAINED_PROMOTED = "retrained_promoted"
    RETRAINED_REJECTED = "retrained_rejected"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class AutopilotPolicy:
    """Políticas de tolerância e rollback para a execução autônoma.

    Agrupado porque estes valores são lidos juntos e ajustados juntos: afrouxar a tolerância
    de erro sem revisar o limiar de calendário produz um sistema que nunca age, ou que age
    sempre.
    """

    months_behind_threshold: int = DEFAULT_MONTHS_BEHIND
    error_tolerance: float = DEFAULT_ERROR_TOLERANCE
    allow_rollback: bool = True
    promotion: PromotionCriteria = field(default_factory=PromotionCriteria)


@dataclass
class AutopilotRun:
    """Resultado da execução autônoma sobre uma frota específica."""

    taxi_type: str
    action: AutopilotAction
    reason: str
    need: RetrainingNeed | None = None
    cycle: RetrainingCycle | None = None


def _attempt_rollback(
    taxi_type: str, need: RetrainingNeed, models_dir: Path, latest_published: YearMonth
) -> AutopilotRun | None:
    """Executa rollback para a versão anterior caso ela supere o champion no mês atual.

    O gate impede promover um modelo pior, mas nada impedia um modelo **já promovido** de se
    revelar ruim depois - com o mundo mudando, ou com uma janela de treino ruim. Com operador,
    alguém percebe e reverte; sem operador, esta é a única defesa.

    O alvo vem do histórico de promoções, não da ordem de gravação do registro: challengers
    reprovados pelo gate também ficam gravados, e voltar para um deles instalaria em produção
    exatamente o que o gate barrou. Ver ADR 0024.
    """
    if need.champion_version is None or need.champion_rmse is None:
        return None

 # Reverter duas vezes seguidas é oscilar. Ojustifica o rollback como a resposta
    if last_promotion_reason(taxi_type, models_dir) is PromotionReason.ROLLBACK:
        logger.info(
            "rollback_skipped_after_rollback",
            extra={"taxi_type": taxi_type, "champion_version": need.champion_version},
        )
        return None

    earlier = previous_champion(taxi_type, models_dir)
    if earlier is None or earlier == need.champion_version:
        return None

    recent = load_month_frame(taxi_type, latest_published)
    candidate_metadata = _metadata_of(taxi_type, earlier, models_dir)
    candidate_rmse = evaluate_model(load_model(candidate_metadata, models_dir), recent).rmse

    if candidate_rmse >= need.champion_rmse:
        return None

    promote_version(taxi_type, earlier, models_dir, reason=PromotionReason.ROLLBACK)
    return AutopilotRun(
        taxi_type=taxi_type,
        action=AutopilotAction.ROLLED_BACK,
        reason=(
            f"Revertido de '{need.champion_version}' (RMSE {need.champion_rmse:.4f} em "
            f"{latest_published}) para '{earlier}' (RMSE {candidate_rmse:.4f}), que se saiu "
            f"melhor sobre o mês publicado."
        ),
        need=need,
    )


def _metadata_of(taxi_type: str, version: str, models_dir: Path) -> ModelMetadata:
    """Lê os metadados de uma versão arquivada específica."""
    path = models_dir / taxi_type / f"{version}{METADATA_SUFFIX}"
    return ModelMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))


def run_autopilot_for_fleet(
    taxi_type: str,
    latest_published: YearMonth,
    models_dir: Path = MODELS_DIR,
    policy: AutopilotPolicy | None = None,
) -> AutopilotRun:
    """Decide and act on one fleet, with nobody watching.

    A ordem importa: avalia a necessidade, tenta o rollback quando há degradação e uma versão
    anterior melhor, e só então retreina. Reverter é mais barato e mais seguro que treinar, e
    resolve o caso em que o problema é o modelo e não o mundo.
    """
    rules = policy if policy is not None else AutopilotPolicy()
    need = evaluate_retraining_need(
        taxi_type=taxi_type,
        latest_published=latest_published,
        models_dir=models_dir,
        months_behind_threshold=rules.months_behind_threshold,
        error_tolerance=rules.error_tolerance,
    )

    if not need.should_retrain:
        return AutopilotRun(
            taxi_type=taxi_type, action=AutopilotAction.HELD, reason=need.reason, need=need
        )

    if rules.allow_rollback:
        rollback = _attempt_rollback(taxi_type, need, models_dir, latest_published)
        if rollback is not None:
            logger.warning(
                "autopilot_rolled_back",
                extra={"taxi_type": taxi_type, "reason": rollback.reason},
            )
            return rollback

 # A janela de treino termina no mês publicado, que também é a arena da disputa.
    months = _training_window(need, latest_published)
    cycle = run_retraining_cycle(
        taxi_type=taxi_type,
        months=months,
        trigger=need.trigger or RetrainingTrigger.SCHEDULED,
        models_dir=models_dir,
        criteria=rules.promotion,
    )

    action = (
        AutopilotAction.RETRAINED_PROMOTED if cycle.promoted else AutopilotAction.RETRAINED_REJECTED
    )
    return AutopilotRun(
        taxi_type=taxi_type,
        action=action,
        reason=f"{need.reason} Gate: {cycle.decision.reason}",
        need=need,
        cycle=cycle,
    )


def _training_window(need: RetrainingNeed, latest_published: YearMonth) -> list[YearMonth]:
    """Define a janela de meses de treino para o modelo challenger.

    A janela cobre o que o campeão viu mais o que ele não viu, para que o challenger tenha ao
    menos tanto contexto quanto ele. O último mês fica de fora do treino pelo próprio ciclo,
    que o reserva para a disputa.

    O recuo para antes do primeiro mês suportado é cortado aqui: sem esse corte, o primeiro
    ciclo do ano pediria um mês que a TLC não publica no formato esperado e falharia - e um
    sistema sem operador que falha na estreia fica sem modelo nenhum.
    """
    first_supported = YearMonth(*FIRST_SUPPORTED_MONTH)
    span = max(need.months_behind, 1) + 1
    months = [latest_published.shifted(-offset) for offset in range(span, -1, -1)]
    return [month for month in months if month >= first_supported]


def run_autopilot(
    latest_published: YearMonth,
    taxi_types: Sequence[str] | None = None,
    models_dir: Path = MODELS_DIR,
    policy: AutopilotPolicy | None = None,
) -> list[AutopilotRun]:
    """Executa o ciclo autônomo para Yellow e Green com isolamento de falhas.

    Uma exceção que sobe mataria o agendador e o sistema pararia de se manter em silêncio -
    o modo de falha mais perigoso num sistema sem operador. Cada frota é isolada: a que
    falhar vira um resultado registrado, e o campeão dela continua servindo.
    """
    fleets = list(taxi_types) if taxi_types is not None else list(VALID_TAXI_TYPES)
    unknown = [fleet for fleet in fleets if fleet not in VALID_TAXI_TYPES]
    if unknown:
        raise ValueError(f"taxi_type inválido: {unknown}. Use 'yellow' ou 'green'.")

    runs: list[AutopilotRun] = []
    for fleet in fleets:
        try:
            runs.append(
                run_autopilot_for_fleet(
                    taxi_type=fleet,
                    latest_published=latest_published,
                    models_dir=models_dir,
                    policy=policy,
                )
            )
        except Exception as failure:
 # Captura ampla de propósito: qualquer falha numa frota - rede, disco, dado
 # malformado - precisa virar resultado em vez de derrubar o ciclo das outras.
            logger.exception("autopilot_failed", extra={"taxi_type": fleet})
            runs.append(
                AutopilotRun(
                    taxi_type=fleet,
                    action=AutopilotAction.FAILED,
                    reason=f"Ciclo de '{fleet}' falhou: {failure}",
                )
            )

    logger.info(
        "autopilot_finished",
        extra={
            "latest_published": str(latest_published),
            "actions": {run.taxi_type: run.action.value for run in runs},
        },
    )
    return runs
