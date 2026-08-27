from dataclasses import dataclass
from pathlib import Path

from src.core.config import MODELS_DIR
from src.core.constants import FEATURE_ORDER, VALID_TAXI_TYPES
from src.core.exceptions import InvalidTaxiTypeError
from src.core.log import get_logger
from src.core.months import YearMonth
from src.ml.registry import load_model, resolve_current_metadata
from src.ml.retraining import RetrainingTrigger
from src.ml.trainer import evaluate_model
from src.monitoring.drift import DriftReport, compare_to_reference
from src.pipeline.training_pipeline import load_month_frame

logger = get_logger(__name__)

# Quantos meses publicados o modelo pode deixar de ter visto antes de valer um ciclo. Um mês
# é pouco dado novo para pagar o custo; muitos meses deixam o modelo envelhecer em silêncio.
DEFAULT_MONTHS_BEHIND: int = 2

# Quanto o erro pode subir acima do baseline de validação antes de disparar. É fração, não
# valor absoluto, para que o mesmo limiar sirva às duas frotas - que têm RMSE bem diferente.
DEFAULT_ERROR_TOLERANCE: float = 0.20


@dataclass
class RetrainingNeed:
    """Whether a retraining cycle is warranted, and on which evidence.

    O protocolo pede gatilho **combinado**: só calendário desperdiça ciclos quando nada mudou
    e chega tarde quando algo mudou; só evento deixa o modelo envelhecer em silêncio quando a
    mudança é gradual demais para cruzar o limiar.
    """

    should_retrain: bool
    trigger: RetrainingTrigger | None
    reason: str
    months_behind: int
    champion_version: str | None
    champion_rmse: float | None
    baseline_rmse: float | None
    drift_report: DriftReport | None


def _months_behind(training_months: list[str], latest_published: YearMonth) -> int:
    """Calcula quantos meses publicados o modelo ativo ainda não processou."""
    if not training_months:
        return 0
    year, month = (int(part) for part in training_months[-1].split("-"))
    last_trained = YearMonth(year, month)
    return max(
        0,
        (latest_published.year - last_trained.year) * 12
        + (latest_published.month - last_trained.month),
    )


def evaluate_retraining_need(
    taxi_type: str,
    latest_published: YearMonth,
    models_dir: Path = MODELS_DIR,
    months_behind_threshold: int = DEFAULT_MONTHS_BEHIND,
    error_tolerance: float = DEFAULT_ERROR_TOLERANCE,
) -> RetrainingNeed:
    """Avalia a necessidade de retreino com base em regras de precedência e evidência.

    O mês recém-publicado é o **ground truth**: ele traz a tarifa real, então o erro do campeão
    sobre ele é medível sem esperar mais nada. É por isso que a degradação pode ser um gatilho
    aqui, enquanto o tráfego de produção - corridas simuladas, sem valor real - não serviria.

    A precedência é erro, depois drift, depois calendário: degradação medida é motivo mais
    forte que deslocamento de entrada, que por sua vez é mais forte que "passou o tempo".
    """
    if taxi_type not in VALID_TAXI_TYPES:
        raise InvalidTaxiTypeError(f"taxi_type inválido: '{taxi_type}'. Use 'yellow' ou 'green'.")

    metadata = resolve_current_metadata(taxi_type, models_dir)
    if metadata is None:
        return RetrainingNeed(
            should_retrain=True,
            trigger=RetrainingTrigger.MANUAL,
            reason=f"Nenhum modelo ativo para '{taxi_type}': o primeiro treino ainda não ocorreu.",
            months_behind=0,
            champion_version=None,
            champion_rmse=None,
            baseline_rmse=None,
            drift_report=None,
        )

    behind = _months_behind(metadata.training_months, latest_published)
    if behind == 0:
        return RetrainingNeed(
            should_retrain=False,
            trigger=None,
            reason=(
                f"O modelo '{metadata.model_version}' já cobre até {latest_published}; "
                f"não há dado novo para avaliar."
            ),
            months_behind=0,
            champion_version=metadata.model_version,
            champion_rmse=None,
            baseline_rmse=metadata.validation_metrics.rmse,
            drift_report=None,
        )

    # Só a partir daqui vale baixar o mês: sem dado novo não há erro nem drift a medir.
    recent = load_month_frame(taxi_type, latest_published)
    champion = load_model(metadata, models_dir)
    champion_rmse = evaluate_model(champion, recent).rmse
    baseline_rmse = metadata.validation_metrics.rmse

    drift_report = (
        compare_to_reference(metadata.reference_profile, recent[list(FEATURE_ORDER)])
        if metadata.reference_profile is not None
        else None
    )

    need = _decide(
        metadata_version=metadata.model_version,
        latest_published=latest_published,
        behind=behind,
        months_behind_threshold=months_behind_threshold,
        champion_rmse=champion_rmse,
        baseline_rmse=baseline_rmse,
        error_tolerance=error_tolerance,
        drift_report=drift_report,
    )

    logger.info(
        "retraining_need_evaluated",
        extra={
            "taxi_type": taxi_type,
            "champion_version": metadata.model_version,
            "should_retrain": need.should_retrain,
            "trigger": need.trigger.value if need.trigger is not None else None,
            "months_behind": need.months_behind,
            "champion_rmse": need.champion_rmse,
            "baseline_rmse": need.baseline_rmse,
            "drifted_features": (
                drift_report.drifted_features() if drift_report is not None else None
            ),
        },
    )
    return need


def _decide(  # noqa: PLR0913
    *,
    metadata_version: str,
    latest_published: YearMonth,
    behind: int,
    months_behind_threshold: int,
    champion_rmse: float,
    baseline_rmse: float,
    error_tolerance: float,
    drift_report: DriftReport | None,
) -> RetrainingNeed:
    """Aplica a hierarquia de gatilhos: erro medido > drift > calendário.

    Os parâmetros são as evidências já coletadas, e todos são obrigatoriamente nomeados:
    agrupá-las num objeto só para reduzir a contagem esconderia o que a decisão considera,
    e oito posicionais seriam fáceis de trocar de ordem sem o compilador perceber.
    """
    def verdict(
        should_retrain: bool, trigger: RetrainingTrigger | None, reason: str
    ) -> RetrainingNeed:
        """Todo veredito carrega a mesma evidência; só o motivo e a conclusão mudam."""
        return RetrainingNeed(
            should_retrain=should_retrain,
            trigger=trigger,
            reason=reason,
            months_behind=behind,
            champion_version=metadata_version,
            champion_rmse=champion_rmse,
            baseline_rmse=baseline_rmse,
            drift_report=drift_report,
        )

    degradation = (champion_rmse - baseline_rmse) / baseline_rmse if baseline_rmse else 0.0

    if degradation > error_tolerance:
        return verdict(
            True,
            RetrainingTrigger.ERROR_THRESHOLD,
            f"Erro do campeão em {latest_published} subiu {degradation:.2%} acima do "
            f"baseline ({champion_rmse:.4f} contra {baseline_rmse:.4f}), além da "
            f"tolerância de {error_tolerance:.2%}.",
        )

    if drift_report is not None and drift_report.has_drift():
        drifted = ", ".join(drift_report.drifted_features())
        return verdict(
            True,
            RetrainingTrigger.DRIFT_DETECTED,
            f"Data drift em {latest_published} nas features: {drifted}.",
        )

    if behind >= months_behind_threshold:
        return verdict(
            True,
            RetrainingTrigger.SCHEDULED,
            f"O modelo não viu {behind} mês(es) publicado(s), no limiar de "
            f"{months_behind_threshold}. Sem sinal de degradação: ciclo de calendário.",
        )

    return verdict(
        False,
        None,
        f"Sem motivo para retreinar: {behind} mês(es) de atraso, erro em "
        f"{degradation:+.2%} do baseline e nenhuma feature deslocada.",
    )
