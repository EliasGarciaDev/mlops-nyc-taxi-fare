import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from pathlib import Path

import joblib

from src.core.constants import FEATURE_ORDER, TARGET_COLUMN
from src.core.exceptions import (
    CorruptedMetadataError,
    CorruptedPromotionLogError,
    IncompatibleModelError,
    ModelNotLoadedError,
    VersionAlreadyExistsError,
)
from src.core.log import get_logger
from src.ml.fare_rules import FlatFareCalibration
from src.ml.protocols import Predictor
from src.ml.segment_metrics import SegmentErrors
from src.ml.trainer import ModelMetrics, TrainingResult
from src.monitoring.drift import ReferenceProfile

logger = get_logger(__name__)

# Precisão de milissegundos, e não de segundos: um ciclo de retreino sobre dados em cache
VERSION_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"
CURRENT_POINTER_FILENAME = "current.json"
# Histórico append-only de promoções. Existe porque a ordem de gravação do registro não é a
PROMOTION_LOG_FILENAME = "promotions.jsonl"
METADATA_SUFFIX = ".metadata.json"
TRACKED_LIBRARIES = ("scikit-learn", "pandas", "numpy")

# Reverter exige um antes e um depois: com uma única promoção no histórico não há para onde
# voltar, e num registro sem histórico nenhum o rollback se abstém em vez de adivinhar.
MIN_HISTORY_FOR_ROLLBACK = 2


class PromotionReason(StrEnum):
    """Motivo pelo qual uma versão foi ativada em produção."""

    PROMOTION = "promotion"
    ROLLBACK = "rollback"


@dataclass
class ArtifactContext:
    """Metadados de rastreabilidade do modelo gerados durante o treinamento.

    Agrupado num objeto porque cresce: hoje são o baseline de drift e o motivo do disparo,
    e o registro de um sistema que retreina sozinho tende a acumular procedência.
    """

    reference_profile: ReferenceProfile | None = None
    trigger: str | None = None
    flat_fare_calibration: FlatFareCalibration | None = None
    segment_errors: SegmentErrors | None = None


@dataclass
class ModelMetadata:
    """Everything needed to audit, compare or roll back a trained artifact.

    Métrica gravada junto do artefato, e não no código, é o que impede o sistema de
    reportar o número de um modelo enquanto serve outro.
    """

    model_version: str
    taxi_type: str
    trained_at: datetime
    training_months: list[str]
    validation_cutoff: datetime
    target_column: str
    feature_order: list[str]
    train_metrics: ModelMetrics
    validation_metrics: ModelMetrics
    library_versions: dict[str, str]
 # Baseline de drift da janela de treino. Opcional porque artefatos anteriores a esta
    reference_profile: ReferenceProfile | None = None
 # Motivo do disparo do retreino. Sem ele, "por que este modelo foi treinado?" não tem
 # resposta meses depois - e num sistema que retreina sozinho essa pergunta é frequente.
    trigger: str | None = None
 # Excedente da tarifa fixa do JFK sobre o valor regulado . Opcional pelo mesmo
    flat_fare_calibration: FlatFareCalibration | None = None
 # Erro por região do desembarque . Opcional pelo mesmo motivo dos campos
    segment_errors: SegmentErrors | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "taxi_type": self.taxi_type,
            "trained_at": self.trained_at.isoformat(),
            "training_months": list(self.training_months),
            "validation_cutoff": self.validation_cutoff.isoformat(),
            "target_column": self.target_column,
            "feature_order": list(self.feature_order),
            "train_metrics": _metrics_to_dict(self.train_metrics),
            "validation_metrics": _metrics_to_dict(self.validation_metrics),
            "library_versions": dict(self.library_versions),
            "reference_profile": (
                self.reference_profile.to_dict() if self.reference_profile is not None else None
            ),
            "trigger": self.trigger,
            "flat_fare_calibration": (
                self.flat_fare_calibration.to_dict()
                if self.flat_fare_calibration is not None
                else None
            ),
            "segment_errors": (
                self.segment_errors.to_dict() if self.segment_errors is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ModelMetadata":
        return cls(
            model_version=str(payload["model_version"]),
            taxi_type=str(payload["taxi_type"]),
            trained_at=datetime.fromisoformat(str(payload["trained_at"])),
            training_months=[str(month) for month in _as_list(payload["training_months"], "training_months")],
            validation_cutoff=datetime.fromisoformat(str(payload["validation_cutoff"])),
            target_column=str(payload["target_column"]),
            feature_order=[str(name) for name in _as_list(payload["feature_order"], "feature_order")],
            train_metrics=_metrics_from_dict(payload["train_metrics"], "train_metrics"),
            validation_metrics=_metrics_from_dict(payload["validation_metrics"], "validation_metrics"),
            library_versions={
                str(name): str(version)
                for name, version in _as_dict(payload["library_versions"], "library_versions").items()
            },
            reference_profile=_reference_profile_from(payload.get("reference_profile")),
            trigger=(str(raw) if (raw := payload.get("trigger")) is not None else None),
            flat_fare_calibration=_flat_fare_calibration_from(payload.get("flat_fare_calibration")),
            segment_errors=_segment_errors_from(payload.get("segment_errors")),
        )


def _segment_errors_from(payload: object) -> SegmentErrors | None:
    """Read the per-region error, tolerating artifacts saved before it existed."""
    if payload is None:
        return None
    return SegmentErrors.from_dict(_as_dict(payload, "segment_errors"))


def _flat_fare_calibration_from(payload: object) -> FlatFareCalibration | None:
    """Read the flat fare calibration, tolerating artifacts saved before it existed."""
    if payload is None:
        return None
    return FlatFareCalibration.from_dict(_as_dict(payload, "flat_fare_calibration"))


def _reference_profile_from(payload: object) -> ReferenceProfile | None:
    """Read the drift baseline, tolerating artifacts saved before it existed."""
    if payload is None:
        return None
    return ReferenceProfile.from_dict(_as_dict(payload, "reference_profile"))


def _as_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise CorruptedMetadataError(
            f"Campo '{field}' deveria ser uma lista, veio {type(value).__name__}."
        )
    return value


def _as_dict(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CorruptedMetadataError(
            f"Campo '{field}' deveria ser um objeto, veio {type(value).__name__}."
        )
    return value


def _as_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CorruptedMetadataError(
            f"Campo '{field}' deveria ser numérico, veio {type(value).__name__}."
        )
    return float(value)


def _metrics_to_dict(metrics: ModelMetrics) -> dict[str, float | int]:
    return {
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "r2": metrics.r2,
        "sample_count": metrics.sample_count,
    }


def _metrics_from_dict(payload: object, field: str) -> ModelMetrics:
    values = _as_dict(payload, field)
    return ModelMetrics(
        rmse=_as_number(values["rmse"], f"{field}.rmse"),
        mae=_as_number(values["mae"], f"{field}.mae"),
        r2=_as_number(values["r2"], f"{field}.r2"),
        sample_count=int(_as_number(values["sample_count"], f"{field}.sample_count")),
    )


def build_model_version(taxi_type: str, trained_at: datetime) -> str:
    """Compose a sortable, unique identifier for one training run.

    O timestamp identifica honestamente o que é: uma execução, não um conteúdo. A ordem
    lexicográfica coincide com a cronológica, o que mantém o histórico legível.
    """
    stamp = trained_at.astimezone(UTC).strftime(VERSION_TIMESTAMP_FORMAT)
 # %f devolve microssegundos; três dígitos bastam para separar dois ciclos e mantêm o
 # identificador curto o suficiente para caber num nome de arquivo legível.
    return f"{taxi_type}-{stamp[:-4]}Z"


def collect_library_versions() -> dict[str, str]:
    """Registra versões de bibliotecas e ambiente para garantir reprodutibilidade."""
    return {name: metadata.version(name) for name in TRACKED_LIBRARIES}


def save_model(
    result: TrainingResult,
    training_months: list[str],
    validation_cutoff: datetime,
    models_dir: Path,
    context: ArtifactContext | None = None,
) -> ModelMetadata:
    """Persiste o modelo serializado e seus metadados sem alterar a versão ativa.

    Gravar e promover são operações distintas de propósito: o gate precisa avaliar um
    challenger que já existe em disco, contra o campeão que continua servindo, antes de
    decidir a troca. Quem treina chama `promote_version` depois, se for o caso.
    """
    provenance = context if context is not None else ArtifactContext()
    model_metadata = ModelMetadata(
        model_version=build_model_version(result.taxi_type, result.trained_at),
        taxi_type=result.taxi_type,
        trained_at=result.trained_at,
        training_months=training_months,
        validation_cutoff=validation_cutoff,
        target_column=TARGET_COLUMN,
        feature_order=list(FEATURE_ORDER),
        train_metrics=result.train_metrics,
        validation_metrics=result.validation_metrics,
        library_versions=collect_library_versions(),
        reference_profile=provenance.reference_profile,
        trigger=provenance.trigger,
        flat_fare_calibration=provenance.flat_fare_calibration,
        segment_errors=provenance.segment_errors,
    )

    fleet_dir = models_dir / result.taxi_type
    fleet_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = fleet_dir / f"{model_metadata.model_version}.joblib"
    if artifact_path.exists():
 # Sobrescrever apagaria um artefato que pode estar em produção, e um sistema que
 # retreina sozinho não pode perder o campeão por causa de uma colisão de relógio.
        raise VersionAlreadyExistsError(
            f"A versão '{model_metadata.model_version}' já existe em {artifact_path}. "
            f"Salvar sobrescreveria um artefato existente."
        )

    joblib.dump(result.model, artifact_path)
    (fleet_dir / f"{model_metadata.model_version}{METADATA_SUFFIX}").write_text(
        json.dumps(model_metadata.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return model_metadata


def promote_version(
    taxi_type: str,
    model_version: str,
    models_dir: Path,
    reason: PromotionReason = PromotionReason.PROMOTION,
) -> None:
    """Ativa uma versão específica atualizando o ponteiro current.json.

    O ponteiro é escrito por último e sozinho: enquanto ele não muda, a versão anterior
    continua sendo servida. Isso faz da promoção - e do rollback, que é a mesma operação
    apontando para trás - uma escrita atômica de um arquivo de uma linha.
    """
    artifact = models_dir / taxi_type / f"{model_version}.joblib"
    if not artifact.exists():
        raise ModelNotLoadedError(
            f"Não é possível promover '{model_version}': artefato ausente em {artifact}. "
            f"A versão ativa permanece inalterada."
        )

    (models_dir / taxi_type / CURRENT_POINTER_FILENAME).write_text(
        json.dumps({"model_version": model_version}, indent=2) + "\n",
        encoding="utf-8",
    )
    _append_promotion(taxi_type, model_version, models_dir, reason)
    logger.info(
        "version_promoted",
        extra={"taxi_type": taxi_type, "model_version": model_version},
    )


def _append_promotion(
    taxi_type: str, model_version: str, models_dir: Path, reason: PromotionReason
) -> None:
    """Registra a promoção no histórico para viabilizar rollback seguro."""
    entry = {
        "model_version": model_version,
        "promoted_at": datetime.now(UTC).isoformat(),
        "reason": reason.value,
    }
    with (models_dir / taxi_type / PROMOTION_LOG_FILENAME).open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")


def promotion_history(taxi_type: str, models_dir: Path) -> list[str]:
    """Every version that actually served, oldest first, with repeats collapsed.

    Uma versão reaparece no histórico a cada rollback que volta para ela. Colapsar repetições
    consecutivas mantém a leitura "quem serviu antes de quem" sem inventar idas e voltas.
    """
    log_path = models_dir / taxi_type / PROMOTION_LOG_FILENAME
    if not log_path.exists():
        return []

    versions: list[str] = []
    for number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = _as_dict(json.loads(line), PROMOTION_LOG_FILENAME)
            version = str(entry["model_version"])
        except (json.JSONDecodeError, CorruptedMetadataError, KeyError) as exc:
 # Uma escrita interrompida - o processo morrendo no meio do append - deixa a última
            raise CorruptedPromotionLogError(
                f"Linha {number} de {log_path} ilegível: {exc}. "
                f"O histórico de promoções precisa ser reparado para o rollback voltar a valer."
            ) from exc
        if not versions or versions[-1] != version:
            versions.append(version)
    return versions


def last_promotion_reason(taxi_type: str, models_dir: Path) -> PromotionReason | None:
    """Retorna o motivo da ativação da versão atual em produção.

    Distinguir as duas é o que impede o piloto de oscilar. Reverter de A para B e, no mês
    seguinte, de B para A consome ciclos sem produzir modelo novo - e o piloto não treina no
    ciclo em que reverte. Ver ADR 0025.
    """
    log_path = models_dir / taxi_type / PROMOTION_LOG_FILENAME
    if not log_path.exists():
        return None

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None

    try:
        entry = _as_dict(json.loads(lines[-1]), PROMOTION_LOG_FILENAME)
    except (json.JSONDecodeError, CorruptedPromotionLogError):
        return None

    raw = entry.get("reason")
    if raw == PromotionReason.ROLLBACK.value:
        return PromotionReason.ROLLBACK
 # Entradas gravadas antes de o motivo existir contam como promoção por mérito, que é o
 # caso comum e o que não desliga o rollback.
    return PromotionReason.PROMOTION


def previous_champion(taxi_type: str, models_dir: Path) -> str | None:
    """Retorna a versão imediatamente anterior no histórico de promoções.

    É o único alvo seguro de rollback. `previous_version` anda para trás na ordem de gravação
    do registro, onde challengers reprovados pelo gate também estão - reverter para lá
    instalaria em produção justamente o que o gate barrou. Ver ADR 0024.
    """
    try:
        history = promotion_history(taxi_type, models_dir)
    except CorruptedPromotionLogError as exc:
 # Sem histórico confiável não há para onde reverter com segurança. Abster-se deixa o
        logger.exception(
            "promotion_history_unreadable",
            extra={"taxi_type": taxi_type, "error": str(exc)},
        )
        return None

    if len(history) < MIN_HISTORY_FOR_ROLLBACK:
        return None

 # O histórico precisa terminar em quem está servindo. Se não terminar, ele não descreve o
    active = _active_version(taxi_type, models_dir)
    if active is None or history[-1] != active:
        logger.warning(
            "promotion_history_out_of_sync",
            extra={"taxi_type": taxi_type, "active": active, "last_recorded": history[-1]},
        )
        return None

    candidate = history[-2]
    if not (models_dir / taxi_type / f"{candidate}.joblib").exists():
        return None
    return candidate


def _active_version(taxi_type: str, models_dir: Path) -> str | None:
    """Lê o ponteiro da versão ativa diretamente do disco."""
    pointer_path = models_dir / taxi_type / CURRENT_POINTER_FILENAME
    if not pointer_path.exists():
        return None
    pointer = _as_dict(json.loads(pointer_path.read_text(encoding="utf-8")), CURRENT_POINTER_FILENAME)
    return str(pointer["model_version"])


def list_versions(taxi_type: str, models_dir: Path) -> list[str]:
    """Every stored version of a fleet, oldest first.

    A ordem lexicográfica do identificador coincide com a cronológica, então ordenar os
    nomes basta - não é preciso abrir cada metadado para descobrir a sequência.
    """
    fleet_dir = models_dir / taxi_type
    if not fleet_dir.is_dir():
        return []
    return sorted(path.stem for path in fleet_dir.glob("*.joblib"))


def previous_version(taxi_type: str, version: str, models_dir: Path) -> str | None:
    """Retorna a versão arquivada imediatamente anterior.

    **Não é o alvo do rollback** - `previous_champion` é. Esta função anda na ordem de
    gravação, que inclui challengers reprovados pelo gate, porque `save_model` grava antes de
    o gate decidir. Reverter para aqui instalaria em produção o que o gate barrou. Ver ADR 0024.

    Continua existindo porque "o que foi gravado antes desta versão" é uma pergunta legítima
    sobre o registro, e porque é ela que torna a diferença entre as duas ordens demonstrável.
    """
    versions = list_versions(taxi_type, models_dir)
    if version not in versions:
        return None
    position = versions.index(version)
    return versions[position - 1] if position > 0 else None


def resolve_current_metadata(taxi_type: str, models_dir: Path) -> ModelMetadata | None:
    """Lê os metadados da versão atualmente em produção."""
    pointer_path = models_dir / taxi_type / CURRENT_POINTER_FILENAME
    if not pointer_path.exists():
        return None

    pointer = _as_dict(json.loads(pointer_path.read_text(encoding="utf-8")), CURRENT_POINTER_FILENAME)
    version = str(pointer["model_version"])
    metadata_path = models_dir / taxi_type / f"{version}{METADATA_SUFFIX}"
    if not metadata_path.exists():
        raise CorruptedMetadataError(
            f"Ponteiro de '{taxi_type}' aponta para a versão '{version}', "
            f"mas {metadata_path} não existe."
        )
    payload = _as_dict(json.loads(metadata_path.read_text(encoding="utf-8")), str(metadata_path))
    return ModelMetadata.from_dict(payload)


def ensure_contract_compatibility(model_metadata: ModelMetadata) -> None:
    """Verifica compatibilidade estrita entre as features do artefato e do código.

    A verificação vive aqui, e não dentro de load_model, de propósito: quem serve predição
    precisa recusar o artefato incompatível, mas o backtest e a análise de versões antigas
    carregam artefatos históricos legitimamente - eles derivam as features do próprio
    metadado, não das constantes atuais.

    A comparação é de igualdade estrita, ordem inclusive. Um estimador ajustado a partir de
    array, sem nomes gravados, lê as colunas por posição - e aí uma lista reordenada falha
    em silêncio, que é o pior dos modos.
    """
    if model_metadata.feature_order != list(FEATURE_ORDER):
        raise IncompatibleModelError(
            f"Artefato '{model_metadata.model_version}' foi treinado com o contrato de "
            f"features {model_metadata.feature_order}, mas o código atual espera "
            f"{list(FEATURE_ORDER)}. Retreine a frota ou reverta o código que mudou o contrato."
        )
    if model_metadata.target_column != TARGET_COLUMN:
        raise IncompatibleModelError(
            f"Artefato '{model_metadata.model_version}' prediz "
            f"'{model_metadata.target_column}', mas o código atual espera '{TARGET_COLUMN}'. "
            f"Servir esse modelo apresentaria um número com outro significado ao passageiro."
        )


def load_model(model_metadata: ModelMetadata, models_dir: Path) -> Predictor:
    """Carrega o artefato joblib correspondente aos metadados fornecidos."""
    model_path = models_dir / model_metadata.taxi_type / f"{model_metadata.model_version}.joblib"
    if not model_path.exists():
        raise ModelNotLoadedError(
            f"Artefato ausente para a versão '{model_metadata.model_version}': {model_path}."
        )
    loaded: Predictor = joblib.load(model_path)
    return loaded
