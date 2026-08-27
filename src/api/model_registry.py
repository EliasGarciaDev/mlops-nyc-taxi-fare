from pathlib import Path
from time import monotonic

from src.core.config import MODELS_DIR
from src.core.constants import FEATURE_ORDER, VALID_TAXI_TYPES
from src.core.exceptions import AppBaseException
from src.core.log import get_logger
from src.ml.protocols import LinearModel
from src.ml.registry import (
    ModelMetadata,
    ensure_contract_compatibility,
    load_model,
    resolve_current_metadata,
)

logger = get_logger(__name__)

# Com que frequência o ponteiro de versão é reconsultado. É uma leitura de um JSON pequeno,
# no máximo a cada 30 segundos — o custo é desprezível perto de servir o modelo errado.
DEFAULT_RELOAD_INTERVAL_SECONDS: float = 30.0


class ModelRegistry:
    """Modelos carregados em memória, com recarga a quente.

    O ciclo autônomo reescreve o ponteiro de versão a cada mês. Sem recarga em processo a API
    continuaria servindo o modelo antigo até alguém reiniciá-la, que é a última intervenção
    humana num sistema desenhado para operar sozinho.

    A troca é conservadora: o que está em memória só muda depois que o artefato novo carrega
    e passa na verificação de contrato. Uma promoção malfeita não tira a frota do ar.
    """

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        reload_interval_seconds: float = DEFAULT_RELOAD_INTERVAL_SECONDS,
    ) -> None:
        self._models_dir = models_dir
        self._reload_interval = reload_interval_seconds
        self._models: dict[str, LinearModel] = {}
        self._metadata: dict[str, ModelMetadata] = {}
        self._checked_at: float | None = None

    def model_of(self, taxi_type: str) -> LinearModel | None:
        return self._models.get(taxi_type)

    def metadata_of(self, taxi_type: str) -> ModelMetadata | None:
        return self._metadata.get(taxi_type)

    def loaded_fleets(self) -> list[str]:
        return sorted(self._models)

    def refresh_if_due(self) -> list[str]:
        """Recarrega se já passou o intervalo desde a última verificação."""
        agora = monotonic()
        if self._checked_at is not None and agora - self._checked_at < self._reload_interval:
            return []
        return self.refresh()

    def refresh(self) -> list[str]:
        """Carrega a versão que o ponteiro de cada frota aponta agora."""
        self._checked_at = monotonic()
        return [frota for frota in VALID_TAXI_TYPES if self._refresh_fleet(frota)]

    def _refresh_fleet(self, taxi_type: str) -> bool:
        try:
            metadata = resolve_current_metadata(taxi_type, self._models_dir)
        except (AppBaseException, ValueError, OSError):
            # Ponteiro corrompido ou ilegível: manter o que já serve é melhor que ficar sem modelo.
            logger.exception("model_pointer_unreadable", extra={"taxi_type": taxi_type})
            return False

        if metadata is None:
            if taxi_type not in self._models:
                logger.warning("model_unavailable", extra={"taxi_type": taxi_type})
            return False

        atual = self._metadata.get(taxi_type)
        if atual is not None and atual.model_version == metadata.model_version:
            return False

        try:
            ensure_contract_compatibility(metadata)
            model = load_model(metadata, self._models_dir)
        except (AppBaseException, ValueError, OSError):
            logger.exception(
                "model_reload_failed",
                extra={
                    "taxi_type": taxi_type,
                    "model_version": metadata.model_version,
                    "expected_features": list(FEATURE_ORDER),
                },
            )
            return False

        # A troca acontece só aqui, depois de carregar e validar: até este ponto a versão
        # anterior seguia servindo.
        self._models[taxi_type] = model
        self._metadata[taxi_type] = metadata
        logger.info(
            "model_loaded",
            extra={
                "taxi_type": taxi_type,
                "model_version": metadata.model_version,
                "previous_version": atual.model_version if atual else None,
                "validation_rmse": metadata.validation_metrics.rmse,
                "trigger": metadata.trigger,
            },
        )
        return True
