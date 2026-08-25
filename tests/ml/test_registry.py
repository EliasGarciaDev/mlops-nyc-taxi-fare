from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sklearn.linear_model import LinearRegression

from src.core.exceptions import (
    IncompatibleModelError,
    ModelNotLoadedError,
    VersionAlreadyExistsError,
)
from src.ml.registry import (
    CURRENT_POINTER_FILENAME,
    METADATA_SUFFIX,
    ModelMetadata,
    ensure_contract_compatibility,
    load_model,
    promote_version,
    resolve_current_metadata,
    save_model,
)
from src.ml.trainer import ModelMetrics, TrainingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRAINED_AT = datetime(2026, 8, 25, 3, 15, 0, tzinfo=UTC)
LATER = datetime(2026, 8, 25, 3, 16, 0, tzinfo=UTC)
CUTOFF = datetime(2024, 11, 1)  # os timestamps do dataset da TLC são naive
TRAINING_MONTHS = ["2024-09", "2024-10", "2024-11"]


def make_metrics(rmse: float = 4.2, sample_count: int = 1000) -> ModelMetrics:
    return ModelMetrics(rmse=rmse, mae=1.5, r2=0.93, sample_count=sample_count)


def make_result(taxi_type: str = "yellow", trained_at: datetime = TRAINED_AT) -> TrainingResult:
    model = LinearRegression()
    model.fit([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0])
    return TrainingResult(
        model=model,
        taxi_type=taxi_type,
        trained_at=trained_at,
        train_metrics=make_metrics(rmse=4.0, sample_count=900),
        validation_metrics=make_metrics(rmse=4.5, sample_count=100),
    )


def save_sample(models_dir: Path, taxi_type: str = "yellow") -> ModelMetadata:
    """Grava e promove, que é o estado em que a maioria dos testes precisa do registro."""
    metadata = save_model(make_result(taxi_type), TRAINING_MONTHS, CUTOFF, models_dir)
    promote_version(taxi_type, metadata.model_version, models_dir)
    return metadata


def store_only(models_dir: Path, minute: int, taxi_type: str = "yellow") -> ModelMetadata:
    """Grava uma versão SEM promovê-la - o estado em que um challenger reprovado fica."""
    return save_model(
        make_result(taxi_type, trained_at=TRAINED_AT.replace(minute=minute)),
        TRAINING_MONTHS,
        CUTOFF,
        models_dir,
    )

# 1. Artefato versionado que carrega os próprios metadados
# ---------------------------------------------------------------------------


class TestSaveAndResolve:
    def test_saving_writes_artifact_metadata_and_pointer(self, tmp_path):
        metadata = save_sample(tmp_path)
        fleet_dir = tmp_path / "yellow"
        assert (fleet_dir / f"{metadata.model_version}.joblib").exists()
        assert (fleet_dir / f"{metadata.model_version}{METADATA_SUFFIX}").exists()
        assert (fleet_dir / CURRENT_POINTER_FILENAME).exists()

    def test_the_metadata_carries_window_metrics_and_contract(self, tmp_path):
        """Métrica junto do artefato impede reportar o número de um modelo servindo outro."""
        metadata = save_sample(tmp_path)
        assert metadata.training_months == TRAINING_MONTHS
        assert metadata.validation_metrics.sample_count == 100
        assert metadata.target_column == "total_amount"
        assert "trip_distance" in metadata.feature_order

    def test_the_metadata_survives_the_round_trip(self, tmp_path):
        original = save_sample(tmp_path)
        assert ModelMetadata.from_dict(original.to_dict()) == original

    def test_resolving_without_any_training_returns_none(self, tmp_path):
        assert resolve_current_metadata("yellow", tmp_path) is None

    def test_saving_never_overwrites_an_existing_version(self, tmp_path):
        """Sobrescrever apagaria um artefato que pode estar em produção."""
        save_sample(tmp_path)
        with pytest.raises(VersionAlreadyExistsError):
            save_model(make_result(), TRAINING_MONTHS, CUTOFF, tmp_path)

    def test_the_saved_model_can_be_loaded_back(self, tmp_path):
        metadata = save_sample(tmp_path)
        assert load_model(metadata, tmp_path).predict([[4.0]])[0] == pytest.approx(8.0, abs=0.5)


class TestContractCompatibility:
    def test_a_matching_contract_passes(self, tmp_path):
        ensure_contract_compatibility(save_sample(tmp_path))

    def test_a_stale_feature_order_is_rejected(self, tmp_path):
        """Artefato treinado com outro conjunto de features vira 503, não 500."""
        metadata = replace(save_sample(tmp_path), feature_order=["trip_distance"])
        with pytest.raises(IncompatibleModelError):
            ensure_contract_compatibility(metadata)


# ---------------------------------------------------------------------------
# 2. Promoção e histórico de quem serviu
# ---------------------------------------------------------------------------


class TestPromotion:
    def test_the_pointer_follows_the_promoted_version(self, tmp_path):
        primeira = store_only(tmp_path, minute=1)
        segunda = store_only(tmp_path, minute=2)
        promote_version("yellow", primeira.model_version, tmp_path)
        promote_version("yellow", segunda.model_version, tmp_path)

        ativo = resolve_current_metadata("yellow", tmp_path)
        assert ativo is not None
        assert ativo.model_version == segunda.model_version

    def test_promoting_a_missing_artifact_is_refused(self, tmp_path):
        save_sample(tmp_path)
        with pytest.raises(ModelNotLoadedError):
            promote_version("yellow", "yellow-19700101T000000000Z", tmp_path)
