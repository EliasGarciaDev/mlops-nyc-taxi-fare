import json
from datetime import UTC, datetime

from sklearn.linear_model import LinearRegression

from src.api.model_registry import ModelRegistry
from src.ml.registry import ArtifactContext, promote_version, save_model
from src.ml.trainer import ModelMetrics, TrainingResult

TRAINED_AT = datetime(2026, 8, 26, 10, 0, tzinfo=UTC)


def store(models_dir, taxi_type="yellow", offset_seconds=0, feature_order=None):
    """Grava e promove uma versão, opcionalmente com contrato de features adulterado."""
    model = LinearRegression()
    model.fit([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0])
    metrics = ModelMetrics(rmse=4.0, mae=1.4, r2=0.94, sample_count=900)
    metadata = save_model(
        TrainingResult(
            model=model,
            taxi_type=taxi_type,
            trained_at=TRAINED_AT.replace(second=offset_seconds),
            train_metrics=metrics,
            validation_metrics=metrics,
        ),
        ["2024-01", "2024-02"],
        datetime(2024, 2, 1),
        models_dir,
        context=ArtifactContext(trigger="scheduled"),
    )
    if feature_order is not None:
        path = models_dir / taxi_type / f"{metadata.model_version}.metadata.json"
        payload = json.loads(path.read_text())
        payload["feature_order"] = feature_order
        path.write_text(json.dumps(payload))
    promote_version(taxi_type, metadata.model_version, models_dir)
    return metadata


# ---------------------------------------------------------------------------
# 1. Carga inicial
# ---------------------------------------------------------------------------


class TestInitialLoad:
    def test_loads_a_stored_fleet(self, tmp_path):
        saved = store(tmp_path)
        registry = ModelRegistry(tmp_path)
        registry.refresh()
        assert registry.metadata_of("yellow").model_version == saved.model_version
        assert registry.model_of("yellow") is not None

    def test_a_fleet_without_a_model_stays_unavailable(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        registry.refresh()
        assert registry.model_of("yellow") is None
        assert registry.metadata_of("yellow") is None

    def test_one_missing_fleet_does_not_block_the_other(self, tmp_path):
        store(tmp_path, taxi_type="green")
        registry = ModelRegistry(tmp_path)
        registry.refresh()
        assert registry.model_of("green") is not None
        assert registry.model_of("yellow") is None
