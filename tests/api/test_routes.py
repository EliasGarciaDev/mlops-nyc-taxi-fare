from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes import get_registry
from src.core.constants import (
    HAIL_EXCLUSIONARY_ZONE_IDS,
    JFK_FLAT_FARE_AMOUNT,
    MIN_PLAUSIBLE_TOTAL_AMOUNT,
    OUT_OF_STATE_LOCATION_IDS,
)
from src.core.months import now_in_nyc
from src.ml.fare_rules import FlatFareCalibration
from src.ml.registry import ModelMetadata
from src.ml.trainer import ModelMetrics

# ---------------------------------------------------------------------------
# Helpers e fixtures
# ---------------------------------------------------------------------------

def make_mock_model(predicted_value: float = 15.0) -> MagicMock:
    """Cria um mock de modelo com predict() retornando um valor fixo."""
    model = MagicMock()
    model.predict.return_value = [predicted_value]
    return model


def make_mock_model_with_coefficients(predicted_value: float = 15.0) -> MagicMock:
    """Mock completo de LinearRegression com coef_, intercept_ e feature_names_in_."""
    model = MagicMock()
    model.predict.return_value = [predicted_value]
    model.feature_names_in_ = [
        "trip_distance", "hour_of_day", "day_of_week", "is_weekend",
        "is_airport_trip", "is_congestion_zone",
        "is_rate_jfk", "is_rate_newark", "is_rate_nassau_westchester", "is_rate_negotiated",
    ]
    model.coef_ = [2.5, 0.1, 0.05, 1.2, 5.0, 0.0, 8.0, 12.0, 20.0, 15.0]
    model.intercept_ = 3.5
    return model


def make_metadata(
    taxi_type: str = "yellow",
    validation_rmse: float = 4.23,
    flat_fare_calibration: FlatFareCalibration | None = None,
) -> ModelMetadata:
    """Metadados equivalentes aos que o pipeline de treino grava junto do artefato."""
    return ModelMetadata(
        model_version=f"{taxi_type}-20260825T031500Z",
        taxi_type=taxi_type,
        trained_at=datetime(2026, 8, 25, 3, 15, tzinfo=UTC),
        training_months=["2024-09", "2024-10", "2024-11"],
        validation_cutoff=datetime(2024, 11, 1),
        target_column="regulated_total",
        feature_order=[
            "trip_distance",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_airport_trip",
            "is_congestion_zone",
            "is_rate_jfk",
            "is_rate_newark",
            "is_rate_nassau_westchester",
            "is_rate_negotiated",
        ],
        train_metrics=ModelMetrics(rmse=4.0, mae=1.4, r2=0.94, sample_count=900),
        validation_metrics=ModelMetrics(rmse=validation_rmse, mae=1.5, r2=0.93, sample_count=100),
        library_versions={"scikit-learn": "1.5.0", "pandas": "2.2.0", "numpy": "2.0.0"},
        flat_fare_calibration=flat_fare_calibration,
    )


def both_fleets_metadata() -> dict[str, ModelMetadata]:
    return {"yellow": make_metadata("yellow"), "green": make_metadata("green")}


# O contrato de pickup_datetime é hora local de Nova York, a mesma convenção dos timestamps
# publicados pela TLC - ver.
def past_datetime(days: int = 30) -> str:
    return (now_in_nyc() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


def future_datetime(minutes: int = 30) -> str:
    return (now_in_nyc() + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S")


VALID_PAYLOAD = {
    "taxi_type": "yellow",
    "trip_distance": 3.5,
    "passenger_count": 2,
    "PULocationID": 100,
    "DOLocationID": 200,
    "RatecodeID": 1,
    "pickup_datetime": past_datetime(),
}


# Zonas escolhidas para os testes de área de serviço, verificadas contra as constantes:
# 33 é Brooklyn, livre para as duas frotas; 42 é Manhattan ao norte da linha da HEZ, onde o
# boro taxi embarca legalmente; 161 é Midtown, dentro da HEZ.
BROOKLYN_ZONE_ID = 33
MANHATTAN_ABOVE_HEZ_ZONE_ID = 42
MIDTOWN_ZONE_ID = 161
NEWARK_ZONE_ID = next(iter(OUT_OF_STATE_LOCATION_IDS))


class StubRegistry:
    """Registro em memória, para que os testes injetem modelos sem tocar em disco.

    Substitui o remendo em variáveis de módulo que os testes faziam antes: com injeção de
    dependência, fornecer o próprio registro é a forma suportada - e o teste deixa de
    depender do nome interno de um atributo do módulo de rotas.
    """

    def __init__(self, models=None, metadata=None):
        self._models = models or {}
        self._metadata = metadata or {}

    def model_of(self, taxi_type):
        return self._models.get(taxi_type)

    def metadata_of(self, taxi_type):
        return self._metadata.get(taxi_type)

    def loaded_fleets(self):
        return sorted(self._models)

    def refresh_if_due(self):
        return []

    def refresh(self):
        return []


@contextmanager
def client_with(registry: StubRegistry):
    """TestClient servindo o registro dado, sem carregar nada do disco."""
    app.dependency_overrides[get_registry] = lambda: registry
    try:
        with patch("src.api.routes.load_models"), TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client_with_model():
    """TestClient com modelo mockado carregado para yellow e green."""
    mock = make_mock_model()
    registry = StubRegistry({"yellow": mock, "green": mock}, both_fleets_metadata())
    with client_with(registry) as client:
        yield client


@pytest.fixture
def client_with_full_model():
    """TestClient com mock completo (coef_, intercept_, feature_names_in_) para /model-info."""
    mock = make_mock_model_with_coefficients()
    registry = StubRegistry({"yellow": mock, "green": mock}, both_fleets_metadata())
    with client_with(registry) as client:
        yield client


@pytest.fixture
def client_without_metadata():
    """TestClient com modelo carregado mas sem metadados - estado inconsistente do registro."""
    mock = make_mock_model_with_coefficients()
    with client_with(StubRegistry({"yellow": mock, "green": mock}, {})) as client:
        yield client


@pytest.fixture
def client_no_model():
    """TestClient sem nenhum modelo carregado (simula cold start)."""
    with client_with(StubRegistry()) as client:
        yield client


# ---------------------------------------------------------------------------
# 1. POST /predict: contrato, validação e indisponibilidade
# ---------------------------------------------------------------------------


class TestPredict:
    def test_a_valid_payload_returns_the_full_contract(self, client_with_model):
        resposta = client_with_model.post("/predict", json=VALID_PAYLOAD)
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert corpo["predicted_fare"] == pytest.approx(15.0)
        assert corpo["taxi_type"] == "yellow"
        assert corpo["model_version"].startswith("yellow-")
        assert corpo["pricing_rule"] == "model"

    def test_both_fleets_are_served(self, client_with_model):
        # Zona do Brooklyn: fora da Hail Exclusionary Zone, então as duas frotas podem embarcar.
        # A zona padrão do payload serve só à yellow, e usá-la aqui testaria o contrário.
        for frota in ("yellow", "green"):
            resposta = client_with_model.post(
                "/predict",
                json={**VALID_PAYLOAD, "taxi_type": frota, "PULocationID": BROOKLYN_ZONE_ID},
            )
            assert resposta.status_code == 200, frota

    def test_without_a_loaded_model_the_answer_is_503(self, client_no_model):
        """Cold start ou artefato incompatível: recusar é melhor que inventar tarifa."""
        assert client_no_model.post("/predict", json=VALID_PAYLOAD).status_code == 503

    @pytest.mark.parametrize(
        ("campo", "valor"),
        [
            ("taxi_type", "blue"),
            ("trip_distance", 0.0),
            ("trip_distance", 500.0),
            ("passenger_count", 0),
            ("passenger_count", 99),
            ("PULocationID", 0),
            ("DOLocationID", 266),
            ("RatecodeID", 7),
        ],
    )
    def test_a_value_outside_the_tlc_domain_is_rejected(self, client_with_model, campo, valor):
        resposta = client_with_model.post("/predict", json={**VALID_PAYLOAD, campo: valor})
        assert resposta.status_code == 422

    def test_a_pickup_in_the_future_is_rejected(self, client_with_model):
        """O contrato é hora local de Nova York; prever o passado é o caso de uso."""
        resposta = client_with_model.post(
            "/predict", json={**VALID_PAYLOAD, "pickup_datetime": future_datetime()}
        )
        assert resposta.status_code == 422


# ---------------------------------------------------------------------------
# 2. Regra de negócio: tarifa fixa do JFK e piso mínimo
# ---------------------------------------------------------------------------

JFK_ZONE = 132
TIMES_SQUARE_ZONE = 230
QUEENS_ZONE = 7

CALIBRATED = FlatFareCalibration(mean_excess=24.06, sample_count=171_939)


def registry_with_calibration(model, calibration):
    metadata = {
        frota: make_metadata(frota, flat_fare_calibration=calibration)
        for frota in ("yellow", "green")
    }
    return StubRegistry({"yellow": model, "green": model}, metadata)


class TestServiceArea:
    """RF04: a regulação decide onde cada frota embarca, e a decisão é do servidor.

    A interface já desabilita o marcador, mas interface não é contrato: uma requisição montada
    fora dela declarava `taxi_type: green` com embarque em Midtown e recebia preço.
    """

    def test_green_cannot_pick_up_inside_the_hail_exclusionary_zone(self, client_with_model):
        resposta = client_with_model.post(
            "/predict",
            json={**VALID_PAYLOAD, "taxi_type": "green", "PULocationID": MIDTOWN_ZONE_ID},
        )
        assert resposta.status_code == 422
        assert "Hail Exclusionary Zone" in resposta.json()["detail"]

    def test_green_still_picks_up_in_manhattan_above_the_line(self, client_with_model):
        """Harlem, Washington Heights e Inwood são de boro taxi: bloqueá-los negaria serviço."""
        assert MANHATTAN_ABOVE_HEZ_ZONE_ID not in HAIL_EXCLUSIONARY_ZONE_IDS
        resposta = client_with_model.post(
            "/predict",
            json={
                **VALID_PAYLOAD,
                "taxi_type": "green",
                "PULocationID": MANHATTAN_ABOVE_HEZ_ZONE_ID,
            },
        )
        assert resposta.status_code == 200

    def test_yellow_picks_up_anywhere_in_manhattan(self, client_with_model):
        resposta = client_with_model.post(
            "/predict",
            json={**VALID_PAYLOAD, "taxi_type": "yellow", "PULocationID": MIDTOWN_ZONE_ID},
        )
        assert resposta.status_code == 200

    @pytest.mark.parametrize("frota", ["yellow", "green"])
    def test_newark_is_refused_as_an_origin_for_every_fleet(self, client_with_model, frota):
        resposta = client_with_model.post(
            "/predict",
            json={**VALID_PAYLOAD, "taxi_type": frota, "PULocationID": NEWARK_ZONE_ID},
        )
        assert resposta.status_code == 422
        assert "Newark" in resposta.json()["detail"]

    def test_newark_remains_valid_as_a_destination(self, client_with_model):
        """A regra é sobre embarque: o táxi de Nova York leva até lá, só não pega lá."""
        resposta = client_with_model.post(
            "/predict",
            json={**VALID_PAYLOAD, "taxi_type": "yellow", "DOLocationID": NEWARK_ZONE_ID},
        )
        assert resposta.status_code == 200


class TestOptionalTripDuration:
    """A duração não é feature do modelo e não pode ser exigida de quem chama."""

    def test_a_request_without_the_duration_is_accepted(self, client_with_model):
        resposta = client_with_model.post("/predict", json=VALID_PAYLOAD)
        assert resposta.status_code == 200

    def test_sending_it_does_not_change_the_prediction(self, client_with_model):
        """Se mudasse, ela estaria entrando no modelo por algum caminho."""
        sem = client_with_model.post("/predict", json=VALID_PAYLOAD).json()["predicted_fare"]
        com = client_with_model.post(
            "/predict", json={**VALID_PAYLOAD, "trip_duration_minutes": 45.0}
        ).json()["predicted_fare"]
        assert sem == com

    def test_a_negative_duration_is_still_rejected(self, client_with_model):
        """Opcional não é sinônimo de sem validação."""
        resposta = client_with_model.post(
            "/predict", json={**VALID_PAYLOAD, "trip_duration_minutes": -1.0}
        )
        assert resposta.status_code == 422


class TestFlatFareRule:
    def _predict(self, payload, calibration=CALIBRATED, predicted=120.0):
        registry = registry_with_calibration(make_mock_model(predicted), calibration)
        with client_with(registry) as client:
            resposta = client.post("/predict", json=payload)
        assert resposta.status_code == 200
        return resposta.json()

    def test_jfk_to_manhattan_returns_the_regulated_fare(self):
        corpo = self._predict(
            {**VALID_PAYLOAD, "PULocationID": JFK_ZONE, "DOLocationID": TIMES_SQUARE_ZONE}
        )
        esperado = JFK_FLAT_FARE_AMOUNT + CALIBRATED.mean_excess
        assert corpo["predicted_fare"] == pytest.approx(esperado)
        assert corpo["pricing_rule"] == "jfk_flat_fare"

    def test_the_rule_works_in_both_directions(self):
        """Olhar só o destino era o defeito D-02: a tarifa fixa vale nos dois sentidos."""
        corpo = self._predict(
            {**VALID_PAYLOAD, "PULocationID": TIMES_SQUARE_ZONE, "DOLocationID": JFK_ZONE}
        )
        assert corpo["pricing_rule"] == "jfk_flat_fare"

    def test_a_forged_rate_code_does_not_buy_the_flat_fare(self):
        """O RatecodeID vem no corpo da requisição; a regra confere as zonas, não o campo."""
        corpo = self._predict(
            {
                **VALID_PAYLOAD,
                "PULocationID": QUEENS_ZONE,
                "DOLocationID": QUEENS_ZONE,
                "RatecodeID": 2,
            },
            predicted=9.75,
        )
        assert corpo["predicted_fare"] == pytest.approx(9.75)
        assert corpo["pricing_rule"] == "model"

    def test_without_calibration_the_rule_abstains(self):
        corpo = self._predict(
            {**VALID_PAYLOAD, "PULocationID": JFK_ZONE, "DOLocationID": TIMES_SQUARE_ZONE},
            calibration=None,
        )
        assert corpo["predicted_fare"] == pytest.approx(120.0)

    def test_an_implausible_prediction_is_raised_to_the_floor(self):
        corpo = self._predict({**VALID_PAYLOAD}, predicted=0.40)
        assert corpo["predicted_fare"] == pytest.approx(MIN_PLAUSIBLE_TOTAL_AMOUNT)
        assert corpo["pricing_rule"] == "minimum_fare"


# ---------------------------------------------------------------------------
# 3. GET /model-info alimenta o painel de explicação
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_it_serves_the_coefficients_the_panel_needs(self, client_with_full_model):
        corpo = client_with_full_model.get("/model-info/yellow").json()
        assert "trip_distance" in corpo["coefficients"]
        assert isinstance(corpo["intercept"], float)
        assert corpo["rmse"] > 0
        assert corpo["training_samples"] > 0

    def test_an_invalid_fleet_is_rejected(self, client_with_full_model):
        assert client_with_full_model.get("/model-info/blue").status_code == 422

    def test_without_a_model_the_answer_is_503(self, client_no_model):
        assert client_no_model.get("/model-info/yellow").status_code == 503


# ---------------------------------------------------------------------------
# 4. Cada predição servida vira registro auditável
# ---------------------------------------------------------------------------


class TestPredictionLog:
    def test_the_log_records_the_estimate_and_the_rule_that_decided(self, log_capture):
        """Auditar a camada exige ver o que o modelo disse antes de a regra decidir."""
        registry = registry_with_calibration(make_mock_model(120.0), CALIBRATED)
        with client_with(registry) as client:
            client.post(
                "/predict",
                json={**VALID_PAYLOAD, "PULocationID": JFK_ZONE, "DOLocationID": TIMES_SQUARE_ZONE},
            )

        registro = log_capture.records("prediction_served")[0]
        assert registro["model_estimate"] == pytest.approx(120.0)
        assert registro["pricing_rule"] == "jfk_flat_fare"


# ---------------------------------------------------------------------------
# 5. Sondas de liveness e readiness
# ---------------------------------------------------------------------------


class TestHealth:
    def test_it_answers_ok_even_without_any_model(self, client_no_model):
        """Liveness pergunta "o processo travou?". Um artefato ausente não se resolve
        reiniciando, então health não pode reprovar por causa dele."""
        resposta = client_no_model.get("/health")
        assert resposta.status_code == 200
        assert resposta.json()["status"] == "ok"


class TestReadiness:
    def test_it_reports_the_fleets_and_versions_being_served(self, client_with_model):
        resposta = client_with_model.get("/ready")
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert corpo["status"] == "ready"
        assert corpo["loaded_fleets"] == ["green", "yellow"]
        assert corpo["model_versions"]["yellow"].startswith("yellow-")

    def test_without_any_model_it_refuses_traffic(self, client_no_model):
        """A API sobe e responde, mas devolve 503 em toda predição: anunciar-se pronta
        colocaria uma instância inútil atrás do balanceador."""
        assert client_no_model.get("/ready").status_code == 503

    def test_one_loaded_fleet_is_enough_to_be_ready(self):
        """A outra frota falha só no próprio `taxi_type`, não na aplicação inteira."""
        registry = StubRegistry({"green": make_mock_model()}, {"green": make_metadata("green")})
        with client_with(registry) as client:
            resposta = client.get("/ready")
        assert resposta.status_code == 200
        assert resposta.json()["loaded_fleets"] == ["green"]
