from time import perf_counter
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.geocoding import GeocodingClient
from src.api.middleware import elapsed_ms
from src.api.model_registry import ModelRegistry
from src.api.schemas import (
    GeocodeSearchResponse,
    ModelInfoResponse,
    PlaceResponse,
    PredictRequest,
    PredictResponse,
    ReverseGeocodeResponse,
)
from src.core.constants import (
    AIRPORT_LOCATION_IDS,
    CRZ_LOCATION_IDS,
    FEATURE_ORDER,
    RATECODE_FEATURES,
    VALID_TAXI_TYPES,
    WEEKEND_START_WEEKDAY,
)
from src.core.exceptions import GeocodingError
from src.core.log import get_logger
from src.ml.fare_rules import apply_fare_rules
from src.ml.protocols import LinearModel
from src.ml.registry import ModelMetadata

logger = get_logger(__name__)

router = APIRouter()

MODEL_UNAVAILABLE_STATUS = 503
INVALID_TAXI_TYPE_STATUS = 422
GEOCODING_UNAVAILABLE_STATUS = 503

MAX_QUERY_LENGTH = 120
MIN_LATITUDE, MAX_LATITUDE = -90.0, 90.0
MIN_LONGITUDE, MAX_LONGITUDE = -180.0, 180.0

# Um cliente por processo, porque o cache e o limite de uma requisição por segundo só fazem
# sentido compartilhados: quem o Nominatim vê é o nosso endereço, não cada usuário.
_geocoder = GeocodingClient()


def get_geocoder() -> GeocodingClient:
    return _geocoder


GeocoderDep = Annotated[GeocodingClient, Depends(get_geocoder)]

# Uma instância por processo. O ciclo autônomo troca o modelo em disco todo mês, e é o
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Provide the process-wide registry, refreshing it when the interval has elapsed.

    Injetado em vez de acessado como global: é o que permite aos testes fornecerem o próprio
    registro sem remendar nomes de módulo, e o que deixa a recarga acontecer no caminho da
    requisição em vez de depender de reinício.
    """
    _registry.refresh_if_due()
    return _registry


RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]


def load_models() -> None:
    """Carga inicial, na subida da aplicação."""
    _registry.refresh()


def require_model(registry: ModelRegistry, taxi_type: str) -> tuple[LinearModel, ModelMetadata]:
    """Devolve o modelo e os metadados da frota, ou recusa servi-la."""
    model = registry.model_of(taxi_type)
    metadata = registry.metadata_of(taxi_type)
    if model is None or metadata is None:
        raise HTTPException(
            status_code=MODEL_UNAVAILABLE_STATUS,
            detail=f"Modelo '{taxi_type}' não está carregado.",
        )
    return model, metadata


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, registry: RegistryDep) -> PredictResponse:
    model, metadata = require_model(registry, request.taxi_type)

    pickup_dt = request.pickup_datetime
    is_airport = int(
        request.PULocationID in AIRPORT_LOCATION_IDS
        or request.DOLocationID in AIRPORT_LOCATION_IDS
    )

    feature_row = {
        "trip_distance": request.trip_distance,
        "hour_of_day": pickup_dt.hour,
        "day_of_week": pickup_dt.weekday(),
        "is_weekend": int(pickup_dt.weekday() >= WEEKEND_START_WEEKDAY),
        "trip_duration_minutes": request.trip_duration_minutes,
        "is_airport_trip": is_airport,
        "is_congestion_zone": int(
            request.PULocationID in CRZ_LOCATION_IDS
            or request.DOLocationID in CRZ_LOCATION_IDS
        ),
 # Um código sem indicador próprio - o padrão e o group ride - fica com todos zerados,
 # que é exatamente a categoria de referência que o treino usou.
        **{
            feature_name: int(request.RatecodeID == code)
            for code, feature_name in RATECODE_FEATURES.items()
        },
    }

    features_df = pd.DataFrame([feature_row])[FEATURE_ORDER]
    started_at = perf_counter()
    estimated = float(model.predict(features_df)[0])

 # A tarifa fixa do JFK é regulada por constante: onde ela vale, o valor não se estima.
    decision = apply_fare_rules(
        estimated,
        request.PULocationID,
        request.DOLocationID,
        metadata.flat_fare_calibration,
    )

 # Este evento é o registro de predição da Fase 4: os campos são exatamente os da tabela
 # prediction_logs, de modo que persistir passe a ser acrescentar um destino ao log.
    logger.info(
        "prediction_served",
        extra={
            "taxi_type": request.taxi_type,
            "trip_distance": request.trip_distance,
            "passenger_count": request.passenger_count,
            "pu_location_id": request.PULocationID,
            "do_location_id": request.DOLocationID,
            "ratecode_id": request.RatecodeID,
            "trip_duration_minutes": request.trip_duration_minutes,
            "pickup_datetime": pickup_dt.isoformat(),
            "predicted_fare": decision.amount,
            "model_estimate": estimated,
            "pricing_rule": decision.rule.value,
            "model_version": metadata.model_version,
            "inference_latency_ms": elapsed_ms(started_at),
        },
    )

    return PredictResponse(
        predicted_fare=decision.amount,
        taxi_type=request.taxi_type,
        model_version=metadata.model_version,
        pricing_rule=decision.rule.value,
    )


@router.get("/model-info/{taxi_type}", response_model=ModelInfoResponse)
def model_info(taxi_type: str, registry: RegistryDep) -> ModelInfoResponse:
    if taxi_type not in VALID_TAXI_TYPES:
        raise HTTPException(
            status_code=INVALID_TAXI_TYPE_STATUS,
            detail=f"taxi_type deve ser 'yellow' ou 'green'. Recebido: '{taxi_type}'.",
        )

    # Um estimador sem coeficientes levanta AttributeError e a rota responde 503, em vez
    # de explicar errado.
    linear_model, metadata = require_model(registry, taxi_type)
    try:
        feature_names = list(linear_model.feature_names_in_)
        coefficients = {
            name: float(coef)
            for name, coef in zip(feature_names, linear_model.coef_, strict=True)
        }
        intercept = float(linear_model.intercept_)
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=MODEL_UNAVAILABLE_STATUS,
            detail=f"Modelo não possui metadados de coeficientes: {exc}",
        ) from exc

    return ModelInfoResponse(
        taxi_type=taxi_type,
        rmse_by_borough=(
            dict(metadata.segment_errors.rmse_by_borough)
            if metadata.segment_errors is not None
            else {}
        ),
        intercept=intercept,
        coefficients=coefficients,
 # O erro divulgado é o de validação: é o único que diz algo sobre dados não vistos.
        rmse=metadata.validation_metrics.rmse,
        training_samples=metadata.train_metrics.sample_count,
        model_version=metadata.model_version,
    )


@router.get("/geocode/search", response_model=GeocodeSearchResponse)
def geocode_search(
    geocoder: GeocoderDep,
    q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_LENGTH)],
) -> GeocodeSearchResponse:
    """Proxy the address search, identifying the application and honouring the rate limit.

    O navegador não conseguia fazer isso: `User-Agent` é header proibido na Fetch API e era
    descartado em silêncio, então as buscas chegavam anônimas ao Nominatim - o caminho mais
    curto para o bloqueio do endereço logo quando houvesse gente usando.
    """
    if not q.strip():
        raise HTTPException(
            status_code=INVALID_TAXI_TYPE_STATUS, detail="A busca não pode ser vazia."
        )
    try:
        places = geocoder.search(q)
    except GeocodingError as failure:
        raise HTTPException(
            status_code=GEOCODING_UNAVAILABLE_STATUS,
            detail="Serviço de busca de endereços indisponível. Tente novamente em instantes.",
        ) from failure

    return GeocodeSearchResponse(
        results=[
            PlaceResponse(display_name=place.display_name, lat=place.lat, lon=place.lon)
            for place in places
        ]
    )


@router.get("/geocode/reverse", response_model=ReverseGeocodeResponse)
def geocode_reverse(
    geocoder: GeocoderDep,
    lat: Annotated[float, Query(ge=MIN_LATITUDE, le=MAX_LATITUDE)],
    lon: Annotated[float, Query(ge=MIN_LONGITUDE, le=MAX_LONGITUDE)],
) -> ReverseGeocodeResponse:
    """Describe a coordinate in words, through the same proxy."""
    try:
        return ReverseGeocodeResponse(display_name=geocoder.reverse(lat, lon))
    except GeocodingError as failure:
        raise HTTPException(
            status_code=GEOCODING_UNAVAILABLE_STATUS,
            detail="Serviço de endereços indisponível. Tente novamente em instantes.",
        ) from failure
