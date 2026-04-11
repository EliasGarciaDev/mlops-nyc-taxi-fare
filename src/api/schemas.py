from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.core.months import now_in_nyc


class PredictRequest(BaseModel):
    taxi_type: Literal["yellow", "green"]
    trip_distance: float = Field(..., ge=0.1, le=100.0, description="Distância estimada da corrida em milhas")
    passenger_count: int = Field(..., ge=1, le=6, description="Número de passageiros")
    PULocationID: int = Field(..., ge=1, le=265, description="ID da zona de embarque no mapa TLC")
    DOLocationID: int = Field(..., ge=1, le=265, description="ID da zona de desembarque no mapa TLC")
    RatecodeID: int = Field(..., ge=1, le=6, description="Código de tarifa TLC")
    pickup_datetime: datetime = Field(
        ..., description="Data e hora do embarque, em hora local de Nova York"
    )
    trip_duration_minutes: float = Field(..., ge=0.0, description="Duração estimada da corrida em minutos")

    @field_validator("pickup_datetime")
    @classmethod
    def validate_pickup_not_in_future(cls, value: datetime) -> datetime:
        """Compare against the city clock, which is the convention of the field.

        Ler o horário como UTC abria uma janela do tamanho do offset da cidade: um embarque
        marcado para daqui a três horas passava como se já tivesse acontecido.
        """
        local_value = now_in_nyc(value) if value.tzinfo else value
        if local_value > now_in_nyc():
            raise ValueError(
                f"pickup_datetime não pode ser no futuro: {local_value.isoformat()} "
                f"é depois de {now_in_nyc().isoformat()} em Nova York."
            )
        return value


class PredictResponse(BaseModel):
    predicted_fare: float = Field(..., description="Tarifa estimada em dólares americanos (USD)")
    taxi_type: str = Field(..., description="Tipo de táxi utilizado no cálculo")
    model_version: str = Field(..., description="Identificador da versão do modelo")
    pricing_rule: str = Field(
        ...,
        description="Regra que determinou o valor: modelo, tarifa fixa do JFK ou piso mínimo",
    )


class PlaceResponse(BaseModel):
    display_name: str = Field(..., description="Nome do local como o Nominatim o descreve")
    lat: float = Field(..., description="Latitude do local")
    lon: float = Field(..., description="Longitude do local")


class GeocodeSearchResponse(BaseModel):
    results: list[PlaceResponse] = Field(..., description="Locais encontrados, do mais relevante")


class ReverseGeocodeResponse(BaseModel):
    display_name: str = Field(..., description="Endereço correspondente às coordenadas")


class ModelInfoResponse(BaseModel):
    taxi_type: str = Field(..., description="Tipo da frota do modelo")
    intercept: float = Field(..., description="Valor do intercepto linear")
    coefficients: dict[str, float] = Field(..., description="Pesos e coeficientes de cada feature")
    rmse: float = Field(..., description="Erro quadrático médio obtido na validação")
    training_samples: int = Field(..., description="Corridas usadas no treino da versão ativa")
    model_version: str = Field(..., description="Arquivo ou tag de versão do modelo")
    rmse_by_borough: dict[str, float] = Field(
        default_factory=dict,
        description="Erro de validação por borough de desembarque, quando medido",
    )
