from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.months import now_in_nyc


class ModelNamespaceAllowed(BaseModel):
    """Base dos schemas que expõem `model_version`.

    `model_` é namespace reservado do Pydantic v2 e cada campo com esse prefixo emite um aviso.
    Renomear o campo mudaria o contrato público da API por causa de uma convenção interna da
    biblioteca; liberar o namespace é a troca certa.
    """

    model_config = ConfigDict(protected_namespaces=())


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
    # Não é feature do modelo: a duração real só existe depois da corrida, e derivá-la da
    # distância a torna colinear com ela. Continua aceita porque o log de predição a registra
    # como contexto de auditoria, e é opcional porque exigir do chamador um número que o
    # modelo ignora só o levaria a inventá-lo, e a invenção acabaria gravada no log.
    trip_duration_minutes: float | None = Field(
        default=None, ge=0.0, description="Duração estimada da corrida em minutos, se conhecida"
    )

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


class PredictResponse(ModelNamespaceAllowed):
    predicted_fare: float = Field(..., description="Tarifa estimada em dólares americanos (USD)")
    taxi_type: str = Field(..., description="Tipo de táxi utilizado no cálculo")
    model_version: str = Field(..., description="Identificador da versão do modelo")
    pricing_rule: str = Field(
        ...,
        description="Regra que determinou o valor: modelo, tarifa fixa do JFK ou piso mínimo",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado do processo que atende as requisições")


class ReadinessResponse(ModelNamespaceAllowed):
    status: str = Field(..., description="Se a aplicação já pode receber tráfego")
    loaded_fleets: list[str] = Field(
        ..., description="Frotas com modelo carregado em memória neste instante"
    )
    model_versions: dict[str, str] = Field(
        ..., description="Versão que cada frota está servindo"
    )


class PlaceResponse(BaseModel):
    display_name: str = Field(..., description="Nome do local como o Nominatim o descreve")
    lat: float = Field(..., description="Latitude do local")
    lon: float = Field(..., description="Longitude do local")


class GeocodeSearchResponse(BaseModel):
    results: list[PlaceResponse] = Field(..., description="Locais encontrados, do mais relevante")


class ReverseGeocodeResponse(BaseModel):
    display_name: str = Field(..., description="Endereço correspondente às coordenadas")


class ModelInfoResponse(ModelNamespaceAllowed):
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
