"""Camada de regras de domínio tarifário aplicadas após a predição do modelo.

Um modelo linear soma a contribuição da distância em toda corrida. Numa tarifa regulada por
constante isso é estruturalmente errado: JFK↔Manhattan custa o mesmo com 15 ou com 25 milhas, e
o modelo não tem como representar essa regra a partir das features que recebe. Medido em
2024-03, o erro do modelo nesse segmento é de US$ 12,94 contra US$ 8,88 da regra - e a camada
vale −3,47% de RMSE no mês inteiro, bem acima da margem de 1% que o gate do ADR 0015 exige.

Esta camada é também onde mora o piso de tarifa: a regressão não conhece o limite inferior do
taxímetro e nada a impede de devolver valor implausível numa corrida curta.
"""

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from src.core.constants import (
    JFK_FLAT_FARE_AMOUNT,
    JFK_LOCATION_ID,
    MANHATTAN_LOCATION_IDS,
    MIN_PLAUSIBLE_TOTAL_AMOUNT,
)
from src.core.exceptions import CorruptedMetadataError

# Abaixo disto a média do excedente é ruído e a camada prefere não existir: sem calibração
# `apply_fare_rules` devolve a predição do modelo, que é o comportamento anterior a ela.
MIN_CALIBRATION_SAMPLES = 1_000


class PricingRule(StrEnum):
    """Regra de tarifação aplicada para justificar o valor final da corrida.

    O painel XAI decompõe a predição em intercepto mais contribuição por coeficiente. Onde a
    regra decidiu, essa decomposição descreveria um mecanismo que não agiu - a distância não
    entra numa tarifa fixa. Dizer qual regra valeu é o que permite ao painel trocar de
    explicação em vez de somar coeficientes que não produziram o número (F-01, F-03).
    """

    MODEL = "model"
    JFK_FLAT_FARE = "jfk_flat_fare"
    MINIMUM_FARE = "minimum_fare"


@dataclass
class FlatFareCalibration:
    """Calibração do excedente médio da tarifa fixa em relação ao taxímetro regulado.

    A tarifa fixa é só a parcela de taxímetro. Sobre ela ainda incidem as sobretaxas da TLC, o
    pedágio da rota escolhida e a gorjeta - juntos, US$ 24,06 em média contra os US$ 70,00
    regulados. A gorjeta é a maior parcela e a mais volátil: o desvio padrão do total cai de
    US$ 8,45 para US$ 3,54 quando ela sai da conta.

    Este excedente é comportamental e envelhece, então viaja no artefato e é recalibrado a cada
    retreino, como o baseline de drift do ADR 0014. Congelá-lo no
    código-fonte reintroduziria o problema que o ADR 0003 resolveu.
    """

    mean_excess: float
    sample_count: int

    def to_dict(self) -> dict[str, object]:
        return {"mean_excess": self.mean_excess, "sample_count": self.sample_count}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FlatFareCalibration":
        excess = payload.get("mean_excess")
        count = payload.get("sample_count")
        if not isinstance(excess, (int, float)) or isinstance(excess, bool):
            raise CorruptedMetadataError(
                f"Campo 'mean_excess' deveria ser numérico, veio {type(excess).__name__}."
            )
        if not isinstance(count, int) or isinstance(count, bool):
            raise CorruptedMetadataError(
                f"Campo 'sample_count' deveria ser inteiro, veio {type(count).__name__}."
            )
        return cls(mean_excess=float(excess), sample_count=count)


def is_flat_fare_trip(pickup_location_id: int, dropoff_location_id: int) -> bool:
    """Verifica se os pontos da corrida caracterizam a tarifa fixa JFK ↔ Manhattan.

    A regra é derivada das zonas e nunca do `RatecodeID` informado. Nos dados já limpos da
    própria TLC, 5,71% das corridas marcadas com o código 2 em 2024-03 não ligam o JFK a
    Manhattan, e 2,28% das que ligam não declaram o código - o campo é digitado e erra nos dois
    sentidos. Numa requisição forjada ele é pior que impreciso: aceitar o código
    declarado deixaria qualquer trajeto reivindicar a tarifa do aeroporto.
    """
    ends = (pickup_location_id, dropoff_location_id)
    touches_jfk = JFK_LOCATION_ID in ends
    touches_manhattan = any(end in MANHATTAN_LOCATION_IDS for end in ends)
    return touches_jfk and touches_manhattan


@dataclass
class FareDecision:
    """Resultado da tarifação com o valor calculado e a regra correspondente."""

    amount: float
    rule: PricingRule


def apply_fare_rules(
    predicted_total: float,
    pickup_location_id: int,
    dropoff_location_id: int,
    calibration: FlatFareCalibration | None,
) -> FareDecision:
    """Aplica regras reguladas de tarifa quando a corrida se enquadra em condições fixas.

    Sem calibração o valor do excedente é desconhecido, e aplicar apenas os US$ 70,00 de
    taxímetro seria pior que não aplicar regra nenhuma - mediria US$ 25,73 de RMSE contra os
    US$ 12,94 do modelo, porque devolveria o taxímetro onde o contrato promete o total pago.
    """
    if calibration is not None and is_flat_fare_trip(pickup_location_id, dropoff_location_id):
        return FareDecision(
            amount=JFK_FLAT_FARE_AMOUNT + calibration.mean_excess,
            rule=PricingRule.JFK_FLAT_FARE,
        )

    if predicted_total < MIN_PLAUSIBLE_TOTAL_AMOUNT:
        return FareDecision(amount=MIN_PLAUSIBLE_TOTAL_AMOUNT, rule=PricingRule.MINIMUM_FARE)

    return FareDecision(amount=predicted_total, rule=PricingRule.MODEL)


def calibrate_flat_fare(frame: pd.DataFrame) -> FlatFareCalibration | None:
    """Calcula o excedente médio da tarifa fixa a partir dos dados de treino.

    Calibra sobre as corridas que **as zonas** confirmam, e não sobre as que declaram o código 2.
    As três populações foram medidas em 2024-01/02 e a diferença entre elas é pequena - US$ 23,46
    a 24,06 de excedente -, mas a que as zonas confirmam é a única que coincide com a população
    onde a regra vai de fato ser aplicada, e foi também a de menor erro (RMSE 8,8810).
    """
    required = {"PULocationID", "DOLocationID", "total_amount"}
    if not required.issubset(frame.columns):
        return None

    confirmed = frame[
        _zones_confirm_flat_fare(frame["PULocationID"], frame["DOLocationID"])
    ]
    if len(confirmed) < MIN_CALIBRATION_SAMPLES:
        return None

    excess = float(confirmed["total_amount"].mean()) - JFK_FLAT_FARE_AMOUNT
    return FlatFareCalibration(mean_excess=excess, sample_count=len(confirmed))


def _zones_confirm_flat_fare(pickup: "pd.Series[int]", dropoff: "pd.Series[int]") -> "pd.Series[bool]":
    """Vectorized counterpart of `is_flat_fare_trip`, for calibration over a whole month."""
    touches_jfk = (pickup == JFK_LOCATION_ID) | (dropoff == JFK_LOCATION_ID)
    touches_manhattan = pickup.isin(MANHATTAN_LOCATION_IDS) | dropoff.isin(MANHATTAN_LOCATION_IDS)
    return touches_jfk & touches_manhattan
