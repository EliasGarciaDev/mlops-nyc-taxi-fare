"""Camada de regras de domínio tarifário aplicadas após a predição do modelo.

Um modelo linear soma a contribuição da distância em toda corrida. Numa tarifa regulada por
constante isso é estruturalmente errado: JFK↔Manhattan custa o mesmo com 15 ou com 25 milhas, e
o modelo não tem como representar essa regra a partir das features que recebe. Medido em
2024-03, o erro do modelo nesse segmento é de US$ 9,69 contra US$ 4,71 da regra, e a camada
vale −4,50% de RMSE no mês inteiro, bem acima da margem de 1% que o gate de promoção exige.

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
    TARGET_COLUMN,
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
    explicação em vez de somar coeficientes que não produziram o número.
    """

    MODEL = "model"
    JFK_FLAT_FARE = "jfk_flat_fare"
    MINIMUM_FARE = "minimum_fare"


@dataclass
class FlatFareCalibration:
    """Calibração do excedente médio da tarifa fixa em relação ao taxímetro regulado.

    A tarifa fixa é só a parcela de taxímetro. Sobre ela ainda incidem as sobretaxas da TLC, os
    pedágios da rota escolhida e as taxas de aeroporto, juntos, cerca de US$ 12 em média contra
    os US$ 70,00 regulados. A gorjeta não entra: ela saiu do alvo do modelo, porque é ato
    voluntário do passageiro e não preço.

    O excedente varia com a rota e com a política tarifária, então viaja no artefato e é
    recalibrado a cada retreino. Congelá-lo no código-fonte faria a regra citar um valor que
    envelhece sem ninguém perceber.
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
    taxímetro seria pior que não aplicar regra nenhuma: sobre as 107.246 corridas de tarifa fixa
    de 2024-03 isso mede RMSE 13,16 contra 9,69 do modelo sozinho, porque devolveria o taxímetro
    onde o contrato promete o total regulado. Com o excedente calibrado o RMSE cai para 4,71.
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

    Calibra sobre as corridas que as zonas confirmam, e não sobre as que declaram o código 2:
    é a única população que coincide com aquela onde a regra vai de fato ser aplicada, já que a
    aplicação também decide pelas zonas. Sobre 2024-01/02 o excedente medido assim é de
    US$ 12,24, valor que viaja no artefato e é refeito a cada retreino.
    """
    required = {"PULocationID", "DOLocationID", TARGET_COLUMN}
    if not required.issubset(frame.columns):
        return None

    confirmed = frame[
        _zones_confirm_flat_fare(frame["PULocationID"], frame["DOLocationID"])
    ]
    if len(confirmed) < MIN_CALIBRATION_SAMPLES:
        return None

    excess = float(confirmed[TARGET_COLUMN].mean()) - JFK_FLAT_FARE_AMOUNT
    return FlatFareCalibration(mean_excess=excess, sample_count=len(confirmed))


def _zones_confirm_flat_fare(pickup: "pd.Series[int]", dropoff: "pd.Series[int]") -> "pd.Series[bool]":
    """Vectorized counterpart of `is_flat_fare_trip`, for calibration over a whole month."""
    touches_jfk = (pickup == JFK_LOCATION_ID) | (dropoff == JFK_LOCATION_ID)
    touches_manhattan = pickup.isin(MANHATTAN_LOCATION_IDS) | dropoff.isin(MANHATTAN_LOCATION_IDS)
    return touches_jfk & touches_manhattan
