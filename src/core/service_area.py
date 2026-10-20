"""Onde cada frota pode aceitar embarque, segundo a regulação da TLC.

A regra vivia apenas no navegador, onde serve para desabilitar o marcador antes do usuário
arrastá-lo. Isso basta para a interface e não basta para o contrato: uma requisição montada
fora dela declarava `taxi_type: green` com embarque em Midtown e recebia preço, porque o
backend não conhecia a restrição.

É o mesmo raciocínio que a camada de regras já aplica à tarifa fixa, onde o regime tarifário é
derivado das zonas e nunca do campo recebido. Decisão que depende de regulação não pode ficar
do lado de quem faz a chamada.
"""

from src.core.constants import (
    HAIL_EXCLUSIONARY_ZONE_IDS,
    OUT_OF_STATE_LOCATION_IDS,
    TaxiType,
)

GREEN_FLEET: TaxiType = "green"


def pickup_rejection_reason(taxi_type: str, pickup_location_id: int) -> str | None:
    """Return why the fleet may not pick up in that zone, or None when it may.

    Devolve o motivo em vez de um booleano porque quem recusa precisa dizer qual regra barrou:
    "fora da área" e "frota errada para esta área" levam o usuário a correções diferentes.
    """
    if pickup_location_id in OUT_OF_STATE_LOCATION_IDS:
        return (
            "Embarque em Newark não é permitido: a zona fica em Nova Jersey e um táxi de "
            "Nova York só pode desembarcar lá."
        )

    if taxi_type == GREEN_FLEET and pickup_location_id in HAIL_EXCLUSIONARY_ZONE_IDS:
        return (
            "A frota green não pode embarcar nesta zona: ela está na Hail Exclusionary Zone, "
            "Manhattan ao sul da W 110th St e da E 96th St. Use a frota yellow, ou um embarque "
            "ao norte dessa linha."
        )

    return None
