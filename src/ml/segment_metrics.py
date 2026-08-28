"""Métricas de erro por região de desembarque, para a incerteza que a interface exibe.

O painel exibia um único número de margem, o RMSE do modelo inteiro, igual em toda corrida.
Esse número é falso fora de Manhattan: medido por região de desembarque na frota amarela, o
erro vai de US$ 4,47 em Manhattan a US$ 20,54 em Staten Island, contra um agregado de US$ 5,40.
Exibir o agregado numa corrida para Staten Island promete uma precisão que o sistema não tem.

O erro por região é medido na mesma janela de validação que produz a métrica agregada e viaja
dentro do artefato, junto do baseline de drift e da calibração da tarifa fixa. O motivo é o
mesmo nos três casos: são números que envelhecem com o modelo, e fixá-los no código faria a
interface citar a incerteza de uma versão que não está mais servindo.
"""

from dataclasses import dataclass

import pandas as pd

from src.core.constants import LOCATION_IDS_BY_BOROUGH, TARGET_COLUMN
from src.core.exceptions import CorruptedMetadataError
from src.ml.protocols import Predictor
from src.ml.trainer import evaluate_model

# Abaixo disto o RMSE do recorte oscila mais que a diferença que ele deveria comunicar, e
# exibi-lo trocaria um número errado por outro. O segmento sem massa cai na margem agregada,
# que é o comportamento anterior a esta medição.
MIN_SEGMENT_SAMPLES_FOR_MARGIN = 500


@dataclass
class SegmentErrors:
    """RMSE de validação por borough de desembarque com volume suficiente."""

    rmse_by_borough: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"rmse_by_borough": dict(self.rmse_by_borough)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SegmentErrors":
        raw = payload.get("rmse_by_borough")
        if not isinstance(raw, dict):
            raise CorruptedMetadataError(
                f"Campo 'rmse_by_borough' deveria ser um objeto, veio {type(raw).__name__}."
            )

        parsed: dict[str, float] = {}
        for borough, value in raw.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise CorruptedMetadataError(
                    f"RMSE do borough '{borough}' deveria ser numérico, "
                    f"veio {type(value).__name__}."
                )
            parsed[str(borough)] = float(value)
        return cls(rmse_by_borough=parsed)


def measure_segment_errors(model: Predictor, validation_df: pd.DataFrame) -> SegmentErrors | None:
    """Avalia o RMSE separadamente por borough de desembarque na janela de validação."""
    if "DOLocationID" not in validation_df.columns or TARGET_COLUMN not in validation_df.columns:
        return None

    measured: dict[str, float] = {}
    for borough, zone_ids in LOCATION_IDS_BY_BOROUGH.items():
        segment = validation_df[validation_df["DOLocationID"].isin(zone_ids)]
        if len(segment) < MIN_SEGMENT_SAMPLES_FOR_MARGIN:
            continue
        measured[borough] = evaluate_model(model, segment).rmse

    return SegmentErrors(rmse_by_borough=measured) if measured else None
