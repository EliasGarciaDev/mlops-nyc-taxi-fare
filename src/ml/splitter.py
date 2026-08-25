from datetime import datetime

import pandas as pd

from src.core.constants import PICKUP_DATETIME_COLUMN
from src.core.exceptions import InvalidSplitError


def split_by_cutoff(df: pd.DataFrame, cutoff: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition trip records into a past training set and a future validation set.

    Corridas no instante exato do corte pertencem à validação: o modelo nunca deve
    enxergar nada que, em produção, ainda não teria acontecido.
    """
    if PICKUP_DATETIME_COLUMN not in df.columns:
        raise ValueError(
            f"Coluna obrigatória ausente para split temporal: '{PICKUP_DATETIME_COLUMN}'."
        )

    if df.empty:
        return df.copy(), df.copy()

    is_training = df[PICKUP_DATETIME_COLUMN] < cutoff
    train = df.loc[is_training].reset_index(drop=True)
    validation = df.loc[~is_training].reset_index(drop=True)

    if train.empty:
        raise InvalidSplitError(
            f"Corte {cutoff.isoformat()} não deixou nenhuma corrida para treino. "
            f"A corrida mais antiga é de {df[PICKUP_DATETIME_COLUMN].min()}."
        )
    if validation.empty:
        raise InvalidSplitError(
            f"Corte {cutoff.isoformat()} não deixou nenhuma corrida para validação. "
            f"A corrida mais recente é de {df[PICKUP_DATETIME_COLUMN].max()}."
        )

    return train, validation
