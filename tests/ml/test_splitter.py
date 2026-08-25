from datetime import datetime

import pandas as pd
import pytest

from src.core.exceptions import InvalidSplitError
from src.ml.splitter import split_by_cutoff

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_df(*pickup_dates: datetime) -> pd.DataFrame:
    """Cria um DataFrame com uma corrida por data de embarque informada."""
    return pd.DataFrame(
        {
            "pickup_datetime": list(pickup_dates),
            "fare_amount": [10.0] * len(pickup_dates),
            "trip_distance": [2.0] * len(pickup_dates),
        }
    )


JANUARY = datetime(2024, 1, 15, 10, 0, 0)
FEBRUARY = datetime(2024, 2, 15, 10, 0, 0)
MARCH = datetime(2024, 3, 15, 10, 0, 0)
FEBRUARY_FIRST = datetime(2024, 2, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# 1. Particionamento pela data de corte
# ---------------------------------------------------------------------------



CUTOFF = datetime(2024, 3, 1)
MESES = (
    datetime(2024, 1, 15),
    datetime(2024, 2, 20),
    datetime(2024, 3, 1),
    datetime(2024, 3, 10),
)


class TestTemporalSplit:
    def test_it_splits_by_date_never_at_random(self):
        """Amostragem aleatória em série temporal vaza o futuro para dentro do treino."""
        treino, validacao = split_by_cutoff(make_df(*MESES), CUTOFF)
        assert (treino["pickup_datetime"] < CUTOFF).all()
        assert (validacao["pickup_datetime"] >= CUTOFF).all()

    def test_the_two_partitions_cover_the_whole_frame(self):
        treino, validacao = split_by_cutoff(make_df(*MESES), CUTOFF)
        assert len(treino) + len(validacao) == len(MESES)

    def test_the_cutoff_instant_belongs_to_the_validation_side(self):
        """O limite precisa ser testado dos dois lados: nenhuma corrida pode ficar de fora."""
        _, validacao = split_by_cutoff(make_df(*MESES), CUTOFF)
        assert (validacao["pickup_datetime"] == CUTOFF).any()

    def test_an_empty_partition_is_refused(self):
        """Sem validação não há corte temporal, e a métrica voltaria a ser in-sample."""
        with pytest.raises(InvalidSplitError):
            split_by_cutoff(make_df(datetime(2024, 1, 1)), CUTOFF)
