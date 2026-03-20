import math

import pandas as pd
import pytest

from src.core.constants import FEATURE_ORDER
from src.core.exceptions import InsufficientDataError, InvalidTaxiTypeError
from src.ml.trainer import evaluate_model, fit_model, train_and_evaluate

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Cópia literal independente da implementação: se o trainer mudar a lista,
# os testes parametrizados continuam cobrando o contrato original.
REQUIRED_FEATURES = [
    "total_amount",
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
]

FEATURE_COLS = [column for column in REQUIRED_FEATURES if column != "total_amount"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_training_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Gera um DataFrame sintético com relação linear real entre features e target.

    fare_amount ≈ 2.5 * trip_distance + 0.3 * trip_duration_minutes + ruído

    Aritmética determinística baseada no índice: reprodutível sem depender de numpy.
    O seed desloca as séries para produzir uma partição distinta da de treino.
    """
    offset = seed * 7
    trip_distance = pd.Series([((i + offset) % 20) * 0.5 + 0.5 for i in range(n)])
    trip_duration = pd.Series([((i + offset) % 30) * 1.0 + 5.0 for i in range(n)])
    noise = pd.Series([((i + offset) % 7 - 3) * 0.5 for i in range(n)])

    return pd.DataFrame(
        {
            "total_amount": 2.5 * trip_distance + 0.3 * trip_duration + noise,
            "trip_distance": trip_distance,
            "trip_duration_minutes": trip_duration,
            "hour_of_day": pd.Series([(i + offset) % 24 for i in range(n)]),
            "day_of_week": pd.Series([(i + offset) % 7 for i in range(n)]),
            "is_weekend": pd.Series([1 if (i + offset) % 7 >= 5 else 0 for i in range(n)]),
            "is_airport_trip": pd.Series([1 if (i + offset) % 10 == 0 else 0 for i in range(n)]),
            "is_congestion_zone": pd.Series([1 if (i + offset) % 5 == 0 else 0 for i in range(n)]),
            # Um regime tarifário por linha, em rodízio: os indicadores são mutuamente
            # exclusivos, como o engineer_features os produz. O resto das linhas cai na
            # categoria de referência, com todos zerados.
            "is_rate_jfk": pd.Series([1 if (i + offset) % 9 == 1 else 0 for i in range(n)]),
            "is_rate_newark": pd.Series([1 if (i + offset) % 9 == 3 else 0 for i in range(n)]),
            "is_rate_nassau_westchester": pd.Series(
                [1 if (i + offset) % 9 == 5 else 0 for i in range(n)]
            ),
            "is_rate_negotiated": pd.Series([1 if (i + offset) % 9 == 7 else 0 for i in range(n)]),
        }
    )


# ---------------------------------------------------------------------------
# 1. fit_model devolve um estimador utilizável
# ---------------------------------------------------------------------------


class TestFitAndEvaluate:
    def test_it_fits_and_reports_train_and_validation_separately(self):
        """Avaliar sobre o treino é o erro que faz um modelo ruim parecer bom."""
        resultado = train_and_evaluate(make_training_df(), make_training_df(seed=1), "yellow")
        assert resultado.train_metrics.sample_count > 0
        assert resultado.validation_metrics.sample_count > 0
        assert resultado.taxi_type == "yellow"

    def test_the_metrics_are_finite_and_consistent(self):
        df = make_training_df()
        metricas = evaluate_model(fit_model(df, taxi_type="yellow"), df)
        assert math.isfinite(metricas.rmse)
        assert math.isfinite(metricas.r2)
        assert metricas.rmse >= 0
        assert metricas.mae >= 0

    def test_the_model_uses_only_the_declared_features(self):
        """Contrato de features é o que impede treino e produção divergirem."""
        modelo = fit_model(make_training_df(), taxi_type="yellow")
        assert list(modelo.feature_names_in_) == list(FEATURE_ORDER)

    def test_an_unknown_fleet_is_rejected(self):
        with pytest.raises(InvalidTaxiTypeError, match="taxi_type"):
            fit_model(make_training_df(), taxi_type="fhv")

    def test_a_window_too_small_to_train_is_refused(self):
        with pytest.raises(InsufficientDataError):
            fit_model(make_training_df().head(5), taxi_type="yellow")

    def test_a_frame_missing_a_feature_is_refused(self):
        incompleto = make_training_df().drop(columns=["trip_distance"])
        with pytest.raises((InsufficientDataError, KeyError, ValueError)):
            fit_model(incompleto, taxi_type="yellow")
