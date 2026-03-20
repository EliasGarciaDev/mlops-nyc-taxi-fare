from datetime import datetime

import pandas as pd

from src.core.constants import CRZ_LOCATION_IDS, FEATURE_ORDER
from src.pipeline.feature_engineer import engineer_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(**overrides) -> pd.DataFrame:
    """Retorna um DataFrame de uma linha com valores válidos. Campos sobrescrevíveis."""
    base = {
        "pickup_datetime": datetime(2024, 6, 3, 14, 30, 0),   # segunda-feira, 14h30
        "dropoff_datetime": datetime(2024, 6, 3, 14, 50, 0),  # 20 minutos depois
        "fare_amount": 15.0,
        "trip_distance": 3.5,
        "passenger_count": 2,
        "PULocationID": 100,
        "DOLocationID": 200,
        "RatecodeID": 1,
        "total_amount": 18.5,
        "payment_type": 1,
        "taxi_type": "yellow",
    }
    base.update(overrides)
    return pd.DataFrame([base])



class TestTemporalFeatures:
    def test_hour_day_and_weekend_come_from_the_pickup(self):
        linha = engineer_features(make_df()).iloc[0]
        assert linha["hour_of_day"] == 14
        assert linha["day_of_week"] == 0  # segunda
        assert linha["is_weekend"] == 0

    def test_the_weekend_boundary_holds_on_both_sides(self):
        sexta = engineer_features(make_df(pickup_datetime=datetime(2024, 6, 7, 10, 0)))
        sabado = engineer_features(make_df(pickup_datetime=datetime(2024, 6, 8, 10, 0)))
        assert sexta.iloc[0]["is_weekend"] == 0
        assert sabado.iloc[0]["is_weekend"] == 1

    def test_duration_is_the_gap_between_pickup_and_dropoff(self):
        df = make_df(
            pickup_datetime=datetime(2024, 6, 3, 8, 0),
            dropoff_datetime=datetime(2024, 6, 3, 9, 30),
        )
        assert engineer_features(df).iloc[0]["trip_duration_minutes"] == 90.0


class TestGeographicFeatures:
    def test_any_end_at_an_airport_marks_the_trip(self):
        """A regra olha as duas pontas: antes do C-06 a feature era sempre 1."""
        embarque = engineer_features(make_df(PULocationID=132, DOLocationID=200))
        desembarque = engineer_features(make_df(PULocationID=100, DOLocationID=138))
        assert embarque.iloc[0]["is_airport_trip"] == 1
        assert desembarque.iloc[0]["is_airport_trip"] == 1

    def test_a_trip_between_ordinary_zones_is_not_an_airport_trip(self):
        assert engineer_features(make_df(PULocationID=7, DOLocationID=33)).iloc[0][
            "is_airport_trip"
        ] == 0

    def test_the_congestion_flag_follows_the_crz_zones(self):
        dentro = sorted(CRZ_LOCATION_IDS)[0]
        assert engineer_features(make_df(PULocationID=dentro)).iloc[0]["is_congestion_zone"] == 1
        # 7 (Astoria) e 33 (Brooklyn Heights) ficam fora da zona de congestionamento.
        fora = engineer_features(make_df(PULocationID=7, DOLocationID=33))
        assert fora.iloc[0]["is_congestion_zone"] == 0


class TestRatecodeIndicators:
    def test_the_flat_fare_code_lights_its_own_indicator(self):
        """Regime tarifário é categoria, não número."""
        linha = engineer_features(make_df(RatecodeID=2)).iloc[0]
        assert linha["is_rate_jfk"] == 1
        assert linha["is_rate_newark"] == 0

    def test_the_standard_code_lights_nothing(self):
        """O código 1 é a categoria de referência do one-hot."""
        linha = engineer_features(make_df(RatecodeID=1)).iloc[0]
        assert all(linha[nome] == 0 for nome in FEATURE_ORDER if nome.startswith("is_rate_"))


class TestContract:
    def test_every_model_feature_is_produced(self):
        assert set(FEATURE_ORDER) <= set(engineer_features(make_df()).columns)

    def test_an_empty_frame_comes_back_empty_instead_of_raising(self):
        """DataFrame vazio é caso esperado no fluxo mensal, não falha."""
        assert engineer_features(pd.DataFrame()).empty

    def test_the_input_frame_is_not_modified(self):
        original = make_df()
        engineer_features(original)
        assert "hour_of_day" not in original.columns
