from datetime import datetime, timedelta

import pandas as pd

from src.pipeline.cleaner import clean_trip_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_valid_row(**overrides) -> dict:
    """Retorna um dict com uma corrida válida. Campos podem ser sobrescritos."""
    base = {
        "pickup_datetime": datetime(2024, 6, 1, 10, 0, 0),
        "dropoff_datetime": datetime(2024, 6, 1, 10, 30, 0),  # 30 min de duração
        "fare_amount": 15.0,
        "total_amount": 21.5,
        "trip_distance": 3.5,
        "passenger_count": 2,
        "RatecodeID": 1,
    }
    base.update(overrides)
    return base


def make_df(*rows: dict) -> pd.DataFrame:
    """Cria um DataFrame a partir de uma lista de dicts de corridas."""
    return pd.DataFrame(list(rows))


class TestOutlierFilters:
    def test_a_valid_trip_survives(self):
        assert len(clean_trip_data(make_df(make_valid_row()))) == 1

    def test_fare_outside_the_sanity_range_is_dropped(self):
        """Negativo é estorno; acima do teto é erro de registro."""
        sujo = make_df(
            make_valid_row(),
            make_valid_row(fare_amount=-5.0),
            make_valid_row(fare_amount=600.0),
        )
        assert len(clean_trip_data(sujo)) == 1

    def test_the_fare_limits_hold_on_both_sides(self):
        limite = make_df(make_valid_row(fare_amount=500.0), make_valid_row(fare_amount=500.01))
        assert len(clean_trip_data(limite)) == 1

    def test_distance_outside_the_range_is_dropped(self):
        sujo = make_df(
            make_valid_row(),
            make_valid_row(trip_distance=0.0),
            make_valid_row(trip_distance=250.0),
        )
        assert len(clean_trip_data(sujo)) == 1

    def test_passenger_count_outside_the_tlc_range_is_dropped(self):
        sujo = make_df(
            make_valid_row(),
            make_valid_row(passenger_count=0),
            make_valid_row(passenger_count=9),
        )
        assert len(clean_trip_data(sujo)) == 1

    def test_a_dropoff_before_the_pickup_is_dropped(self):
        invertida = make_valid_row(
            pickup_datetime=datetime(2024, 6, 1, 10, 30),
            dropoff_datetime=datetime(2024, 6, 1, 10, 0),
        )
        assert clean_trip_data(make_df(make_valid_row(), invertida)).shape[0] == 1

    def test_a_trip_longer_than_three_hours_is_dropped(self):
        longa = make_valid_row(
            dropoff_datetime=datetime(2024, 6, 1, 10, 0) + timedelta(hours=4),
        )
        assert len(clean_trip_data(make_df(make_valid_row(), longa))) == 1
