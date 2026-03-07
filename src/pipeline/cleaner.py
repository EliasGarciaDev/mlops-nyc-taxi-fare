import pandas as pd

from src.core.constants import (
    MAX_DURATION_SECONDS,
    MAX_FARE_AMOUNT,
    MAX_PASSENGERS,
    MAX_TOTAL_AMOUNT,
    MAX_TRIP_DISTANCE,
    MIN_FARE_AMOUNT,
    MIN_PASSENGERS,
    MIN_TOTAL_AMOUNT,
    MIN_TRIP_DISTANCE,
)


def clean_trip_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out anomalous records according to NYC TLC quality standards."""
    if df.empty:
        return df.copy()

    duration = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()

    valid_mask = (
        (df["fare_amount"] >= MIN_FARE_AMOUNT)
        & (df["fare_amount"] <= MAX_FARE_AMOUNT)
        & (df["total_amount"] >= MIN_TOTAL_AMOUNT)
        & (df["total_amount"] <= MAX_TOTAL_AMOUNT)
        & (df["trip_distance"] >= MIN_TRIP_DISTANCE)
        & (df["trip_distance"] <= MAX_TRIP_DISTANCE)
        & (df["passenger_count"].between(MIN_PASSENGERS, MAX_PASSENGERS))
        & (df["dropoff_datetime"] > df["pickup_datetime"])
        & (duration <= MAX_DURATION_SECONDS)
        & (df["RatecodeID"].notna())
        & (df["RatecodeID"].between(1, 6))
    )

    return df.loc[valid_mask].reset_index(drop=True)
