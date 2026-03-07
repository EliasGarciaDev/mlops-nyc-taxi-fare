import pandas as pd

from src.core.constants import (
    AIRPORT_LOCATION_IDS,
    CRZ_LOCATION_IDS,
    RATECODE_FEATURES,
    WEEKEND_START_WEEKDAY,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal and geospatial features from cleaned taxi trip records."""
    if df.empty:
        return df.copy()

    result = df.copy()

    result["hour_of_day"] = result["pickup_datetime"].dt.hour
    result["day_of_week"] = result["pickup_datetime"].dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= WEEKEND_START_WEEKDAY).astype(int)

    duration = result["dropoff_datetime"] - result["pickup_datetime"]
    result["trip_duration_minutes"] = duration.dt.total_seconds() / 60.0

    is_airport = (
        result["PULocationID"].isin(AIRPORT_LOCATION_IDS)
        | result["DOLocationID"].isin(AIRPORT_LOCATION_IDS)
    )
    result["is_airport_trip"] = is_airport.astype(int)

    for code, feature_name in RATECODE_FEATURES.items():
        result[feature_name] = (result["RatecodeID"] == code).astype(int)

    # Derivada das pontas, nunca da taxa cobrada: a inferência só conhece embarque e
    in_crz = (
        result["PULocationID"].isin(CRZ_LOCATION_IDS)
        | result["DOLocationID"].isin(CRZ_LOCATION_IDS)
    )
    result["is_congestion_zone"] = in_crz.astype(int)

    return result
