import pandas as pd

from src.core.constants import (
    AIRPORT_LOCATION_IDS,
    CRZ_LOCATION_IDS,
    RATECODE_FEATURES,
    TARGET_COLUMN,
    TIP_AMOUNT_COLUMN,
    TOTAL_AMOUNT_COLUMN,
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
    # desembarque, e usar cbd_congestion_fee no treino criaria uma feature que produção
    # não sabe reproduzir. O conjunto de zonas vale para 2024 e 2025 igualmente, o que
    # muda em janeiro de 2025 é o preço, e essa mudança é o caso de estudo de drift.
    in_crz = (
        result["PULocationID"].isin(CRZ_LOCATION_IDS)
        | result["DOLocationID"].isin(CRZ_LOCATION_IDS)
    )
    result["is_congestion_zone"] = in_crz.astype(int)

    # O alvo: total pago menos a gorjeta, ou seja, tudo que é obrigatório. A subtração vale
    # nos dois meios de pagamento, no cartão a gorjeta está nas duas colunas e se cancela; no
    # dinheiro ela não entra em nenhuma, e `tip_amount` vem zerado.
    result[TARGET_COLUMN] = result[TOTAL_AMOUNT_COLUMN] - result[TIP_AMOUNT_COLUMN].fillna(0.0)

    return result
