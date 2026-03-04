from typing import Literal

TaxiType = Literal["yellow", "green"]

VALID_TAXI_TYPES: tuple[str, str] = ("yellow", "green")

NYC_TLC_BASE_URL: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/"

# Fuso horário oficial de Nova York (timestamps da TLC são em horário local)
NYC_TIMEZONE: str = "America/New_York"

AIRPORT_LOCATION_IDS: set[int] = {1, 132, 138}

JFK_LOCATION_ID: int = 132

# Zonas oficiais de Manhattan (utilizadas na verificação da tarifa fixa do JFK e restrições da frota Green)
MANHATTAN_LOCATION_IDS: set[int] = {
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 103, 107, 113,
    114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158,
    161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233,
    234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263,
}

# Tarifa fixa regulamentada pela TLC para corridas entre JFK e Manhattan
JFK_FLAT_FARE_AMOUNT: float = 70.00

# Mapeamento de LocationIDs por Borough para cálculo de métricas regionais
LOCATION_IDS_BY_BOROUGH: dict[str, set[int]] = {
    "Manhattan": {
        4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 103,
        107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148,
        151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211,
        224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249,
        261, 262, 263,
    },
    "Brooklyn": {
        11, 14, 17, 21, 22, 25, 26, 29, 33, 34, 35, 36, 37, 39, 40, 49, 52, 54, 55,
        61, 62, 63, 65, 66, 67, 71, 72, 76, 77, 80, 85, 89, 91, 97, 106, 108, 111,
        112, 123, 133, 149, 150, 154, 155, 165, 177, 178, 181, 188, 189, 190, 195,
        210, 217, 222, 225, 227, 228, 255, 256, 257,
    },
    "Queens": {
        2, 7, 8, 9, 10, 15, 16, 19, 27, 28, 30, 38, 53, 56, 64, 70, 73, 82, 83, 86,
        92, 93, 95, 96, 98, 101, 102, 117, 121, 122, 124, 129, 130, 131, 132, 134,
        135, 138, 139, 145, 146, 157, 160, 171, 173, 175, 179, 180, 191, 192, 193,
        196, 197, 198, 201, 203, 205, 207, 215, 216, 218, 219, 223, 226, 252, 253,
        258, 260,
    },
    "Bronx": {
        3, 18, 20, 31, 32, 46, 47, 51, 58, 59, 60, 69, 78, 81, 94, 119, 126, 136,
        147, 159, 167, 168, 169, 174, 182, 183, 184, 185, 199, 200, 208, 212, 213,
        220, 235, 240, 241, 242, 247, 248, 250, 254, 259,
    },
    "Staten Island": {
        5, 6, 23, 44, 84, 99, 109, 110, 115, 118, 156, 172, 176, 187, 204, 206, 214,
        221, 245, 251,
    },
    "EWR": {
        1,
    },
}

# Zonas por borough, do mesmo índice oficial que gera MANHATTAN_LOCATION_IDS. Existem para o
OUTER_BOROUGH_LOCATION_IDS: dict[str, set[int]] = {
    "Bronx": {
        3, 18, 20, 31, 32, 46, 47, 51, 58, 59, 60, 69, 78, 81, 94, 119, 126, 136,
        147, 159, 167, 168, 169, 174, 182, 183, 184, 185, 199, 200, 208, 212, 213,
        220, 235, 240, 241, 242, 247, 248, 250, 254, 259,
    },
    "Brooklyn": {
        11, 14, 17, 21, 22, 25, 26, 29, 33, 34, 35, 36, 37, 39, 40, 49, 52, 54, 55,
        61, 62, 63, 65, 66, 67, 71, 72, 76, 77, 80, 85, 89, 91, 97, 106, 108, 111,
        112, 123, 133, 149, 150, 154, 155, 165, 177, 178, 181, 188, 189, 190, 195,
        210, 217, 222, 225, 227, 228, 255, 256, 257,
    },
    "Queens": {
        2, 7, 8, 9, 10, 15, 16, 19, 27, 28, 30, 38, 53, 56, 64, 70, 73, 82, 83, 86,
        92, 93, 95, 96, 98, 101, 102, 117, 121, 122, 124, 129, 130, 131, 132, 134,
        135, 138, 139, 145, 146, 157, 160, 171, 173, 175, 179, 180, 191, 192, 193,
        196, 197, 198, 201, 203, 205, 207, 215, 216, 218, 219, 223, 226, 252, 253,
        258, 260,
    },
    "Staten Island": {
        5, 6, 23, 44, 84, 99, 109, 110, 115, 118, 156, 172, 176, 187, 204, 206, 214,
        221, 245, 251,
    },
}

# Zonas da Congestion Relief Zone - Manhattan ao sul da 60th St, onde a cobrança de
CRZ_LOCATION_IDS: set[int] = {
    4, 13, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 144, 148,
    158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234,
    246, 249, 261,
}

# RatecodeID identifica o regime tarifário; não é uma quantidade. Medido em 2024-01/02, a
RATECODE_FEATURES: dict[int, str] = {
    2: "is_rate_jfk",
    3: "is_rate_newark",
    4: "is_rate_nassau_westchester",
    5: "is_rate_negotiated",
}

# `trip_duration_minutes` saiu do contrato. Ela é a melhor feature quando a duração real é
FEATURE_ORDER: list[str] = [
    "trip_distance",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_airport_trip",
    "is_congestion_zone",
    *RATECODE_FEATURES.values(),
]

# O alvo é o que o passageiro entrega ao motorista, não a componente de taxímetro. Prever
TARGET_COLUMN: str = "total_amount"
PICKUP_DATETIME_COLUMN: str = "pickup_datetime"

DATETIME_COLUMNS_MAP: dict[str, dict[str, str]] = {
    "yellow": {
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
    },
    "green": {
        "lpep_pickup_datetime": "pickup_datetime",
        "lpep_dropoff_datetime": "dropoff_datetime",
    },
}

MIN_FARE_AMOUNT: float = 0.01
MAX_FARE_AMOUNT: float = 500.0

# O total pago também precisa de faixa de sanidade: negativo é estorno, e valores muito acima
MIN_TOTAL_AMOUNT: float = 0.01
MAX_TOTAL_AMOUNT: float = 1000.0

# Piso do que se pode cobrar de uma corrida: initial charge de US$ 3,00 mais as duas sobretaxas
MIN_PLAUSIBLE_TOTAL_AMOUNT: float = 4.50
MIN_TRIP_DISTANCE: float = 0.01
MAX_TRIP_DISTANCE: float = 200.0
MIN_PASSENGERS: int = 1
MAX_PASSENGERS: int = 6
MAX_DURATION_SECONDS: int = 3 * 3600
MIN_TRAINING_SAMPLES: int = 100

# R² é indefinido com menos de duas amostras. O tamanho adequado da janela de validação
# é responsabilidade do pipeline, que trabalha em meses inteiros do dataset da TLC.
MIN_EVALUATION_SAMPLES: int = 2
MIN_MONTH: int = 1
MAX_MONTH: int = 12

# Primeiro mês que este projeto suporta. Anterior a isso o esquema publicado pela TLC
FIRST_SUPPORTED_MONTH: tuple[int, int] = (2024, 1)

# A TLC publica cada mês com atraso. O limite superior acompanha o calendário em vez de
# ser fixado, senão o pipeline de retreino para de encontrar dados novos sozinho.
PUBLICATION_LAG_MONTHS: int = 2

# Segunda-feira é 0 em datetime.weekday(); sábado e domingo são 5 e 6.
WEEKEND_START_WEEKDAY: int = 5
