#!/usr/bin/env python3
"""Avalia o impacto da regra de tarifa fixa do JFK em relação ao alvo total_amount.

Produz os números de `documentacao` e do. O achado original do
foi medido contra `fare_amount`, antes dotrocar o alvo para `total_amount`: os
US$ 70,00 da tarifa fixa são componente de taxímetro, e o desvio padrão de 17 centavos que
justificava a regra é a variância daquela parcela, não a do total pago. Este script remede tudo
sobre o alvo vigente e responde às perguntas que decidem o desenho da camada:

- quanta variância sobra no segmento sob `total_amount`, e quanto dela é gorjeta;
- qual desenho de regra ganha, e quanto vale no agregado;
- sobre qual população calibrar o excedente da tarifa fixa;
- com que frequência o `RatecodeID` declarado discorda das zonas da corrida.

Reaproveita extração, limpeza e engenharia de features do próprio pipeline, de modo que o que
ele mede seja o que o treino produz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import DATA_DIR
from src.core.constants import (
    FEATURE_ORDER,
    JFK_FLAT_FARE_AMOUNT,
    JFK_LOCATION_ID,
    MANHATTAN_LOCATION_IDS,
    MIN_PLAUSIBLE_TOTAL_AMOUNT,
)
from src.pipeline.cleaner import clean_trip_data
from src.pipeline.extractor import extract_trip_data
from src.pipeline.feature_engineer import engineer_features

FLAT_FARE_RATECODE = 2
TAXI_TYPE = "yellow"
TRAIN_MONTHS = ((2024, 1), (2024, 2))
VALIDATION_MONTH = (2024, 3)

CACHE_DIR = DATA_DIR / "flat-fare-cache"
CACHED_COLUMNS = [
    *FEATURE_ORDER,
    "total_amount",
    "tip_amount",
    "tolls_amount",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
]


def load_month(year: int, month: int) -> pd.DataFrame:
    """Extract, clean and engineer one month, keeping the zone columns and caching the result."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{TAXI_TYPE}-{year}-{month:02d}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    featured = engineer_features(clean_trip_data(extract_trip_data(year, month, TAXI_TYPE)))
    result = featured[[name for name in CACHED_COLUMNS if name in featured.columns]].copy()
    result.to_parquet(cached, index=False)
    return result


def zones_confirm_flat_fare(frame: pd.DataFrame) -> pd.Series:
    """Flag the trips whose ends confirm a JFK↔Manhattan flat fare, ignoring the declared code."""
    touches_jfk = (frame["PULocationID"] == JFK_LOCATION_ID) | (
        frame["DOLocationID"] == JFK_LOCATION_ID
    )
    touches_manhattan = frame["PULocationID"].isin(MANHATTAN_LOCATION_IDS) | frame[
        "DOLocationID"
    ].isin(MANHATTAN_LOCATION_IDS)
    return touches_jfk & touches_manhattan


def rmse(observed: pd.Series, predicted: np.ndarray) -> float:
    return float(mean_squared_error(observed, predicted) ** 0.5)


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def report_variance(flat: pd.DataFrame) -> None:
    """Show what is left varying once the meter amount is fixed by regulation."""
    heading("1. O QUE SOBRA VARIANDO QUANDO O TAXÍMETRO É CONSTANTE")

    print(f"\nCorridas de RatecodeID {FLAT_FARE_RATECODE}: {len(flat):,}")
    print(f"{'componente':>34} {'média':>9} {'desvio padrão':>15}")
    for column in ("fare_amount", "total_amount", "tip_amount", "tolls_amount"):
        if column not in flat.columns:
            continue
        values = flat[column]
        print(f"{column:>34} {values.mean():>9.2f} {values.std():>15.2f}")

    without_tip = flat["total_amount"] - flat["tip_amount"]
    without_both = without_tip - flat["tolls_amount"]
    print(f"{'total sem gorjeta':>34} {without_tip.mean():>9.2f} {without_tip.std():>15.2f}")
    print(f"{'total sem gorjeta nem pedágio':>34} {without_both.mean():>9.2f} {without_both.std():>15.2f}")

    print(
        "\nA gorjeta é a maior fonte de variância que sobra: o desvio padrão do total cai de "
        f"{flat['total_amount'].std():.2f} para {without_tip.std():.2f} ao removê-la."
    )


def report_coherence(valid: pd.DataFrame) -> None:
    """Measure how often the declared rate code disagrees with the trip ends."""
    heading("2. COERÊNCIA ENTRE O RatecodeID DECLARADO E AS ZONAS DA CORRIDA")

    declared = valid["RatecodeID"] == FLAT_FARE_RATECODE
    confirmed = zones_confirm_flat_fare(valid)

    print(f"\nDeclaram o código {FLAT_FARE_RATECODE}: {declared.sum():,}")
    incoherent = (declared & ~confirmed).sum()
    print(f"  destas, as zonas NÃO confirmam: {incoherent:,} ({incoherent / declared.sum() * 100:.2f}%)")

    print(f"\nZonas confirmam JFK↔Manhattan: {confirmed.sum():,}")
    missing = (confirmed & ~declared).sum()
    share = missing / confirmed.sum() * 100
    print(f"  destas, NÃO declaram o código {FLAT_FARE_RATECODE}: {missing:,} ({share:.2f}%)")

    print(
        "\nO campo é digitado e erra nos dois sentidos. Uma regra que confie nele aceita que "
        "qualquer trajeto reivindique a tarifa do aeroporto."
    )


def report_calibration_population(train: pd.DataFrame, valid: pd.DataFrame) -> float:
    """Decide which population the flat fare excess should be calibrated on.

    Devolve o excedente escolhido para que a comparação de desenhos o reaproveite.
    """
    heading("3. SOBRE QUAL POPULAÇÃO CALIBRAR O EXCEDENTE")

    applied = valid[zones_confirm_flat_fare(valid)]
    observed = applied["total_amount"]

    candidates: dict[str, pd.DataFrame] = {
        "só as zonas confirmam": train[zones_confirm_flat_fare(train)],
        "só o código declarado": train[train["RatecodeID"] == FLAT_FARE_RATECODE],
        "zonas e código concordam": train[
            zones_confirm_flat_fare(train) & (train["RatecodeID"] == FLAT_FARE_RATECODE)
        ],
    }

    print(f"\nPopulação de aplicação (zonas confirmam), validação: {len(applied):,} corridas")
    print(f"\n{'calibrado em':>30} {'n treino':>10} {'excedente':>11} {'RMSE aplicado':>15}")

    best_excess = 0.0
    best_rmse = float("inf")
    for label, subset in candidates.items():
        excess = float(subset["total_amount"].mean() - JFK_FLAT_FARE_AMOUNT)
        scored = rmse(observed, np.full(len(observed), JFK_FLAT_FARE_AMOUNT + excess))
        print(f"{label:>30} {len(subset):>10,} {excess:>11.2f} {scored:>15.4f}")
        if scored < best_rmse:
            best_rmse, best_excess = scored, excess

    print(f"\nEscolhido: excedente de US$ {best_excess:.2f} (RMSE {best_rmse:.4f}).")
    return best_excess


def report_designs(
    train: pd.DataFrame, valid: pd.DataFrame, excess: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compare the candidate rule designs on the population the rule would apply to.

    Devolve a predição sem e com a camada, para a seção de impacto.
    """
    heading("4. DESENHOS DA CAMADA")

    model = LinearRegression().fit(train[FEATURE_ORDER], train["total_amount"])
    predicted = model.predict(valid[FEATURE_ORDER])

    confirmed = zones_confirm_flat_fare(valid).to_numpy()
    applied = valid[confirmed]
    observed = applied["total_amount"]
    baseline = predicted[confirmed]

    print(f"\nSegmento onde a regra se aplica: {len(applied):,} corridas")
    print(f"\n{'desenho':>52} {'RMSE':>9}")
    print(f"{'modelo hoje, sem camada':>52} {rmse(observed, baseline):>9.4f}")
    raw = rmse(observed, np.full(len(observed), JFK_FLAT_FARE_AMOUNT))
    print(f"{'US$ 70,00 cru (o que o backlog sugeria)':>52} {raw:>9.4f}")

    chosen = JFK_FLAT_FARE_AMOUNT + excess
    print(f"{'US$ 70,00 + excedente calibrado':>52} {rmse(observed, np.full(len(observed), chosen)):>9.4f}")

    extra_features = ["hour_of_day", "day_of_week", "is_weekend", "is_congestion_zone", "trip_distance"]
    train_applied = train[zones_confirm_flat_fare(train)]
    excess_model = LinearRegression().fit(
        train_applied[extra_features], train_applied["total_amount"] - JFK_FLAT_FARE_AMOUNT
    )
    modelled = JFK_FLAT_FARE_AMOUNT + excess_model.predict(applied[extra_features])
    print(f"{'US$ 70,00 + modelo dedicado do excedente':>52} {rmse(observed, modelled):>9.4f}")

    print(
        "\nO modelo dedicado ganha pouco e custa um segundo modelo para treinar, versionar e\n"
        "monitorar - o mesmo argumento que ousou para descartar o modelo de duração."
    )

    with_rule = predicted.copy()
    with_rule[confirmed] = chosen
    return predicted, np.maximum(with_rule, MIN_PLAUSIBLE_TOTAL_AMOUNT)


def report_impact(valid: pd.DataFrame, baseline: np.ndarray, with_rule: np.ndarray) -> None:
    """Show what the layer is worth over the whole validation month."""
    heading("5. IMPACTO AGREGADO E PISO DO")

    observed = valid["total_amount"]
    print(f"\n{'':>28} {'RMSE':>9} {'MAE':>9}")
    for label, predicted in (("sem a camada", baseline), ("com a camada", with_rule)):
        scored, absolute = rmse(observed, predicted), mean_absolute_error(observed, predicted)
        print(f"{label:>28} {scored:>9.4f} {absolute:>9.4f}")
    gain = (rmse(observed, with_rule) / rmse(observed, baseline) - 1) * 100
    print(f"{'ganho':>28} {gain:>8.2f}%")

    below = int((with_rule <= MIN_PLAUSIBLE_TOTAL_AMOUNT).sum())
    print(
        f"\nPiso do(US$ {MIN_PLAUSIBLE_TOTAL_AMOUNT:.2f}): acionado em "
        f"{below:,} de {len(valid):,} predições."
    )
    print("Segue sendo salvaguarda, não correção - mas é barata e mora nesta mesma camada.")


def main() -> None:
    train = pd.concat([load_month(*month) for month in TRAIN_MONTHS], ignore_index=True)
    valid = load_month(*VALIDATION_MONTH)
    print(f"Treino: {len(train):,} corridas | Validação: {len(valid):,} corridas")

    report_variance(valid[valid["RatecodeID"] == FLAT_FARE_RATECODE])
    report_coherence(valid)
    excess = report_calibration_population(train, valid)
    baseline, with_rule = report_designs(train, valid, excess)
    report_impact(valid, baseline, with_rule)


if __name__ == "__main__":
    main()
