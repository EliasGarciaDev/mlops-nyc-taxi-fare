#!/usr/bin/env python3
"""Mede e valida hipóteses de modelagem do projeto contra os dados reais da TLC.

Produz os números de `documentacao`. Cada seção responde a uma pergunta
que estava em aberto no backlog e que só podia ser respondida medindo:

- quanto custa, em dólares, a duração estimada por velocidade constante no frontend ;
- quanto se ganha tratando `RatecodeID` como categórico em vez de contínuo ;
- quanto o passageiro deixa de ver ao prever `fare_amount` em vez do valor pago .

O script reaproveita extração, limpeza e engenharia de features do próprio pipeline, de modo
que o que ele mede é o que o treino produz - e não uma segunda implementação que poderia
divergir silenciosamente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import DATA_DIR
from src.core.constants import FEATURE_ORDER, PICKUP_DATETIME_COLUMN, TARGET_COLUMN
from src.ml.trainer import ModelMetrics
from src.pipeline.cleaner import clean_trip_data
from src.pipeline.extractor import extract_trip_data
from src.pipeline.feature_engineer import engineer_features

# Constante que o frontend usa para derivar a duração a partir da distância, em mi/min.
# Mantida sincronizada com AVG_SPEED em src/web/app.js.
FRONTEND_AVG_SPEED = 0.28

DURATION_FEATURE = "trip_duration_minutes"
RATECODE_FEATURE = "RatecodeID"
TOTAL_COLUMN = "total_amount"
TIP_COLUMN = "tip_amount"

# Códigos tarifários da TLC exceto o padrão, que fica como categoria de referência do one-hot.
NON_STANDARD_RATECODES = (2, 3, 4, 5, 6)

REPORTED_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
MINUTES_PER_HOUR = 60

TRAIN_MONTHS = ((2024, 1), (2024, 2))
VALIDATION_MONTH = (2024, 3)
TAXI_TYPE = "yellow"

CACHE_DIR = DATA_DIR / "measurement-cache"


def load_month(year: int, month: int, taxi_type: str) -> pd.DataFrame:
    """Extract, clean and engineer one month, caching the result between runs.

    O cache existe porque cada mês de Yellow Taxi custa dezenas de megabytes de download e
    este script é rodado várias vezes enquanto as decisões estão sendo avaliadas.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{taxi_type}-{year}-{month:02d}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    featured = engineer_features(clean_trip_data(extract_trip_data(year, month, taxi_type)))

    columns = [PICKUP_DATETIME_COLUMN, TARGET_COLUMN, *FEATURE_ORDER]
    columns += [name for name in (TOTAL_COLUMN, TIP_COLUMN) if name in featured.columns]

    result = featured[columns].copy()
    result.to_parquet(cached, index=False)
    return result


def measure(observed: pd.Series, predicted: np.ndarray) -> ModelMetrics:
    """Score a prediction with the same metrics the training pipeline reports."""
    return ModelMetrics(
        rmse=float(mean_squared_error(observed, predicted) ** 0.5),
        mae=float(mean_absolute_error(observed, predicted)),
        r2=float(r2_score(observed, predicted)),
        sample_count=len(observed),
    )


def show(label: str, metrics: ModelMetrics) -> None:
    print(f"  {label:<46} RMSE {metrics.rmse:7.4f} | MAE {metrics.mae:7.4f} | R2 {metrics.r2:.4f}")


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def encode_ratecode(features: pd.DataFrame) -> pd.DataFrame:
    """Replace the raw rate code by one indicator per non-standard code.

    O código padrão fica de fora por ser a categoria de referência: incluí-lo tornaria a
    matriz singular, já que as colunas somariam exatamente um em toda linha.
    """
    encoded = features.drop(columns=[RATECODE_FEATURE])
    for code in NON_STANDARD_RATECODES:
        encoded[f"rate_{code}"] = (features[RATECODE_FEATURE] == code).astype(int)
    return encoded


def report_duration(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    """Quantify the cost of deriving duration from distance, as the frontend does.

    Devolve a predição de produção para que a seção final possa compará-la sem refazer o ajuste.
    """
    heading("DECISÃO 2 - trip_duration_minutes")

    speed = train["trip_distance"] / train[DURATION_FEATURE].replace(0, np.nan)
    speed = speed.replace([np.inf, -np.inf], np.nan).dropna()

    print("\nVelocidade real no treino (mi/min):")
    for quantile in REPORTED_QUANTILES:
        value = speed.quantile(quantile)
        print(f"  p{int(quantile * 100):<3} {value:.4f}   ({value * MINUTES_PER_HOUR:5.1f} mph)")
    print(f"  média {speed.mean():.4f}   ({speed.mean() * MINUTES_PER_HOUR:5.1f} mph)")
    print(
        f"  FRONTEND usa constante  {FRONTEND_AVG_SPEED:.4f}   "
        f"({FRONTEND_AVG_SPEED * MINUTES_PER_HOUR:5.1f} mph)"
    )

    correlation = train[["trip_distance", DURATION_FEATURE]].corr().to_numpy()[0][1]
    print(f"\nCorrelação real distance x duration no treino: {correlation:.4f}")
    print("Em produção o frontend força essa correlação para 1.0000")

    X_train, y_train = train[FEATURE_ORDER], train[TARGET_COLUMN]
    X_valid, y_valid = valid[FEATURE_ORDER], valid[TARGET_COLUMN]

    model = LinearRegression().fit(X_train, y_train)
    print("\nModelo atual (8 features), avaliado na validação:")
    show("duração REAL (como o modelo foi validado)", measure(y_valid, model.predict(X_valid)))

    production_features = X_valid.copy()
    production_features[DURATION_FEATURE] = production_features["trip_distance"] / FRONTEND_AVG_SPEED
    production_prediction = model.predict(production_features)
    show("duração ESTIMADA dist/0.28 (o que o site faz)", measure(y_valid, production_prediction))

    bias = float(np.mean(production_prediction - y_valid))
    print(f"\n  >>> Viés médio em produção: {bias:+.4f} USD por corrida")

    without = [name for name in FEATURE_ORDER if name != DURATION_FEATURE]
    dropped = LinearRegression().fit(X_train[without], y_train)
    show("SEM trip_duration_minutes (7 features)", measure(y_valid, dropped.predict(X_valid[without])))

    return production_prediction


def report_target(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    """Quantify the gap between the metered fare and what the passenger actually pays."""
    heading("DECISÃO 1 - alvo: fare_amount vs total_amount")

    if TOTAL_COLUMN not in valid.columns:
        print(f"  {TOTAL_COLUMN} ausente no dataset - medição impossível")
        return

    gap = valid[TOTAL_COLUMN] - valid[TARGET_COLUMN]
    print(f"\nDiferença {TOTAL_COLUMN} - {TARGET_COLUMN} na validação (USD):")
    for quantile in REPORTED_QUANTILES:
        print(f"  p{int(quantile * 100):<3} {gap.quantile(quantile):7.2f}")
    print(f"  média {gap.mean():7.2f}")

    ratio = float((valid[TOTAL_COLUMN] / valid[TARGET_COLUMN]).median())
    print(f"  razão mediana total/fare: {ratio:.4f}  (+{(ratio - 1) * 100:.1f}%)")
    print(f"\n  >>> O passageiro vê, em média, {gap.mean():.2f} USD A MENOS do que vai pagar")

    X_train, X_valid = train[FEATURE_ORDER], valid[FEATURE_ORDER]

    print("\nMesmas 8 features, alvo trocado:")
    fare = LinearRegression().fit(X_train, train[TARGET_COLUMN])
    show("alvo fare_amount", measure(valid[TARGET_COLUMN], fare.predict(X_valid)))

    total = LinearRegression().fit(X_train, train[TOTAL_COLUMN])
    show("alvo total_amount", measure(valid[TOTAL_COLUMN], total.predict(X_valid)))

    if TIP_COLUMN in valid.columns:
        # A gorjeta é comportamental e opcional: não faz parte do que a TLC cobra, então
        # removê-la isola o que o sistema poderia prometer prever de fato.
        train_target = train[TOTAL_COLUMN] - train[TIP_COLUMN]
        valid_target = valid[TOTAL_COLUMN] - valid[TIP_COLUMN]
        without_tip = LinearRegression().fit(X_train, train_target)
        show("alvo total_amount SEM gorjeta", measure(valid_target, without_tip.predict(X_valid)))


def report_ratecode(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    """Quantify the cost of treating a categorical rate code as a continuous quantity."""
    heading("- RatecodeID tratado como número contínuo")

    X_train, y_train = train[FEATURE_ORDER], train[TARGET_COLUMN]
    X_valid, y_valid = valid[FEATURE_ORDER], valid[TARGET_COLUMN]

    numeric = LinearRegression().fit(X_train, y_train)
    one_hot = LinearRegression().fit(encode_ratecode(X_train), y_train)

    print("\nValidação, alvo fare_amount, duração real:")
    show("RatecodeID numérico (atual)", measure(y_valid, numeric.predict(X_valid)))
    show("RatecodeID one-hot", measure(y_valid, one_hot.predict(encode_ratecode(X_valid))))

    print("\nDistribuição de RatecodeID no treino:")
    for code, count in train[RATECODE_FEATURE].value_counts().sort_index().items():
        share = 100 * count / len(train)
        mean_fare = train.loc[train[RATECODE_FEATURE] == code, TARGET_COLUMN].mean()
        print(f"  código {int(code)}: {count:>9,} ({share:5.2f}%)  tarifa média {mean_fare:7.2f}")


def report_combined(
    train: pd.DataFrame, valid: pd.DataFrame, production_prediction: np.ndarray
) -> None:
    """Score both corrections together, under the conditions the site actually runs in."""
    heading("COMBINADO - one-hot + sem duração, como o site realmente opera")

    columns = [name for name in FEATURE_ORDER if name != DURATION_FEATURE]
    y_valid = valid[TARGET_COLUMN]

    combined = LinearRegression().fit(encode_ratecode(train[columns]), train[TARGET_COLUMN])
    prediction = combined.predict(encode_ratecode(valid[columns]))

    print()
    show("atual, em produção (dist/0.28)", measure(y_valid, production_prediction))
    show("one-hot + sem duração (imune ao viés)", measure(y_valid, prediction))


def main() -> None:
    print("Carregando meses de treino e validação...", flush=True)
    train = pd.concat(
        [load_month(year, month, TAXI_TYPE) for year, month in TRAIN_MONTHS], ignore_index=True
    )
    valid = load_month(*VALIDATION_MONTH, TAXI_TYPE)
    print(f"  treino {len(train):,} | validação {len(valid):,}")

    production_prediction = report_duration(train, valid)
    report_target(train, valid)
    report_ratecode(train, valid)
    report_combined(train, valid, production_prediction)


if __name__ == "__main__":
    main()
