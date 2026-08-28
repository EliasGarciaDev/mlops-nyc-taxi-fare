#!/usr/bin/env python3
"""Mede a distribuição geográfica do erro do modelo pelos boroughs da cidade.

Produz os números de `documentacao`. É a análise que faltava da Fase 5, e
ela só se tornou possível quando ofez `PULocationID` e `DOLocationID` sobreviverem à
projeção de `MODEL_COLUMNS` - ohavia registrado essa ausência como o bloqueio.

A pergunta não é "qual o erro do modelo", que o backtest já responde. É se esse erro **cai de
forma desigual sobre a cidade**, e em que direção: um modelo que subestima sistematicamente numa
região entrega ali a pior experiência possível, que é a de pagar mais do que foi prometido.

Três recortes, porque medem coisas diferentes:

- **erro absoluto** diz quanto dinheiro está em jogo;
- **viés** diz para que lado o erro cai, e é o que o passageiro sente como quebra de promessa;
- **erro relativo** corrige o efeito de escala - US$ 5 numa corrida de US$ 10 é outra coisa que
  US$ 5 numa de US$ 90, e é o recorte em que a periferia tende a aparecer pior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import DATA_DIR
from src.core.constants import FEATURE_ORDER, JFK_FLAT_FARE_AMOUNT, MIN_PLAUSIBLE_TOTAL_AMOUNT
from src.ml.fare_rules import _zones_confirm_flat_fare

TAXI_TYPE = "yellow"
TRAIN_MONTHS = ((2024, 1), (2024, 2))
VALIDATION_MONTH = (2024, 3)

CACHE_DIR = DATA_DIR / "flat-fare-cache"
ZONES_PATH = Path(__file__).resolve().parent.parent / "src" / "web" / "data" / "taxi_zones.json"

# Abaixo disto a estatística do recorte é ruído e reportá-la convida a conclusão errada.
MIN_SEGMENT_TRIPS = 500
WITHIN_TOLERANCE_USD = 5.0


def load_month(year: int, month: int) -> pd.DataFrame:
    cached = CACHE_DIR / f"{TAXI_TYPE}-{year}-{month:02d}.parquet"
    if not cached.exists():
        raise SystemExit(
            f"Cache ausente: {cached}. Rode scripts/measure_flat_fare.py primeiro, "
            "que é quem baixa e prepara os meses."
        )
    return pd.read_parquet(cached)


def load_borough_by_zone() -> dict[int, str]:
    payload = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
    return {int(zone["id"]): str(zone["borough"]) for zone in payload["zones"]}


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def score_segments(frame: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    """Summarize error, bias and relative error for each value of a grouping column."""
    error = frame["prediction"] - frame["total_amount"]
    working = frame.assign(
        error=error,
        absolute_error=error.abs(),
        relative_error=error.abs() / frame["total_amount"],
        within=(error.abs() <= WITHIN_TOLERANCE_USD),
    )

    grouped = working.groupby(by).agg(
        corridas=("total_amount", "size"),
        tarifa_media=("total_amount", "mean"),
        mae=("absolute_error", "mean"),
        vies=("error", "mean"),
        erro_relativo=("relative_error", "median"),
        dentro=("within", "mean"),
    )
    grouped = grouped[grouped["corridas"] >= MIN_SEGMENT_TRIPS]
    grouped["dentro"] *= 100
    grouped["erro_relativo"] *= 100

    print(f"\n{label}")
    print(
        f"{by:>18} {'corridas':>10} {'tarifa':>8} {'MAE':>7} "
        f"{'viés':>8} {'erro rel.':>10} {'±$5':>7}"
    )
    for name, row in grouped.sort_values("vies").iterrows():
        print(
            f"{name!s:>18} {int(row['corridas']):>10,} {row['tarifa_media']:>8.2f} "
            f"{row['mae']:>7.2f} {row['vies']:>+8.2f} {row['erro_relativo']:>9.1f}% "
            f"{row['dentro']:>6.1f}%"
        )
    return grouped


def report_by_borough(frame: pd.DataFrame) -> pd.DataFrame:
    heading("1. ERRO POR BOROUGH DE EMBARQUE")
    print(
        "\nViés negativo significa que o sistema promete MENOS do que o passageiro paga -\n"
        "é a direção que o usuário sente como quebra de promessa."
    )
    grouped = score_segments(frame, "pickup_borough", "Embarque:")
    score_segments(frame, "dropoff_borough", "\nDesembarque:")
    return grouped


def report_worst_zones(frame: pd.DataFrame, borough_by_zone: dict[int, str]) -> None:
    heading("2. AS ZONAS ONDE O SISTEMA MAIS SUBESTIMA")
    print(
        "\nO borough é uma média grosseira: dentro dele há zonas muito diferentes. Este recorte\n"
        "mostra se a desigualdade se concentra em bairros específicos."
    )

    error = frame["prediction"] - frame["total_amount"]
    working = frame.assign(error=error, absolute_error=error.abs())
    grouped = working.groupby("PULocationID").agg(
        corridas=("total_amount", "size"),
        tarifa_media=("total_amount", "mean"),
        vies=("error", "mean"),
        mae=("absolute_error", "mean"),
    )
    grouped = grouped[grouped["corridas"] >= MIN_SEGMENT_TRIPS]

    print(f"\n{'zona':>6} {'borough':>16} {'corridas':>10} {'tarifa':>8} {'viés':>8} {'MAE':>7}")
    for zone_id, row in grouped.sort_values("vies").head(10).iterrows():
        borough = borough_by_zone.get(int(zone_id), "?")
        print(
            f"{int(zone_id):>6} {borough:>16} {int(row['corridas']):>10,} "
            f"{row['tarifa_media']:>8.2f} {row['vies']:>+8.2f} {row['mae']:>7.2f}"
        )

    print(f"\n{'':>6} {'e as que mais superestimam':>16}")
    for zone_id, row in grouped.sort_values("vies", ascending=False).head(5).iterrows():
        borough = borough_by_zone.get(int(zone_id), "?")
        print(
            f"{int(zone_id):>6} {borough:>16} {int(row['corridas']):>10,} "
            f"{row['tarifa_media']:>8.2f} {row['vies']:>+8.2f} {row['mae']:>7.2f}"
        )


def report_fare_scale(frame: pd.DataFrame) -> None:
    """Check whether the relative error penalizes cheap trips, which are the peripheral ones."""
    heading("3. O ERRO RELATIVO CONTRA O VALOR DA CORRIDA")

    bins = [0, 10, 20, 30, 50, 1000]
    labels = ["até $10", "$10–20", "$20–30", "$30–50", "acima de $50"]
    working = frame.assign(faixa=pd.cut(frame["total_amount"], bins=bins, labels=labels))

    error = working["prediction"] - working["total_amount"]
    working = working.assign(
        absolute_error=error.abs(), relative_error=error.abs() / working["total_amount"]
    )

    grouped = working.groupby("faixa", observed=True).agg(
        corridas=("total_amount", "size"),
        mae=("absolute_error", "mean"),
        erro_relativo=("relative_error", "median"),
    )
    print(f"\n{'faixa':>14} {'corridas':>10} {'MAE':>7} {'erro relativo':>14}")
    for name, row in grouped.iterrows():
        print(
            f"{name!s:>14} {int(row['corridas']):>10,} {row['mae']:>7.2f} "
            f"{row['erro_relativo'] * 100:>13.1f}%"
        )

    print(
        "\nSe o erro relativo sobe conforme a corrida barateia, a desigualdade geográfica é em\n"
        "boa parte um efeito de escala: corridas curtas e baratas são as da periferia."
    )



def report_representativity(train: pd.DataFrame, borough_by_zone: dict[int, str]) -> None:
    """Show how much of the training window each borough accounts for."""
    heading("4. DE ONDE VÊM AS CORRIDAS QUE TREINARAM O MODELO")

    share = train["PULocationID"].map(borough_by_zone).value_counts(normalize=True) * 100
    print(f"\n{'borough de embarque':>22} {'participação no treino':>24}")
    for name, value in share.items():
        print(f"{name!s:>22} {value:>23.2f}%")

    print(
        "\nUm mínimo quadrado minimiza o erro médio, e o erro médio é dominado por quem tem\n"
        "massa. Não é viés de amostragem: o yellow taxi opera mesmo assim. É o produto que\n"
        "promete mais do que os dados sustentam, ao aceitar pedido de qualquer ponto da cidade."
    )


def report_scale_versus_geography(frame: pd.DataFrame) -> None:
    """Compare boroughs within the same fare band, to tell scale apart from geography."""
    heading("5. ESCALA OU GEOGRAFIA? O MESMO RECORTE DE PREÇO EM CADA BOROUGH")

    bins = [0, 15, 30, 60, 10_000]
    labels = ["até $15", "$15–30", "$30–60", "acima de $60"]
    error = frame["prediction"] - frame["total_amount"]
    working = frame.assign(
        faixa=pd.cut(frame["total_amount"], bins=bins, labels=labels),
        relative_error=error.abs() / frame["total_amount"],
    )

    grouped = working.groupby(["dropoff_borough", "faixa"], observed=True).agg(
        corridas=("total_amount", "size"),
        erro_relativo=("relative_error", "median"),
    )
    grouped = grouped[grouped["corridas"] >= MIN_SEGMENT_TRIPS]

    print(f"\n{'desembarque':>16} {'faixa':>14} {'corridas':>10} {'erro relativo':>14}")
    for (borough, band), row in grouped.iterrows():
        print(
            f"{borough!s:>16} {band!s:>14} {int(row['corridas']):>10,} "
            f"{row['erro_relativo'] * 100:>13.1f}%"
        )

    print(
        "\nSe a diferença entre boroughs sobrevive ao controle de faixa, há desigualdade\n"
        "geográfica além do efeito de escala - e é nas corridas baratas que ela aparece."
    )


def main() -> None:
    borough_by_zone = load_borough_by_zone()

    train = pd.concat([load_month(*month) for month in TRAIN_MONTHS], ignore_index=True)
    valid = load_month(*VALIDATION_MONTH)
    print(f"Treino: {len(train):,} corridas | Validação: {len(valid):,} corridas")

    model = LinearRegression().fit(train[FEATURE_ORDER], train["total_amount"])
    estimate = model.predict(valid[FEATURE_ORDER])

    # A camada de regras doé parte do sistema: medir sem ela mediria outro produto.
    confirmed = _zones_confirm_flat_fare(valid["PULocationID"], valid["DOLocationID"]).to_numpy()
    excess = float(
        train[_zones_confirm_flat_fare(train["PULocationID"], train["DOLocationID"])][
            "total_amount"
        ].mean()
        - JFK_FLAT_FARE_AMOUNT
    )
    served = np.maximum(estimate, MIN_PLAUSIBLE_TOTAL_AMOUNT)
    served[confirmed] = JFK_FLAT_FARE_AMOUNT + excess

    scored = valid.assign(
        prediction=served,
        pickup_borough=valid["PULocationID"].map(borough_by_zone),
        dropoff_borough=valid["DOLocationID"].map(borough_by_zone),
    ).dropna(subset=["pickup_borough", "dropoff_borough"])

    report_by_borough(scored)
    report_worst_zones(scored, borough_by_zone)
    report_fare_scale(scored)
    report_representativity(train, borough_by_zone)
    report_scale_versus_geography(scored)


if __name__ == "__main__":
    main()
