#!/usr/bin/env python3
"""Constrói a tabela de distâncias reais entre zonas para uso na interface web.

O frontend calcula haversine entre os marcadores, mas o taxímetro cobra pela distância
rodada. Em Manhattan a razão entre as duas é de 1,20; fora dela, 1,50 - e um fator único
deixa viés residual em qualquer das pontas.

Como o cliente já resolve as zonas de embarque e desembarque, dá para fazer melhor que um
fator: usar a distância que as corridas daquele par de zonas de fato percorreram. Medido
sobre 2024, a tabela cobre 99% das corridas e reduz o erro médio de US$ 6,30 para US$ 4,48
contra US$ 3,84 do teto teórico, que é conhecer a rota real.

    python scripts/build_distance_table.py --from 2024-01 --to 2024-03
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.log import configure_logging, get_logger
from src.core.months import YearMonth, month_range
from src.pipeline.cleaner import clean_trip_data
from src.pipeline.extractor import extract_trip_data

logger = get_logger(__name__)

OUTPUT_PATH: Final[Path] = Path("src/web/data/trip_distances.json")

# Abaixo disso a mediana do par é instável e o fator de área serve melhor. Trinta corridas
# num mês inteiro já indica um par que praticamente não ocorre.
MIN_PAIR_TRIPS: Final[int] = 30

# Fatores de fallback, medidos sobre 2,8 milhões de corridas de 2024-03 comparando a
# haversine entre centroides de zona com a distância registrada pelo taxímetro. Manhattan é
# mais baixo porque a malha em grade aproxima melhor a linha reta que as travessias de ponte.
MANHATTAN_DETOUR: Final[float] = 1.2040
DEFAULT_DETOUR: Final[float] = 1.4976


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_distance_table.py",
        description="Gera a tabela de distância rodada por par de zonas, usada pelo frontend.",
    )
    parser.add_argument("--from", dest="start", required=True, metavar="AAAA-MM")
    parser.add_argument("--to", dest="end", required=True, metavar="AAAA-MM")
    parser.add_argument("--taxi-type", default="yellow", choices=("yellow", "green"))
    parser.add_argument("--min-trips", type=int, default=MIN_PAIR_TRIPS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def parse_month(value: str) -> YearMonth:
    year, month = value.split("-")
    return YearMonth(int(year), int(month))


def collect_pairs(taxi_type: str, months: Sequence[YearMonth], min_trips: int) -> dict[str, float]:
    """Median metered distance of each origin/destination zone pair."""
    frames = []
    for month in months:
        cleaned = clean_trip_data(extract_trip_data(month.year, month.month, taxi_type))
        frames.append(cleaned[["PULocationID", "DOLocationID", "trip_distance"]])
        logger.info(
            "month_collected",
            extra={"taxi_type": taxi_type, "month": str(month), "rows": len(cleaned)},
        )

    trips = pd.concat(frames, ignore_index=True)
    grouped = trips.groupby(["PULocationID", "DOLocationID"])["trip_distance"].agg(
        ["median", "size"]
    )
    kept = grouped[grouped["size"] >= min_trips]

    logger.info(
        "pairs_selected",
        extra={
            "total_pairs": len(grouped),
            "kept_pairs": len(kept),
            "min_trips": min_trips,
            "coverage": float(kept["size"].sum() / grouped["size"].sum()),
        },
    )
    # A chave textual "origem-destino" mantém o JSON legível e é o que o cliente monta.
    return {
        f"{int(pu)}-{int(do)}": round(float(median), 3)
        for (pu, do), median in kept["median"].items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)

    months = month_range(parse_month(args.start), parse_month(args.end))
    pairs = collect_pairs(args.taxi_type, months, args.min_trips)

    payload = {
        "source": f"NYC TLC {args.taxi_type} trip records",
        "months": [str(month) for month in months],
        "min_trips": args.min_trips,
        "manhattan_detour": MANHATTAN_DETOUR,
        "default_detour": DEFAULT_DETOUR,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

    size_kb = args.output.stat().st_size / 1024
    print(f"{len(pairs):,} pares gravados em {args.output} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
