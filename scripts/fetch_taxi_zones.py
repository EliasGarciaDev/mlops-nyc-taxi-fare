#!/usr/bin/env python3
"""Baixa os contornos oficiais das zonas de táxi da NYC TLC e prepara a geometria para a web.

O contorno publicado pela TLC tem precisão cadastral e pesa dezenas de megabytes, o que é
inviável para o navegador. Este comando baixa a versão oficial, simplifica a geometria e
grava apenas os campos que a aplicação usa.

    python scripts/fetch_taxi_zones.py

O resultado é versionado no repositório: sem ele o mapa não sabe onde a cidade termina, e um
clone novo precisaria de rede para funcionar.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geo import (
    DEFAULT_SIMPLIFY_TOLERANCE,
    Ring,
    bounding_box,
    simplify_ring,
)

TAXI_ZONES_URL: Final[str] = (
    "https://data.cityofnewyork.us/api/geospatial/8meu-9t5y?method=export&format=GeoJSON"
)
OUTPUT_PATH: Final[Path] = Path("src/web/data/taxi_zones.json")
DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 120

# A TLC modela cada zona como MultiPolygon; o primeiro anel de cada polígono é o contorno
# externo e os seguintes são recortes internos.
EXTERIOR_RING_INDEX: Final[int] = 0


class ZoneBuildError(Exception):
    """Raised when the downloaded boundaries cannot be turned into a usable zone index."""


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
        return bytes(response.read())


RawPolygon = Sequence[Sequence[Sequence[float]]]


def _as_rings(polygon: RawPolygon) -> list[Ring]:
    return [[(float(point[0]), float(point[1])) for point in ring] for ring in polygon]


def _polygons_of(geometry: dict[str, Any]) -> list[RawPolygon]:
    kind = geometry.get("type")
    if kind == "MultiPolygon":
        return list(geometry["coordinates"])
    if kind == "Polygon":
        return [geometry["coordinates"]]
    raise ZoneBuildError(f"Geometria não suportada: {kind!r}.")


def build_zone(feature: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Reduce one GeoJSON feature to the fields and precision the client needs."""
    properties = feature.get("properties") or {}
    try:
        location_id = int(properties["locationid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ZoneBuildError(f"Zona sem locationid utilizável: {properties!r}") from exc

    polygons = [
        [simplify_ring(ring, tolerance) for ring in _as_rings(polygon)]
        for polygon in _polygons_of(feature["geometry"])
    ]
    every_ring = [ring for polygon in polygons for ring in polygon]

    return {
        "id": location_id,
        "zone": str(properties.get("zone", "")),
        "borough": str(properties.get("borough", "")),
        "bbox": [round(value, 6) for value in bounding_box(every_ring)],
        "polygons": [
            [[[round(x, 6), round(y, 6)] for x, y in ring] for ring in polygon]
            for polygon in polygons
        ],
    }


def merge_by_location_id(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine features that share a LocationID into a single, multi-part zone.

    A TLC publica uma feature por polígono, então zonas fisicamente descontínuas aparecem
    repetidas - Corona é partida em dois trechos e a zona das ilhas do porto tem uma feature
    por ilha. Sem fundir, o índice teria entradas redundantes com o mesmo identificador.
    """
    merged: dict[int, dict[str, Any]] = {}
    for zone in zones:
        existing = merged.get(zone["id"])
        if existing is None:
            merged[zone["id"]] = zone
            continue
        existing["polygons"].extend(zone["polygons"])
        existing["bbox"] = [
            round(value, 6)
            for value in bounding_box(
                [ring for polygon in existing["polygons"] for ring in polygon]
            )
        ]
    return sorted(merged.values(), key=lambda zone: zone["id"])


def build_index(collection: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Turn the official FeatureCollection into the compact index served to the browser."""
    features = collection.get("features")
    if not features:
        raise ZoneBuildError("A coleção baixada não contém nenhuma zona.")

    zones = merge_by_location_id([build_zone(feature, tolerance) for feature in features])
    every_ring = [ring for zone in zones for polygon in zone["polygons"] for ring in polygon]

    return {
        "source": TAXI_ZONES_URL,
        "tolerance": tolerance,
        "bbox": [round(value, 6) for value in bounding_box(every_ring)],
        "zones": zones,
    }


def count_vertices(index: dict[str, Any]) -> int:
    return sum(
        len(ring) for zone in index["zones"] for polygon in zone["polygons"] for ring in polygon
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=TAXI_ZONES_URL)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_SIMPLIFY_TOLERANCE)
    args = parser.parse_args(argv)

    print(f"Baixando {args.url}")
    try:
        raw = download(args.url)
    except OSError as failure:
        sys.stderr.write(f"Falha ao baixar os contornos: {failure}\n")
        return 1

    try:
        index = build_index(json.loads(raw), args.tolerance)
    except (ZoneBuildError, ValueError) as failure:
        sys.stderr.write(f"Falha ao preparar os contornos: {failure}\n")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")

    size_kb = args.output.stat().st_size / 1024
    print(
        f"{len(index['zones'])} zonas, {count_vertices(index)} vértices, "
        f"{size_kb:.0f} KB em {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
