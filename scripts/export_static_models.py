#!/usr/bin/env python3
"""Exporta os modelos ativos em JSON estático para execução da interface no GitHub Pages.

O GitHub Pages serve arquivo estático e nada mais: não há processo para responder `/predict`.
Como o modelo é uma regressão linear, a predição é intercepto mais soma de coeficiente vezes
feature - uma conta que o cliente já sabe fazer, porque o painel XAI a refaz para explicar cada
estimativa desde o ``.

Este script não cria uma segunda implementação do modelo: ele publica os **mesmos coeficientes**
que a API serviria em `/model-info`, lidos do mesmo artefato promovido. Quem calcula no cliente é
`buildExplanation`, a função que já era testada por reproduzir a conta do servidor.

O que o modo estático não tem é a geocodificação, que depende do proxy do backend . O
mapa continua inteiro: arrastar marcadores é o fluxo principal e não precisa de rede.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import MODELS_DIR
from src.core.constants import (
    JFK_FLAT_FARE_AMOUNT,
    MIN_PLAUSIBLE_TOTAL_AMOUNT,
    VALID_TAXI_TYPES,
)
from src.ml.protocols import LinearModel
from src.ml.registry import load_model, resolve_current_metadata

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "src" / "web" / "data" / "models.json"


def export_fleet(taxi_type: str) -> dict[str, object] | None:
    """Read the promoted artifact of one fleet into the same shape `/model-info` serves."""
    metadata = resolve_current_metadata(taxi_type, MODELS_DIR)
    if metadata is None:
        print(f"  {taxi_type}: nenhum modelo promovido, pulando.")
        return None

    model: LinearModel = load_model(metadata, MODELS_DIR)
    coefficients = {
        str(name): float(coef)
        for name, coef in zip(model.feature_names_in_, model.coef_, strict=True)
    }

    calibration = metadata.flat_fare_calibration
    print(
        f"  {taxi_type}: {metadata.model_version} | RMSE {metadata.validation_metrics.rmse:.4f}"
        f" | tarifa fixa {'calibrada' if calibration else 'sem calibração'}"
    )

    return {
        "taxi_type": taxi_type,
        "intercept": float(model.intercept_),
        "coefficients": coefficients,
        "rmse": metadata.validation_metrics.rmse,
        "training_samples": metadata.train_metrics.sample_count,
        "model_version": metadata.model_version,
        # A camada de regras dodecide antes do modelo em parte das corridas, e o
        # cliente precisa aplicar a mesma regra para não exibir número diferente do da API.
        "flat_fare_excess": calibration.mean_excess if calibration else None,
        # A margem exibida é a da região do desembarque ; sem ela a interface
        # cita a agregada, que é falsa fora de Manhattan.
        "rmse_by_borough": (
            dict(metadata.segment_errors.rmse_by_borough) if metadata.segment_errors else {}
        ),
    }


def main() -> None:
    print("Exportando modelos promovidos para o modo estático:")
    fleets = {
        taxi_type: exported
        for taxi_type in VALID_TAXI_TYPES
        if (exported := export_fleet(taxi_type)) is not None
    }

    if not fleets:
        raise SystemExit(
            "Nenhuma frota tem modelo promovido. Rode `make train` antes de exportar."
        )

    payload = {
        "generated_from": "models/<frota>/current.json",
        "flat_fare_amount": JFK_FLAT_FARE_AMOUNT,
        "minimum_total_amount": MIN_PLAUSIBLE_TOTAL_AMOUNT,
        "fleets": fleets,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nEscrito em {OUTPUT_PATH.relative_to(Path.cwd())} ({OUTPUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
