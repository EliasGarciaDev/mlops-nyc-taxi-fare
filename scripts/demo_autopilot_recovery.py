#!/usr/bin/env python3
"""Simula os cenários de salvaguarda e recuperação automática do piloto com dados sintéticos.

A primeira execução real do ciclo (`documentacao`) percorreu gatilho,
treino, gate e promoção - mas não o **rollback automático** nem o **gatilho por erro**, porque a
realidade de 2024-05 não os provocou. Ficar esperando o mês em que a realidade colabore não é
verificação: é sorte.

Este script monta o cenário de propósito, e diz que é montagem.

O modo de falha reproduzido é conhecido em MLOps e não é hipotético: **um retreino que rodou
sobre uma fatia estreita dos dados**, por filtro errado ou ingestão parcial. O modelo resultante
mede excelente contra si mesmo - a validação herda o mesmo recorte - e é ruim no mundo inteiro.
O gate donão veria problema, porque ele compara challenger e campeão na mesma janela, e
essa janela também estaria estreita.

É exatamente a lacuna que odescreve ao justificar o rollback: o gate impede promover um
modelo pior, mas nada impedia um modelo **já promovido** de se revelar ruim depois.

O campeão ruim é promovido aqui **contornando o gate**, com `promote_version` direto. Isso é
deliberado: o objetivo é exercitar a rede de segurança seguinte, não fingir que o gate falhou.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.months import YearMonth
from src.ml.autopilot import AutopilotAction, run_autopilot_for_fleet
from src.ml.registry import (
    METADATA_SUFFIX,
    ModelMetadata,
    load_model,
    promote_version,
    resolve_current_metadata,
    save_model,
)
from src.ml.trainer import evaluate_model, train_and_evaluate
from src.pipeline.training_pipeline import load_month_frame

TAXI_TYPE = "green"
BASELINE_TRAIN = YearMonth(2024, 1)
BASELINE_VALIDATION = YearMonth(2024, 2)
NEWLY_PUBLISHED = YearMonth(2024, 3)

# O recorte que estreita o treino do modelo defeituoso: só corridas curtas. Ele aprende a
# inclinação de uma ponta da distribuição e a extrapola para todas as outras.
NARROW_MAX_DISTANCE_MILES = 1.5


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def train_healthy(models_dir: Path) -> str:
    """Train and promote a model over a normal window - the version to fall back to."""
    train_df = load_month_frame(TAXI_TYPE, BASELINE_TRAIN)
    validation_df = load_month_frame(TAXI_TYPE, BASELINE_VALIDATION)

    result = train_and_evaluate(train_df, validation_df, TAXI_TYPE)
    metadata = save_model(
        result,
        [str(BASELINE_TRAIN), str(BASELINE_VALIDATION)],
        BASELINE_VALIDATION.first_day(),
        models_dir,
    )
    promote_version(TAXI_TYPE, metadata.model_version, models_dir)

    print(f"\nv1 (saudável): {metadata.model_version}")
    print(f"   treinada em {BASELINE_TRAIN}, validada em {BASELINE_VALIDATION}")
    print(f"   RMSE de validação: {result.validation_metrics.rmse:.4f}")
    return metadata.model_version


def train_defective(models_dir: Path) -> str:
    """Train over a narrow slice and promote it, bypassing the gate on purpose."""
    month = load_month_frame(TAXI_TYPE, BASELINE_VALIDATION)
    narrow = month[month["trip_distance"] <= NARROW_MAX_DISTANCE_MILES]

    split = len(narrow) // 2
    result = train_and_evaluate(narrow.iloc[:split], narrow.iloc[split:], TAXI_TYPE)
    metadata = save_model(
        result,
        [str(BASELINE_VALIDATION)],
        BASELINE_VALIDATION.first_day(),
        models_dir,
    )
    promote_version(TAXI_TYPE, metadata.model_version, models_dir)

    print(f"\nv2 (defeituosa): {metadata.model_version}")
    print(
        f"   treinada só em corridas de até {NARROW_MAX_DISTANCE_MILES} milha "
        f"({len(narrow):,} de {len(month):,} do mês)"
    )
    print(f"   RMSE de validação: {result.validation_metrics.rmse:.4f}  <- parece ótimo")
    print("   ...porque a validação herdou o mesmo recorte estreito do treino.")
    print("   Promovida DIRETO, contornando o gate: é a rede seguinte que está sob teste.")
    return metadata.model_version


def _metadata_of(version: str, models_dir: Path) -> ModelMetadata:
    path = models_dir / TAXI_TYPE / f"{version}{METADATA_SUFFIX}"
    return ModelMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))


def show_truth(models_dir: Path, healthy: str, defective: str) -> None:
    """Score both versions over the whole newly published month, which is the real world."""
    recent = load_month_frame(TAXI_TYPE, NEWLY_PUBLISHED)
    print(f"\nSobre o mês inteiro de {NEWLY_PUBLISHED} ({len(recent):,} corridas):")
    for label, version in (("v1 saudável", healthy), ("v2 defeituosa", defective)):
        metadata = _metadata_of(version, models_dir)
        rmse = evaluate_model(load_model(metadata, models_dir), recent).rmse
        print(f"   {label:<16} RMSE {rmse:8.4f}")


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="autopilot-recovery-"))
    models_dir = workspace / "models"
    models_dir.mkdir(parents=True)

    try:
        heading("CENÁRIO - um retreino que rodou sobre uma fatia estreita dos dados")
        healthy = train_healthy(models_dir)
        defective = train_defective(models_dir)
        show_truth(models_dir, healthy, defective)

        heading(f"O PILOTO ACORDA - mês publicado: {NEWLY_PUBLISHED}")
        run = run_autopilot_for_fleet(TAXI_TYPE, NEWLY_PUBLISHED, models_dir=models_dir)

        print(f"\nAção:   {run.action.value}")
        print(f"Motivo: {run.reason}")
        if run.need is not None:
            print(f"\nGatilho detectado: {run.need.trigger.value if run.need.trigger else '-'}")
            if run.need.champion_rmse is not None and run.need.baseline_rmse is not None:
                degradation = run.need.champion_rmse / run.need.baseline_rmse - 1
                print(
                    f"   RMSE do campeão no mês novo: {run.need.champion_rmse:.4f} contra "
                    f"baseline {run.need.baseline_rmse:.4f} ({degradation:+.1%})"
                )

        heading("RESULTADO")
        active = resolve_current_metadata(TAXI_TYPE, models_dir)
        if active is None:
            raise SystemExit("Registro sem modelo ativo ao fim do ciclo - cenário inválido.")
        print(f"\nPonteiro ativo agora: {active.model_version}")
        if active.model_version == healthy:
            print("   -> é a v1 saudável. O piloto reverteu sozinho, sem operador.")
        elif active.model_version == defective:
            print("   -> ainda é a v2 defeituosa. O rollback NÃO agiu.")
        else:
            print("   -> é uma versão nova: o piloto preferiu retreinar a reverter.")

        print(f"\nRollback exercitado: {run.action is AutopilotAction.ROLLED_BACK}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
