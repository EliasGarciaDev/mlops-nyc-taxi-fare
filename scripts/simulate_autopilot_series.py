#!/usr/bin/env python3
"""Executa a simulação mensal do piloto autônomo demonstrando o ciclo de vida completo.

A execução de `documentacao` provou que o ciclo fecha, e o cenário de
recuperação provou que as redes de segurança pegam. Nenhum dos dois mostra o que o título do
trabalho promete: o **ciclo de vida**. Um mês é uma decisão; catorze meses são um comportamento.

A diferença para o backtest doé o que está sendo medido. O backtest compara políticas
de retreino com a decisão já tomada de fora - retreina sempre, nunca, ou por janela. Aqui quem
decide é o próprio sistema, com o gatilho combinado, o gate e o rollback no caminho. O que sai
não é a curva de uma política: é o histórico de operação de um sistema que se manteve sozinho.

O carregamento mensal é memoizado, como no backtest, porque a janela deslizante relê os mesmos
meses a cada ciclo e a TLC seria consultada dezenas de vezes para os mesmos dados.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.ml.autopilot as autopilot_module
import src.ml.retraining as retraining_module
from src.cli.train import parse_year_month
from src.core.months import YearMonth
from src.ml.autopilot import AutopilotAction, AutopilotRun, run_autopilot_for_fleet
from src.ml.registry import promote_version, resolve_current_metadata, save_model
from src.ml.trainer import train_and_evaluate
from src.pipeline.training_pipeline import load_month_frame

SEED_TRAIN = YearMonth(2024, 1)
SEED_VALIDATION = YearMonth(2024, 2)

ACTION_LABEL = {
    AutopilotAction.HELD: "segurou",
    AutopilotAction.RETRAINED_PROMOTED: "promoveu",
    AutopilotAction.RETRAINED_REJECTED: "reprovou",
    AutopilotAction.ROLLED_BACK: "reverteu",
    AutopilotAction.FAILED: "falhou",
}


def install_month_cache() -> dict[tuple[str, str], pd.DataFrame]:
    """Memoize monthly loading across the whole series, in every module that reads months."""
    cache: dict[tuple[str, str], pd.DataFrame] = {}

    def cached(taxi_type: str, month: YearMonth) -> pd.DataFrame:
        key = (taxi_type, str(month))
        if key not in cache:
            cache[key] = load_month_frame(taxi_type, month)
        return cache[key]

    autopilot_module.load_month_frame = cached
    retraining_module.load_month_frame = cached
    return cache


def seed_registry(taxi_type: str, models_dir: Path) -> str:
    """Train the first champion, the way `make train` would."""
    train_df = load_month_frame(taxi_type, SEED_TRAIN)
    validation_df = load_month_frame(taxi_type, SEED_VALIDATION)

    result = train_and_evaluate(train_df, validation_df, taxi_type)
    metadata = save_model(
        result,
        [str(SEED_TRAIN), str(SEED_VALIDATION)],
        SEED_VALIDATION.first_day(),
        models_dir,
    )
    promote_version(taxi_type, metadata.model_version, models_dir)

    print(f"Campeão inicial: {metadata.model_version}")
    print(f"   treinado em {SEED_TRAIN}, validado em {SEED_VALIDATION}")
    print(f"   RMSE de validação: {result.validation_metrics.rmse:.4f}\n")
    return metadata.model_version


def describe(run: AutopilotRun) -> tuple[str, str, str]:
    """Reduce one pass to the three columns that matter in a series."""
    trigger = run.need.trigger.value if run.need is not None and run.need.trigger else "-"
    champion_rmse = (
        f"{run.need.champion_rmse:.4f}"
        if run.need is not None and run.need.champion_rmse is not None
        else "-"
    )
    return ACTION_LABEL[run.action], trigger, champion_rmse


def months_between(first: YearMonth, last: YearMonth) -> list[YearMonth]:
    months: list[YearMonth] = []
    current = first
    while current <= last:
        months.append(current)
        current = current.shifted(1)
    return months


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxi-type", default="green", choices=("yellow", "green"))
    parser.add_argument("--from", dest="first", default="2024-03")
    parser.add_argument("--to", dest="last", default="2025-03")
    args = parser.parse_args()

    first, last = parse_year_month(args.first), parse_year_month(args.last)
    install_month_cache()

    workspace = Path(tempfile.mkdtemp(prefix="autopilot-series-"))
    models_dir = workspace / "models"
    models_dir.mkdir(parents=True)

    try:
        print(f"Série do piloto - frota {args.taxi_type}, de {first} a {last}\n")
        seed_registry(args.taxi_type, models_dir)

        print(f"{'mês':>9} {'ação':>10} {'gatilho':>18} {'RMSE campeão':>14}  versão ativa")
        tally: dict[str, int] = {}
        promotions = 0

        # Toda versão que já serviu, para saber se um rollback voltou para algo que
        # realmente foi campeão - ou para um challenger que o gate havia reprovado.
        seed = resolve_current_metadata(args.taxi_type, models_dir)
        ever_champion = {seed.model_version} if seed is not None else set()
        suspicious: list[tuple[str, str]] = []

        for month in months_between(first, last):
            run = run_autopilot_for_fleet(args.taxi_type, month, models_dir=models_dir)
            action, trigger, champion_rmse = describe(run)
            tally[action] = tally.get(action, 0) + 1
            if run.action is AutopilotAction.RETRAINED_PROMOTED:
                promotions += 1

            active = resolve_current_metadata(args.taxi_type, models_dir)
            version = active.model_version if active is not None else "-"

            flag = ""
            if run.action is AutopilotAction.ROLLED_BACK and version not in ever_champion:
                flag = "  <-- NUNCA FOI CAMPEÃ"
                suspicious.append((str(month), version))
            ever_champion.add(version)

            print(f"{month!s:>9} {action:>10} {trigger:>18} {champion_rmse:>14}  {version}{flag}")

        print(f"\n{len(months_between(first, last))} ciclos, sem operador:")
        for action, count in sorted(tally.items(), key=lambda item: -item[1]):
            print(f"   {action:>10}: {count}")
        print(f"\nModelos que entraram em produção ao longo da série: {promotions}")
        if suspicious:
            print(
                f"\nATENÇÃO: {len(suspicious)} reversão(ões) para versão que nunca foi campeã.\n"
                "O rollback anda para trás na ordem de GRAVAÇÃO do registro, e challengers\n"
                "reprovados também são gravados - então ele pode promover o que o gate barrou."
            )
            for month, version in suspicious:
                print(f"   {month}: {version}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
