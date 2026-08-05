from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from src.core.constants import VALID_TAXI_TYPES
from src.core.exceptions import InvalidTaxiTypeError
from src.core.log import get_logger
from src.core.months import YearMonth
from src.ml.trainer import ModelMetrics, evaluate_model, fit_model

logger = get_logger(__name__)

MonthLoader = Callable[[str, YearMonth], pd.DataFrame]


class RetrainingPolicy(Protocol):
    """Política que determina quais meses passados compõem a janela de treino.

    A política é o objeto de estudo da Fase 5: trocar de política é trocar de estratégia de
    retreino, e o backtest existe para comparar essas estratégias sobre os mesmos dados.
    """

    def training_months(
        self, history: Sequence[YearMonth], target: YearMonth
    ) -> list[YearMonth]: ...


def _months_before(history: Sequence[YearMonth], target: YearMonth) -> list[YearMonth]:
    """Todo mês anterior ao alvo - nunca o próprio, nunca um posterior.

    O corte é aqui, num lugar só, porque é o ponto onde vazamento temporal entraria: um mês
    posterior no treino significa prever o passado sabendo o futuro.
    """
    return [month for month in history if month < target]


@dataclass
class ExpandingWindow:
    """Retreina a cada mês usando todo o histórico disponível."""

    def training_months(
        self, history: Sequence[YearMonth], target: YearMonth
    ) -> list[YearMonth]:
        return _months_before(history, target)


@dataclass
class SlidingWindow:
    """Retreina a cada mês usando apenas os meses mais recentes.

    Esquecer o passado distante é o que permite acompanhar uma mudança de regime - e é
    também o que joga fora sazonalidade, que é o custo da política.
    """

    size: int

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"size inválido: {self.size}. A janela precisa de ao menos um mês.")

    def training_months(
        self, history: Sequence[YearMonth], target: YearMonth
    ) -> list[YearMonth]:
        return _months_before(history, target)[-self.size :]


@dataclass
class FrozenModel:
    """Treina uma única vez, no início da janela, e nunca mais.

    É o contrafactual do trabalho: a curva de erro desta política é o que acontece quando
    ninguém retreina, e é contra ela que o ganho das demais é medido.
    """

    initial_months: int

    def __post_init__(self) -> None:
        if self.initial_months < 1:
            raise ValueError(
                f"initial_months inválido: {self.initial_months}. "
                f"O modelo congelado precisa de ao menos um mês de treino."
            )

    def training_months(
        self, history: Sequence[YearMonth], target: YearMonth
    ) -> list[YearMonth]:
        return _months_before(history, target)[: self.initial_months]


@dataclass
class ReplayWindow:
    """Erro medido em um mês de teste junto com o histórico de treino utilizado."""

    month: YearMonth
    train_months: list[YearMonth]
    train_sample_count: int
    metrics: ModelMetrics


@dataclass
class BacktestResult:
    """Curva de degradação temporal de uma política de retreino em uma frota."""

    taxi_type: str
    policy: str
    windows: list[ReplayWindow]

    def mean_rmse(self) -> float:
        """Média do RMSE de validação ao longo de todos os meses reavaliados."""
        if not self.windows:
            return float("nan")
        return sum(window.metrics.rmse for window in self.windows) / len(self.windows)


def run_replay_backtest(
    taxi_type: str,
    months: Sequence[YearMonth],
    policy: RetrainingPolicy,
    load_month: MonthLoader,
    min_train_months: int = 1,
) -> BacktestResult:
    """Executa o replay histórico mês a mês simulando operação real.

    Cada mês é avaliado por um modelo que só viu meses anteriores a ele, o que reproduz a
    posição de quem estava em produção naquela data. A série de erros resultante é a curva
    de degradação, e comparar políticas sobre a mesma janela é o que permite escolher uma
    estratégia de retreino sem esperar o tempo passar.
    """
    if taxi_type not in VALID_TAXI_TYPES:
        raise InvalidTaxiTypeError(f"taxi_type inválido: '{taxi_type}'. Use 'yellow' ou 'green'.")

    ordered = sorted(months)
    evaluated = ordered[min_train_months:]
    if not evaluated:
        raise ValueError(
            f"min_train_months inválido: {min_train_months}. A janela de "
            f"{len(ordered)} meses não deixa nenhum mês para avaliar."
        )

    # Cada mês é materializado uma única vez: um mês de Yellow Taxi custa dezenas de
    # megabytes, e a varredura reusa o mesmo mês em várias janelas de treino.
    cache: dict[YearMonth, pd.DataFrame] = {}

    def frame_of(month: YearMonth) -> pd.DataFrame:
        if month not in cache:
            cache[month] = load_month(taxi_type, month)
        return cache[month]

    policy_name = type(policy).__name__
    logger.info(
        "backtest_started",
        extra={
            "taxi_type": taxi_type,
            "policy": policy_name,
            "months": [str(month) for month in ordered],
            "evaluated_months": [str(month) for month in evaluated],
        },
    )

    windows: list[ReplayWindow] = []
    for target in evaluated:
        train_months = policy.training_months(ordered, target)
        if not train_months:
            raise ValueError(
                f"A política {policy_name} não indicou nenhum mês de treino para {target}."
            )

        train_df = pd.concat([frame_of(month) for month in train_months], ignore_index=True)
        model = fit_model(train_df, taxi_type)
        metrics = evaluate_model(model, frame_of(target))

        logger.info(
            "backtest_window",
            extra={
                "taxi_type": taxi_type,
                "policy": policy_name,
                "month": str(target),
                "train_months": [str(month) for month in train_months],
                "train_rows": len(train_df),
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
            },
        )
        windows.append(
            ReplayWindow(
                month=target,
                train_months=train_months,
                train_sample_count=len(train_df),
                metrics=metrics,
            )
        )

    result = BacktestResult(taxi_type=taxi_type, policy=policy_name, windows=windows)
    logger.info(
        "backtest_finished",
        extra={"taxi_type": taxi_type, "policy": policy_name, "mean_rmse": result.mean_rmse()},
    )
    return result
