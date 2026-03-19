from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.core.constants import (
    FEATURE_ORDER,
    MIN_EVALUATION_SAMPLES,
    MIN_TRAINING_SAMPLES,
    TARGET_COLUMN,
    VALID_TAXI_TYPES,
)
from src.core.exceptions import InsufficientDataError, InvalidTaxiTypeError
from src.ml.protocols import Predictor

REQUIRED_FEATURES = [TARGET_COLUMN, *FEATURE_ORDER]


@dataclass
class ModelMetrics:
    """Regression error measured over a single dataset."""

    rmse: float
    mae: float
    r2: float
    sample_count: int


@dataclass
class TrainingResult:
    """Modelo ajustado acompanhado das métricas de generalização em validação.

    As duas métricas existem separadas de propósito: o erro de treino sozinho não diz
    nada sobre generalização, e a diferença entre as duas é a medida de overfitting.
    """

    model: LinearRegression
    taxi_type: str
    trained_at: datetime
    train_metrics: ModelMetrics
    validation_metrics: ModelMetrics


def _require_taxi_type(taxi_type: str) -> None:
    if taxi_type not in VALID_TAXI_TYPES:
        raise InvalidTaxiTypeError(f"taxi_type inválido: '{taxi_type}'. Use 'yellow' ou 'green'.")


def _require_feature_columns(df: pd.DataFrame, dataset_name: str) -> None:
    missing = [column for column in REQUIRED_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes em {dataset_name}: {', '.join(missing)}"
        )


def fit_model(df: pd.DataFrame, taxi_type: str) -> LinearRegression:
    """Ajusta um modelo de Regressão Linear sobre a partição de treino."""
    _require_taxi_type(taxi_type)

    if len(df) < MIN_TRAINING_SAMPLES:
        raise InsufficientDataError(
            f"DataFrame de treino insuficiente: {len(df)} linhas. "
            f"Mínimo de {MIN_TRAINING_SAMPLES} linhas exigido."
        )

    _require_feature_columns(df, "treino")

    model = LinearRegression()
    model.fit(df[FEATURE_ORDER], df[TARGET_COLUMN])
    return model


def evaluate_model(model: Predictor, df: pd.DataFrame) -> ModelMetrics:
    """Measure regression error of a fitted model over an arbitrary dataset."""
    if len(df) < MIN_EVALUATION_SAMPLES:
        raise InsufficientDataError(
            f"DataFrame de avaliação insuficiente: {len(df)} linhas. "
            f"Mínimo de {MIN_EVALUATION_SAMPLES} linhas exigido para R² ser definido."
        )

    _require_feature_columns(df, "avaliação")

    observed = df[TARGET_COLUMN]
    predicted = model.predict(df[FEATURE_ORDER])

    return ModelMetrics(
        rmse=float(mean_squared_error(observed, predicted) ** 0.5),
        mae=float(mean_absolute_error(observed, predicted)),
        r2=float(r2_score(observed, predicted)),
        sample_count=len(df),
    )


def train_and_evaluate(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, taxi_type: str
) -> TrainingResult:
    """Treina com o histórico e avalia no período futuro, evitando vazamento de dados."""
    trained_at = datetime.now(UTC)
    model = fit_model(train_df, taxi_type)

    return TrainingResult(
        model=model,
        taxi_type=taxi_type,
        trained_at=trained_at,
        train_metrics=evaluate_model(model, train_df),
        validation_metrics=evaluate_model(model, validation_df),
    )
