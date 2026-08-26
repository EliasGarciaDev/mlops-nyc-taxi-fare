from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.core.exceptions import CorruptedMetadataError, InsufficientDataError

# Faixas de PSI usadas na indústria e adotadas aqui. São limiares de tamanho de efeito:
MODERATE_PSI_THRESHOLD: float = 0.10
SIGNIFICANT_PSI_THRESHOLD: float = 0.25

DEFAULT_BIN_COUNT: int = 10
MIN_REFERENCE_SAMPLES: int = 100

# Acima disso a feature é tratada como contínua e binada por quantis. Abaixo, cada valor
MAX_CATEGORICAL_VALUES: int = 20

# Piso aplicado à proporção de cada bin antes do logaritmo. Sem ele, um bin que existe na
PROPORTION_FLOOR: float = 1e-6


class DriftSeverity(StrEnum):
    """Classificação da severidade do drift de acordo com as faixas de PSI."""

    STABLE = "stable"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"

    @classmethod
    def from_psi(cls, psi: float) -> "DriftSeverity":
        if psi >= SIGNIFICANT_PSI_THRESHOLD:
            return cls.SIGNIFICANT
        if psi >= MODERATE_PSI_THRESHOLD:
            return cls.MODERATE
        return cls.STABLE


@dataclass
class FeatureBins:
    """Frozen bin edges and reference proportions of a single feature.

    As bordas são congeladas na janela de referência e reutilizadas em toda comparação
    posterior. Recalcular os bins a cada período mediria o binning e não o drift: duas
    distribuições bem diferentes, cada uma reparticionada nos próprios decis, produzem
    proporções quase idênticas e o deslocamento desaparece.
    """

    feature: str
    edges: list[float]
    reference_shares: list[float]
    categories: list[float] | None = None

    @property
    def is_categorical(self) -> bool:
        """Indica se a feature é tratada de forma categórica ou contínua."""
        return self.categories is not None

    @classmethod
    def from_series(cls, feature: str, values: pd.Series, bin_count: int = DEFAULT_BIN_COUNT) -> Self:
        """Gera distribuição de referência: proporções categóricas ou quantis contínuos."""
        clean = pd.to_numeric(values, errors="coerce").dropna()
        distinct = np.unique(clean.to_numpy(dtype=float)) if len(clean) else np.array([0.0])

        if len(distinct) <= MAX_CATEGORICAL_VALUES:
            categories = [float(value) for value in distinct]
            shares = _category_shares(clean, categories)
            return cls(
                feature=feature,
                edges=[],
                # A última posição acumula tudo que não é nenhuma das categorias conhecidas:
                # categoria inédita é drift, não silêncio.
                reference_shares=[*shares, 0.0],
                categories=categories,
            )

        quantiles = np.linspace(0.0, 1.0, bin_count + 1)
        edges = np.unique(np.quantile(clean, quantiles))
        shares = _bin_shares(clean, edges)
        return cls(feature=feature, edges=[float(edge) for edge in edges], reference_shares=shares)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "edges": list(self.edges),
            "reference_shares": list(self.reference_shares),
            "categories": list(self.categories) if self.categories is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        edges = payload["edges"]
        shares = payload["reference_shares"]
        if not isinstance(edges, list) or not isinstance(shares, list):
            raise CorruptedMetadataError(
                f"Perfil de '{payload.get('feature')}' com bordas ou proporções inválidas: "
                f"edges é {type(edges).__name__}, reference_shares é {type(shares).__name__}."
            )
        raw_categories = payload.get("categories")
        return cls(
            feature=str(payload["feature"]),
            edges=[float(edge) for edge in edges],
            reference_shares=[float(share) for share in shares],
            categories=(
                [float(value) for value in raw_categories]
                if isinstance(raw_categories, list)
                else None
            ),
        )


def _bin_shares(
    values: pd.Series | NDArray[np.float64], edges: NDArray[np.float64] | list[float]
) -> list[float]:
    """Calcula a proporção de observações em cada faixa de quantil fixada.

    Valores fora do intervalo da referência caem nos bins extremos em vez de serem
    descartados: um deslocamento que leva a massa para fora da faixa conhecida é exatamente
    o drift que se quer enxergar.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return [0.0] * (len(edges) - 1)

    indices = np.clip(np.searchsorted(np.asarray(edges), array, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(indices, minlength=len(edges) - 1)
    return [float(share) for share in counts / array.size]


def _category_shares(values: pd.Series, categories: list[float]) -> list[float]:
    """Calcula a proporção de observações em cada categoria observada."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return [0.0] * len(categories)
    return [float(np.count_nonzero(array == category) / array.size) for category in categories]


def ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    """Calcula a estatística D de Kolmogorov-Smirnov entre duas distribuições.

    Devolve a magnitude, nunca o p-value. Em amostras da ordem de milhões o p-value responde
    sempre "sim, é diferente" e deixa de informar; D responde "diferente o quanto", que é o
    que um limiar operacional precisa.
    """
    left = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    right = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if left.size == 0 or right.size == 0:
        return 0.0

    left.sort()
    right.sort()
    grid = np.union1d(left, right)
    left_cdf = np.searchsorted(left, grid, side="right") / left.size
    right_cdf = np.searchsorted(right, grid, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def population_stability_index(bins: FeatureBins, current: pd.Series) -> float:
    """Calcula o Population Stability Index (PSI) em relação à base de referência."""
    clean = pd.to_numeric(current, errors="coerce").dropna()

    if bins.categories is not None:
        shares = _category_shares(clean, bins.categories)
        # O resto - tudo que não caiu em categoria conhecida - ocupa a última posição.
        current_shares = np.asarray([*shares, max(0.0, 1.0 - sum(shares))], dtype=float)
    else:
        current_shares = np.asarray(_bin_shares(clean, bins.edges), dtype=float)

    reference_shares = np.asarray(bins.reference_shares, dtype=float)

    expected = np.maximum(reference_shares, PROPORTION_FLOOR)
    observed = np.maximum(current_shares, PROPORTION_FLOOR)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


@dataclass
class ReferenceProfile:
    """Perfil de referência da distribuição das features no conjunto de treino.

    Viaja junto do artefato: sem o baseline de quando o modelo foi treinado, uma janela de
    produção não tem contra o que ser comparada, e o monitoramento vira opinião.
    """

    bins: dict[str, FeatureBins]
    sample_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "bins": {name: feature_bins.to_dict() for name, feature_bins in self.bins.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        raw_bins = payload["bins"]
        if not isinstance(raw_bins, dict):
            raise CorruptedMetadataError(
                f"Campo 'bins' deveria ser um objeto, veio {type(raw_bins).__name__}."
            )
        return cls(
            bins={
                str(name): FeatureBins.from_dict(value)
                for name, value in raw_bins.items()
                if isinstance(value, dict)
            },
            sample_count=int(str(payload["sample_count"])),
        )


def build_reference_profile(
    df: pd.DataFrame, features: list[str], bin_count: int = DEFAULT_BIN_COUNT
) -> ReferenceProfile:
    """Gera o perfil de referência das distribuições com bordas de quantil fixadas."""
    missing = [name for name in features if name not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes no DataFrame de referência: {', '.join(missing)}.")

    if len(df) < MIN_REFERENCE_SAMPLES:
        raise InsufficientDataError(
            f"Janela de referência insuficiente: {len(df)} linhas. "
            f"Mínimo de {MIN_REFERENCE_SAMPLES} para descrever uma distribuição."
        )

    return ReferenceProfile(
        bins={name: FeatureBins.from_series(name, df[name], bin_count) for name in features},
        sample_count=len(df),
    )


@dataclass
class FeatureDrift:
    """How much one feature moved away from its reference distribution.

    `ks_statistic` só é preenchido quando as duas amostras existem de fato. A partir do
    perfil serializado ele ficaria de fora: reconstruir uma amostra a partir dos bins produz
    um número que parece uma medida e não é - o PSI é o que decide.
    """

    feature: str
    psi: float
    severity: DriftSeverity
    ks_statistic: float | None = None


@dataclass
class DriftReport:
    """Relatório consolidado de drift de todas as features em uma janela de produção.

    Mede **data drift**: o deslocamento de P(X). Não diz nada sobre concept drift, que é a
    mudança de P(y|X) e só aparece com rótulo, pela degradação do erro. Os dois se
    complementam - data drift estável junto de erro crescente é a evidência mais forte de
    que a relação mudou, e não as entradas.
    """

    results: list[FeatureDrift]
    missing_features: list[str]
    sample_count: int

    def drifted_features(self) -> list[str]:
        return [
            result.feature for result in self.results if result.severity is not DriftSeverity.STABLE
        ]

    def has_drift(self) -> bool:
        return bool(self.drifted_features())


def compare_to_reference(profile: ReferenceProfile, current: pd.DataFrame) -> DriftReport:
    """Compara uma janela de produção com a distribuição de referência de treino.

    Feature ausente na janela atual é registrada à parte, não como deslocamento: coluna que
    desapareceu é schema drift, e a resposta a isso é corrigir o contrato de dados - não
    retreinar sobre uma distribuição que não existe.
    """
    results: list[FeatureDrift] = []
    missing: list[str] = []

    for name, bins in profile.bins.items():
        if name not in current.columns:
            missing.append(name)
            continue
        psi = population_stability_index(bins, current[name])
        results.append(
            FeatureDrift(feature=name, psi=psi, severity=DriftSeverity.from_psi(psi))
        )

    results.sort(key=lambda result: result.psi, reverse=True)
    return DriftReport(results=results, missing_features=missing, sample_count=len(current))


def compare_samples(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    bin_count: int = DEFAULT_BIN_COUNT,
) -> DriftReport:
    """Compara duas janelas de dados calculando métricas de PSI e teste KS.

    Usado na análise offline - backtest, estudo de uma janela histórica - onde a amostra de
    treino ainda está em memória. Em produção só existe o perfil serializado, e aí vale o
    `compare_to_reference`.
    """
    results: list[FeatureDrift] = []
    missing: list[str] = []

    for name in features:
        if name not in current.columns or name not in reference.columns:
            missing.append(name)
            continue
        bins = FeatureBins.from_series(name, reference[name], bin_count)
        psi = population_stability_index(bins, current[name])
        results.append(
            FeatureDrift(
                feature=name,
                psi=psi,
                severity=DriftSeverity.from_psi(psi),
                ks_statistic=ks_statistic(reference[name], current[name]),
            )
        )

    results.sort(key=lambda result: result.psi, reverse=True)
    return DriftReport(results=results, missing_features=missing, sample_count=len(current))
