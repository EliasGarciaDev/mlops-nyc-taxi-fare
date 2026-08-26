import json

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import InsufficientDataError
from src.monitoring.drift import (
    MODERATE_PSI_THRESHOLD,
    SIGNIFICANT_PSI_THRESHOLD,
    DriftSeverity,
    ReferenceProfile,
    build_reference_profile,
    ks_statistic,
)


def make_frame(values: dict[str, list[float] | np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(values)


def normal(mean: float, size: int = 5_000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, 1.0, size)


# ---------------------------------------------------------------------------

class TestEffectSizeMetrics:
    def test_identical_distributions_show_no_drift(self):
        valores = pd.Series(np.linspace(0, 100, 500))
        assert ks_statistic(valores, valores) == pytest.approx(0.0, abs=1e-9)

    def test_a_shifted_distribution_is_detected(self):
        base = pd.Series(np.linspace(0, 100, 500))
        deslocada = base + 60
        assert ks_statistic(base, deslocada) > 0.3

    def test_the_severity_bands_follow_the_industry_thresholds(self):
        """PSI é tamanho de efeito, não p-value: sobre milhões de linhas qualquer p-value
        cruza qualquer limiar."""
        assert DriftSeverity.from_psi(0.05) is DriftSeverity.STABLE
        assert DriftSeverity.from_psi(MODERATE_PSI_THRESHOLD) is DriftSeverity.MODERATE
        assert DriftSeverity.from_psi(SIGNIFICANT_PSI_THRESHOLD) is DriftSeverity.SIGNIFICANT


class TestReferenceProfileLifecycle:
    def _frame(self, deslocamento: float = 0.0) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return make_frame({"trip_distance": rng.normal(5 + deslocamento, 2, 400)})

    def test_a_profile_describes_every_requested_feature(self):
        perfil = build_reference_profile(self._frame(), ["trip_distance"])
        assert set(perfil.bins) == {"trip_distance"}

    def test_a_window_too_small_to_describe_is_refused(self):
        pequeno = make_frame({"trip_distance": [1.0, 2.0, 3.0]})
        with pytest.raises(InsufficientDataError):
            build_reference_profile(pequeno, ["trip_distance"])

    def test_the_profile_survives_the_round_trip_to_json(self):
        """O baseline viaja dentro do artefato, então precisa serializar."""
        perfil = build_reference_profile(self._frame(), ["trip_distance"])
        recuperado = ReferenceProfile.from_dict(json.loads(json.dumps(perfil.to_dict())))
        assert recuperado.bins["trip_distance"].edges == perfil.bins["trip_distance"].edges
