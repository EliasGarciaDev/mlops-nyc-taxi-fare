import pandas as pd
import pytest

from src.core.constants import (
    JFK_FLAT_FARE_AMOUNT,
    JFK_LOCATION_ID,
    MIN_PLAUSIBLE_TOTAL_AMOUNT,
)
from src.core.exceptions import CorruptedMetadataError
from src.ml.fare_rules import (
    MIN_CALIBRATION_SAMPLES,
    FlatFareCalibration,
    PricingRule,
    apply_fare_rules,
    calibrate_flat_fare,
    is_flat_fare_trip,
)

# Zonas usadas nos testes, do índice oficial da TLC.
TIMES_SQUARE = 230
UPPER_EAST_SIDE = 236
LAGUARDIA = 138
NEWARK = 1
ASTORIA = 7
BROOKLYN_HEIGHTS = 33

CALIBRATION = FlatFareCalibration(mean_excess=24.06, sample_count=171_939)


def make_trips(pairs: list[tuple[int, int]], total: float = 94.0) -> pd.DataFrame:
    """Retorna um DataFrame de corridas nas zonas dadas. O total pode ser sobrescrito."""
    return pd.DataFrame(
        {
            "PULocationID": [pickup for pickup, _ in pairs],
            "DOLocationID": [dropoff for _, dropoff in pairs],
            "total_amount": [total] * len(pairs),
        }
    )


class TestFlatFareRule:
    def test_the_rule_fires_between_jfk_and_manhattan_in_both_directions(self):
        assert is_flat_fare_trip(JFK_LOCATION_ID, TIMES_SQUARE) is True
        assert is_flat_fare_trip(TIMES_SQUARE, JFK_LOCATION_ID) is True

    def test_only_jfk_has_a_flat_fare(self):
        """LaGuardia e Newark são taxímetro; só o JFK tem tarifa regulada por constante."""
        assert is_flat_fare_trip(LAGUARDIA, TIMES_SQUARE) is False
        assert is_flat_fare_trip(NEWARK, TIMES_SQUARE) is False
        assert is_flat_fare_trip(JFK_LOCATION_ID, BROOKLYN_HEIGHTS) is False

    def test_a_trip_inside_manhattan_is_not_a_flat_fare(self):
        assert is_flat_fare_trip(TIMES_SQUARE, UPPER_EAST_SIDE) is False

    def test_the_rule_overrides_the_model_in_both_directions_of_error(self):
        """O modelo erra para os dois lados no segmento: a regra não é teto nem piso."""
        baixo = apply_fare_rules(40.0, JFK_LOCATION_ID, TIMES_SQUARE, CALIBRATION)
        alto = apply_fare_rules(200.0, JFK_LOCATION_ID, TIMES_SQUARE, CALIBRATION)
        esperado = JFK_FLAT_FARE_AMOUNT + CALIBRATION.mean_excess
        assert baixo.amount == pytest.approx(esperado)
        assert alto.amount == pytest.approx(esperado)
        assert baixo.rule is PricingRule.JFK_FLAT_FARE

    def test_an_ordinary_trip_keeps_the_model_prediction(self):
        resultado = apply_fare_rules(31.5, TIMES_SQUARE, ASTORIA, CALIBRATION)
        assert resultado.amount == pytest.approx(31.5)
        assert resultado.rule is PricingRule.MODEL

    def test_without_calibration_the_rule_abstains(self):
        """Aplicar só os US$ 70,00 devolveria o taxímetro onde o contrato promete o total."""
        resultado = apply_fare_rules(120.0, JFK_LOCATION_ID, TIMES_SQUARE, None)
        assert resultado.amount == pytest.approx(120.0)

    def test_a_forged_request_cannot_claim_the_flat_fare(self):
        """`apply_fare_rules` nem recebe o RatecodeID — a ausência do parâmetro é a garantia."""
        forjada = apply_fare_rules(8.0, ASTORIA, BROOKLYN_HEIGHTS, CALIBRATION)
        assert forjada.amount == pytest.approx(8.0)
        assert forjada.rule is PricingRule.MODEL


class TestFareFloor:
    def test_an_implausible_prediction_is_raised(self):
        resultado = apply_fare_rules(1.25, TIMES_SQUARE, UPPER_EAST_SIDE, CALIBRATION)
        assert resultado.amount == pytest.approx(MIN_PLAUSIBLE_TOTAL_AMOUNT)
        assert resultado.rule is PricingRule.MINIMUM_FARE

    def test_the_floor_holds_on_both_sides(self):
        no_limite = apply_fare_rules(MIN_PLAUSIBLE_TOTAL_AMOUNT, ASTORIA, ASTORIA, CALIBRATION)
        acima = apply_fare_rules(MIN_PLAUSIBLE_TOTAL_AMOUNT + 0.01, ASTORIA, ASTORIA, CALIBRATION)
        assert no_limite.rule is PricingRule.MODEL
        assert acima.amount == pytest.approx(MIN_PLAUSIBLE_TOTAL_AMOUNT + 0.01)


class TestCalibration:
    def test_the_excess_is_measured_against_the_regulated_amount(self):
        viagens = make_trips([(JFK_LOCATION_ID, TIMES_SQUARE)] * MIN_CALIBRATION_SAMPLES, total=94.0)
        calibracao = calibrate_flat_fare(viagens)
        assert calibracao is not None
        assert calibracao.mean_excess == pytest.approx(94.0 - JFK_FLAT_FARE_AMOUNT)

    def test_only_zone_confirmed_trips_are_calibrated_on(self):
        """Corridas que não são JFK↔Manhattan não podem puxar a média do excedente."""
        viagens = pd.concat(
            [
                make_trips([(JFK_LOCATION_ID, TIMES_SQUARE)] * MIN_CALIBRATION_SAMPLES, total=94.0),
                make_trips([(ASTORIA, ASTORIA)] * 500, total=15.0),
            ],
            ignore_index=True,
        )
        calibracao = calibrate_flat_fare(viagens)
        assert calibracao is not None
        assert calibracao.mean_excess == pytest.approx(24.0)

    def test_too_few_samples_yields_no_calibration(self):
        poucas = make_trips([(JFK_LOCATION_ID, TIMES_SQUARE)] * (MIN_CALIBRATION_SAMPLES - 1))
        assert calibrate_flat_fare(poucas) is None

    def test_it_survives_the_round_trip_in_the_artifact(self):
        assert FlatFareCalibration.from_dict(CALIBRATION.to_dict()) == CALIBRATION

    def test_a_corrupted_calibration_is_rejected(self):
        with pytest.raises(CorruptedMetadataError):
            FlatFareCalibration.from_dict({"mean_excess": "24.06", "sample_count": 10})
