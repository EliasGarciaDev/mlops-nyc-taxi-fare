import pandas as pd
from sklearn.linear_model import LinearRegression

from src.core.constants import FEATURE_ORDER, TARGET_COLUMN
from src.ml.promotion import (
    PromotionOutcome,
    default_segments,
    evaluate_promotion,
)

# ---------------------------------------------------------------------------
# Helpers - modelos lineares com erro controlado, sem depender de treino real
# ---------------------------------------------------------------------------


class FixedErrorModel:
    """Prediz o alvo com um viés constante, para que o RMSE seja conhecido de antemão."""

    def __init__(self, bias: float, target: pd.Series, segment_bias: pd.Series | None = None):
        self._values = target + bias if segment_bias is None else target + bias + segment_bias

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return self._values.loc[features.index]


def make_validation_frame(rows: int = 600) -> pd.DataFrame:
    frame = pd.DataFrame({name: [float(index % 5) for index in range(rows)] for name in FEATURE_ORDER})
    frame["trip_distance"] = [1.0 + (index % 20) for index in range(rows)]
    frame["hour_of_day"] = [index % 24 for index in range(rows)]
    frame["is_airport_trip"] = [1 if index % 10 == 0 else 0 for index in range(rows)]
    frame["is_congestion_zone"] = [1 if index % 4 == 0 else 0 for index in range(rows)]
    frame[TARGET_COLUMN] = 3.0 + 2.5 * frame["trip_distance"]
    return frame


def fitted_model(frame: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(frame[FEATURE_ORDER], frame[TARGET_COLUMN])
    return model


# ---------------------------------------------------------------------------
# 1. A decisão central: o challenger só entra se vencer por margem

class TestPromotionGate:
    def test_a_clearly_better_challenger_is_promoted(self):
        frame = make_validation_frame()
        decisao = evaluate_promotion(
            champion=FixedErrorModel(4.0, frame[TARGET_COLUMN]),
            challenger=FixedErrorModel(1.0, frame[TARGET_COLUMN]),
            validation_df=frame,
        )
        assert decisao.promoted is True
        assert decisao.outcome is PromotionOutcome.PROMOTED

    def test_a_worse_challenger_is_rejected(self):
        frame = make_validation_frame()
        decisao = evaluate_promotion(
            champion=FixedErrorModel(1.0, frame[TARGET_COLUMN]),
            challenger=FixedErrorModel(4.0, frame[TARGET_COLUMN]),
            validation_df=frame,
        )
        assert decisao.promoted is False
        assert decisao.outcome is PromotionOutcome.REJECTED_WORSE

    def test_an_improvement_below_the_margin_is_rejected(self):
        """Melhora dentro do ruído troca o modelo de produção sem ganho real."""
        frame = make_validation_frame()
        decisao = evaluate_promotion(
            champion=FixedErrorModel(4.0, frame[TARGET_COLUMN]),
            challenger=FixedErrorModel(3.99, frame[TARGET_COLUMN]),
            validation_df=frame,
        )
        assert decisao.outcome is PromotionOutcome.REJECTED_INSUFFICIENT_MARGIN

    def test_the_first_version_of_a_fleet_is_promoted_without_a_contest(self):
        """Recusar deixaria o sistema sem modelo nenhum."""
        frame = make_validation_frame()
        decisao = evaluate_promotion(
            champion=None,
            challenger=FixedErrorModel(1.0, frame[TARGET_COLUMN]),
            validation_df=frame,
        )
        assert decisao.outcome is PromotionOutcome.PROMOTED_FIRST


class TestSegmentVeto:
    def test_a_challenger_that_ruins_one_segment_is_rejected(self):
        """Um modelo pode melhorar na média e destruir um subgrupo."""
        frame = make_validation_frame()
        dano = frame["is_airport_trip"] * 30.0
        decisao = evaluate_promotion(
            champion=FixedErrorModel(4.0, frame[TARGET_COLUMN]),
            challenger=FixedErrorModel(1.0, frame[TARGET_COLUMN], segment_bias=dano),
            validation_df=frame,
        )
        assert decisao.promoted is False
        assert any(resultado.regressed for resultado in decisao.segments)

    def test_a_small_wobble_does_not_block_a_good_challenger(self):
        """Limiar zero reprovaria qualquer challenger por ruído em algum recorte."""
        frame = make_validation_frame()
        ruido = frame["is_airport_trip"] * 0.01
        decisao = evaluate_promotion(
            champion=FixedErrorModel(4.0, frame[TARGET_COLUMN]),
            challenger=FixedErrorModel(1.0, frame[TARGET_COLUMN], segment_bias=ruido),
            validation_df=frame,
        )
        assert decisao.promoted is True

    def test_a_segment_without_enough_rows_is_skipped(self):
        frame = make_validation_frame()
        frame["is_airport_trip"] = 0
        decisao = evaluate_promotion(
            champion=FixedErrorModel(4.0, frame[TARGET_COLUMN]),
            challenger=FixedErrorModel(1.0, frame[TARGET_COLUMN]),
            validation_df=frame,
        )
        assert all(resultado.segment != "airport" for resultado in decisao.segments)

    def test_manhattan_has_no_veto_because_the_average_already_protects_it(self):
        """Manhattan é 91% da janela: um veto dela seria o mesmo teste duas vezes."""
        assert "dropoff_manhattan" not in default_segments()
