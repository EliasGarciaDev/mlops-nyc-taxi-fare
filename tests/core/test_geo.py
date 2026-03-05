
from src.core.geo import perpendicular_distance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SQUARE = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]


def straight_line(points: int, spacing: float = 0.001) -> list[tuple[float, float]]:
    """Anel cujo lado superior é uma reta com muitos vértices redundantes."""
    top = [(index * spacing, 1.0) for index in range(points)]
    return [(0.0, 0.0), *top, ((points - 1) * spacing, 0.0), (0.0, 0.0)]


# ---------------------------------------------------------------------------
# 1. Distância perpendicular
# ---------------------------------------------------------------------------


class TestPerpendicularDistance:
    def test_point_on_the_segment_is_at_zero(self):
        assert perpendicular_distance((0.5, 0.0), (0.0, 0.0), (1.0, 0.0)) == 0.0
