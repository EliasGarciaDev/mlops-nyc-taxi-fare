from datetime import date

from src.core.months import (
    YearMonth,
    available_month_range,
    month_range,
)


class TestYearMonth:
    def test_it_renders_and_orders_like_the_calendar(self):
        assert str(YearMonth(2024, 3)) == "2024-03"
        assert YearMonth(2024, 3) < YearMonth(2024, 11) < YearMonth(2025, 1)

    def test_shifting_crosses_the_year_boundary(self):
        assert YearMonth(2024, 12).shifted(1) == YearMonth(2025, 1)
        assert YearMonth(2024, 1).shifted(-1) == YearMonth(2023, 12)

    def test_the_first_day_is_naive_like_the_tlc_timestamps(self):
        primeiro = YearMonth(2024, 3).first_day()
        assert primeiro.isoformat() == "2024-03-01T00:00:00"
        assert primeiro.tzinfo is None


class TestWindow:
    def test_the_range_covers_both_ends(self):
        meses = month_range(YearMonth(2024, 11), YearMonth(2025, 2))
        assert meses == [
            YearMonth(2024, 11),
            YearMonth(2024, 12),
            YearMonth(2025, 1),
            YearMonth(2025, 2),
        ]

    def test_the_available_window_follows_the_publication_lag(self):
        """A TLC publica com atraso: fixar o limite superior faria o retreino parar sozinho."""
        primeiro, ultimo = available_month_range(date(2026, 8, 25))
        assert primeiro == YearMonth(2024, 1)
        assert ultimo < YearMonth(2026, 8)
