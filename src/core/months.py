from datetime import UTC, date, datetime
from typing import NamedTuple
from zoneinfo import ZoneInfo

from src.core.constants import (
    FIRST_SUPPORTED_MONTH,
    MAX_MONTH,
    MIN_MONTH,
    NYC_TIMEZONE,
    PUBLICATION_LAG_MONTHS,
)

MONTHS_IN_YEAR = 12


class YearMonth(NamedTuple):
    """One monthly TLC dataset, identified the same way the source files are.

    A ordem natural da tupla - ano antes de mês - já é a ordem cronológica, então
    comparação e ordenação funcionam sem código adicional.
    """

    year: int
    month: int

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def first_day(self) -> datetime:
        """Return the first instant of the month, naive like the TLC timestamps."""
        return datetime(self.year, self.month, 1)  # noqa: DTZ001

    def shifted(self, months: int) -> "YearMonth":
        """Move the calendar forward or backward by whole months."""
        total = self.year * MONTHS_IN_YEAR + (self.month - 1) + months
        return YearMonth(total // MONTHS_IN_YEAR, total % MONTHS_IN_YEAR + 1)


def today_utc() -> date:
    """The current date in UTC, isolated here so tests can replace it."""
    return datetime.now(UTC).date()


def now_in_nyc(moment: datetime | None = None) -> datetime:
    """The current wall clock in New York, naive like the TLC timestamps.

    O dataset publica horários sem fuso, medidos na cidade. Comparar um `pickup_datetime`
    com "agora em UTC" desloca a referência em quatro ou cinco horas conforme a estação -
    e o horário de verão é exatamente por que o deslocamento não pode ser uma constante.
    """
    reference = moment if moment is not None else datetime.now(UTC)
    return reference.astimezone(ZoneInfo(NYC_TIMEZONE)).replace(tzinfo=None)


def month_range(start: YearMonth, end: YearMonth) -> list[YearMonth]:
    """Expand an inclusive interval into the monthly datasets it covers."""
    if end < start:
        raise ValueError(f"Intervalo invertido: {start} vem depois de {end}.")

    months = []
    current = start
    while current <= end:
        months.append(current)
        current = current.shifted(1)
    return months


def latest_published_month(today: date) -> YearMonth:
    """Return the most recent month expected to be available at the TLC."""
    return YearMonth(today.year, today.month).shifted(-PUBLICATION_LAG_MONTHS)


def available_month_range(today: date) -> tuple[YearMonth, YearMonth]:
    """Return the interval of monthly datasets the pipeline can currently request."""
    return YearMonth(*FIRST_SUPPORTED_MONTH), latest_published_month(today)


def validate_month_is_available(year: int, month: int, today: date) -> None:
    """Reject a month outside the calendar or outside what the TLC has published."""
    if not MIN_MONTH <= month <= MAX_MONTH:
        raise ValueError(f"month inválido: {month}. Use um valor entre {MIN_MONTH} e {MAX_MONTH}.")

    requested = YearMonth(year, month)
    first, last = available_month_range(today)
    if not first <= requested <= last:
        raise ValueError(
            f"Mês indisponível: {requested}. "
            f"O intervalo suportado hoje vai de {first} a {last} - a TLC publica cada mês "
            f"com cerca de {PUBLICATION_LAG_MONTHS} meses de atraso."
        )
