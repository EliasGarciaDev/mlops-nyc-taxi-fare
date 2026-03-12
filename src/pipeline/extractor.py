import io
import time
import urllib.error
import urllib.request
from datetime import date

import pandas as pd

from src.core.constants import (
    DATETIME_COLUMNS_MAP,
    MAX_DISCOVERY_MONTHS,
    NYC_TLC_BASE_URL,
    VALID_TAXI_TYPES,
)
from src.core.exceptions import DataExtractionError, InvalidTaxiTypeError
from src.core.log import get_logger
from src.core.months import YearMonth, today_utc, validate_month_is_available

logger = get_logger(__name__)

DOWNLOAD_TIMEOUT_SECONDS: int = 120
# A sondagem é um HEAD e não transfere corpo: esperar dois minutos por ela seria absurdo.
DISCOVERY_TIMEOUT_SECONDS: int = 15
DOWNLOAD_MAX_ATTEMPTS: int = 3
RETRY_BACKOFF_BASE_SECONDS: float = 2.0

# Abaixo disso o erro HTTP é do cliente - arquivo inexistente, requisição malformada - e
# repetir a mesma requisição produz o mesmo resultado, só que mais tarde.
SERVER_ERROR_STATUS: int = 500
OK_STATUS: int = 200

# O CloudFront da TLC devolve 403, e não 404, para objeto que não existe: o bucket nega a
# listagem antes de chegar a dizer que a chave está ausente. Para a sondagem os dois
# significam a mesma coisa - o mês ainda não saiu.
MISSING_OBJECT_STATUSES: frozenset[int] = frozenset({403, 404})


def fetch_parquet(url: str) -> pd.DataFrame:
    """Download one monthly parquet with timeout and retries, then parse it in memory.

    O `pd.read_parquet(url)` anterior delegava o download ao pandas, sem timeout e sem
    retry - e o bucket da TLC derruba conexões com frequência suficiente para ter falhado
    duas de três execuções num único dia. Um retreino agendado precisa sobreviver a falha
    transitória sem um humano por perto.
    """
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            # A URL é montada a partir de constantes do módulo, não de entrada externa.
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
                payload = bytes(response.read())
        except urllib.error.HTTPError as error:
            if error.code < SERVER_ERROR_STATUS:
                raise
            last_error = error
        except OSError as error:
            # URLError, connection reset e timeout de socket são todos OSError - e todos
            # transitórios do ponto de vista de quem baixa um arquivo público.
            last_error = error
        else:
            # Erro de parse não é retentado de propósito: um parquet corrompido no bucket
            # não melhora na segunda tentativa, e o chamador o transforma em erro de domínio.
            return pd.read_parquet(io.BytesIO(payload))

        if attempt < DOWNLOAD_MAX_ATTEMPTS:
            delay = RETRY_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
            logger.warning(
                "download_retry",
                extra={
                    "url": url,
                    "attempt": attempt,
                    "max_attempts": DOWNLOAD_MAX_ATTEMPTS,
                    "delay_seconds": delay,
                    "error": str(last_error),
                },
            )
            time.sleep(delay)

    raise DataExtractionError(
        f"Falha ao baixar {url} após {DOWNLOAD_MAX_ATTEMPTS} tentativas: {last_error}."
    ) from last_error


def dataset_url(taxi_type: str, month: YearMonth) -> str:
    """Compose the public URL of one monthly TLC dataset."""
    return f"{NYC_TLC_BASE_URL}{taxi_type}_tripdata_{month}.parquet"


def month_is_published(taxi_type: str, month: YearMonth) -> bool:
    """Ask the TLC whether one monthly dataset already exists, without downloading it.

    Precisa do mesmo retry do download, e por um motivo mais forte: aqui uma conexão
    derrubada seria lida como "o mês não saiu", e o ciclo pararia de enxergar dado que
    existe. Falha transitória tem que se distinguir de ausência real.
    """
    # A URL é montada a partir de constantes do módulo, não de entrada externa.
    request = urllib.request.Request(dataset_url(taxi_type, month), method="HEAD")  # noqa: S310
    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=DISCOVERY_TIMEOUT_SECONDS) as response:  # noqa: S310
                return bool(OK_STATUS <= response.status < SERVER_ERROR_STATUS)
        except urllib.error.HTTPError as error:
            if error.code in MISSING_OBJECT_STATUSES:
                return False
            if error.code < SERVER_ERROR_STATUS:
                raise
            last_error = error
        except OSError as error:
            last_error = error

        if attempt < DOWNLOAD_MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))

    raise DataExtractionError(
        f"Não foi possível verificar se {month} existe para '{taxi_type}' após "
        f"{DOWNLOAD_MAX_ATTEMPTS} tentativas: {last_error}."
    ) from last_error


def discover_latest_published_month(taxi_type: str, today: date | None = None) -> YearMonth:
    """Find the newest month the TLC has actually published, by asking it.

    O atraso de publicação da TLC não é constante: em 2026-09 o mês mais recente disponível
    era 2026-05, quatro meses atrás, contra os dois que `PUBLICATION_LAG_MONTHS` presumia.
    Uma constante errada faz o ciclo autônomo pedir um arquivo inexistente e morrer antes de
    avaliar qualquer gatilho - foi o que aconteceu. Perguntar custa uma requisição HEAD por
    mês tentado e não envelhece.
    """
    if taxi_type not in VALID_TAXI_TYPES:
        raise InvalidTaxiTypeError(f"taxi_type inválido: '{taxi_type}'. Use 'yellow' ou 'green'.")

    reference = today if today is not None else today_utc()
    # O mês corrente nunca está publicado; a busca começa no anterior.
    candidate = YearMonth(reference.year, reference.month).shifted(-1)

    for _ in range(MAX_DISCOVERY_MONTHS):
        if month_is_published(taxi_type, candidate):
            logger.info(
                "latest_published_month_discovered",
                extra={"taxi_type": taxi_type, "month": str(candidate)},
            )
            return candidate
        candidate = candidate.shifted(-1)

    raise DataExtractionError(
        f"Nenhum mês publicado encontrado para '{taxi_type}' nos últimos "
        f"{MAX_DISCOVERY_MONTHS} meses a partir de {reference}. "
        f"Verifique a conectividade ou se a TLC mudou o endereço de publicação."
    )


def extract_trip_data(year: int, month: int, taxi_type: str) -> pd.DataFrame:
    """Download monthly trip parquet data from NYC TLC and normalize datetime columns."""
    if taxi_type not in VALID_TAXI_TYPES:
        raise InvalidTaxiTypeError(f"taxi_type inválido: '{taxi_type}'. Use 'yellow' ou 'green'.")

    validate_month_is_available(year, month, today_utc())

    url = dataset_url(taxi_type, YearMonth(year, month))

    try:
        df = fetch_parquet(url)
    except DataExtractionError:
        raise
    except Exception as exc:
        raise DataExtractionError(
            f"Falha ao baixar dataset TLC de {url}: {exc}. "
            f"O arquivo pode ainda não ter sido publicado para {year}-{month:02d}."
        ) from exc

    df = df.rename(columns=DATETIME_COLUMNS_MAP[taxi_type])
    df["taxi_type"] = taxi_type

    return df
