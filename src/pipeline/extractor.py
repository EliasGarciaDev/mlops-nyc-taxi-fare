import io
import time
import urllib.error
import urllib.request

import pandas as pd

from src.core.constants import DATETIME_COLUMNS_MAP, NYC_TLC_BASE_URL, VALID_TAXI_TYPES
from src.core.exceptions import DataExtractionError, InvalidTaxiTypeError
from src.core.log import get_logger
from src.core.months import today_utc, validate_month_is_available

logger = get_logger(__name__)

DOWNLOAD_TIMEOUT_SECONDS: int = 120
DOWNLOAD_MAX_ATTEMPTS: int = 3
RETRY_BACKOFF_BASE_SECONDS: float = 2.0

# Abaixo disso o erro HTTP é do cliente - arquivo inexistente, requisição malformada - e
# repetir a mesma requisição produz o mesmo resultado, só que mais tarde.
SERVER_ERROR_STATUS: int = 500


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


def extract_trip_data(year: int, month: int, taxi_type: str) -> pd.DataFrame:
    """Download monthly trip parquet data from NYC TLC and normalize datetime columns."""
    if taxi_type not in VALID_TAXI_TYPES:
        raise InvalidTaxiTypeError(f"taxi_type inválido: '{taxi_type}'. Use 'yellow' ou 'green'.")

    validate_month_is_available(year, month, today_utc())

    url = f"{NYC_TLC_BASE_URL}{taxi_type}_tripdata_{year}-{month:02d}.parquet"

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
