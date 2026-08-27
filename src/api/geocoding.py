import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass

from src.core.exceptions import GeocodingError
from src.core.log import get_logger

logger = get_logger(__name__)

NOMINATIM_BASE_URL: str = "https://nominatim.openstreetmap.org"

# A política de uso do Nominatim exige identificação da aplicação. No navegador isso era
NOMINATIM_USER_AGENT: str = (
    "nyc-taxi-fare-predictor/1.0 (TCC IFSul; https://github.com/eliasgarcia/nyc-taxi-fare-predictor)"
)

# A mesma política pede no máximo uma requisição por segundo.
DEFAULT_MIN_INTERVAL_SECONDS: float = 1.0

DEFAULT_TIMEOUT_SECONDS: float = 8.0
DEFAULT_CACHE_SIZE: int = 512
DEFAULT_SEARCH_LIMIT: int = 5

# Casas decimais usadas para agrupar consultas reversas. Quatro casas são cerca de onze metros
REVERSE_PRECISION: int = 4

# Recorte da cidade, para que a busca não devolva homônimos de outros estados.
NYC_VIEWBOX: str = "-74.2591,40.9176,-73.7004,40.4774"


@dataclass
class Place:
    """One geocoding result, reduced to what the interface uses."""

    display_name: str
    lat: float
    lon: float


class LruCache[T]:
    """A bounded most-recently-used cache.

    Sem teto, um cache numa API pública é vazamento de memória com nome bonito: cada busca
    inédita de cada turista fica guardada para sempre.
    """

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._entries: OrderedDict[str, T] = OrderedDict()

    def get(self, key: str) -> T | None:
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: str, value: T) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)


class RateLimiter:
    """Spaces outgoing calls so the shared service is not abused.

    O limite é do processo inteiro, não por usuário: quem impõe a política é o Nominatim, e
    para ele o que existe é o nosso endereço. Vários turistas digitando ao mesmo tempo somam
    no mesmo balde.
    """

    def __init__(self, min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS) -> None:
        self._min_interval = min_interval_seconds
        self._last_call: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_call is not None:
            remaining = self._min_interval - (now - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
        self._last_call = now


class GeocodingClient:
    """Server-side proxy to Nominatim, with identification, cache and throttling.

    Existe por três motivos que o cliente não conseguia atender: identificar a aplicação,
    respeitar o limite de uma requisição por segundo, e não repetir buscas - turistas
    procuram os mesmos lugares, e o cache transforma isso numa requisição só.
    """

    def __init__(
        self,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        cache_size: int = DEFAULT_CACHE_SIZE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._limiter = RateLimiter(min_interval_seconds)
        self._timeout = timeout_seconds
        self._searches: LruCache[list[Place]] = LruCache(cache_size)
        self._reverses: LruCache[str] = LruCache(cache_size)

    def _fetch(self, path: str, params: dict[str, str]) -> object:
        url = f"{NOMINATIM_BASE_URL}{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": NOMINATIM_USER_AGENT})  # noqa: S310

        self._limiter.wait()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                payload = response.read()
            return json.loads(payload)
        except (OSError, ValueError) as failure:
            # Rede fora, timeout ou resposta malformada. A falha vira erro de domínio para a
            logger.warning(
                "geocoding_failed", extra={"path": path, "error": str(failure)}
            )
            raise GeocodingError(
                f"Falha ao consultar o Nominatim em {path}: {failure}."
            ) from failure

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Place]:
        """Busca endereços em Nova York correspondentes à consulta textual informada."""
        normalized = " ".join(query.split()).lower()
        if not normalized:
            return []

        key = f"{normalized}:{limit}"
        cached = self._searches.get(key)
        if cached is not None:
            return list(cached)

        payload = self._fetch(
            "/search",
            {
                "q": normalized,
                "format": "json",
                "limit": str(limit),
                "viewbox": NYC_VIEWBOX,
                "bounded": "1",
            },
        )
        places = _places_from(payload)
        self._searches.put(key, places)
        return list(places)

    def reverse(self, lat: float, lon: float) -> str:
        """Obtém a descrição textual de uma coordenada geográfica para exibição ao passageiro."""
        key = f"{round(lat, REVERSE_PRECISION)}:{round(lon, REVERSE_PRECISION)}"
        cached = self._reverses.get(key)
        if cached is not None:
            return cached

        payload = self._fetch(
            "/reverse", {"lat": str(lat), "lon": str(lon), "format": "json"}
        )
        name = str(payload.get("display_name", "")) if isinstance(payload, dict) else ""
        self._reverses.put(key, name)
        return name


def _places_from(payload: object) -> list[Place]:
    """Read the entries that carry usable coordinates, ignoring the rest."""
    if not isinstance(payload, list):
        return []

    places: list[Place] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        try:
            places.append(
                Place(
                    display_name=str(entry["display_name"]),
                    lat=float(entry["lat"]),
                    lon=float(entry["lon"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            # Entrada sem coordenada não serve para posicionar o marcador; descartar uma é
            # melhor que recusar a busca inteira.
            continue
    return places
