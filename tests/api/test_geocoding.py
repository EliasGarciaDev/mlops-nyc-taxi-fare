import json
from unittest.mock import MagicMock, patch

from src.api.geocoding import (
    NOMINATIM_USER_AGENT,
    GeocodingClient,
)

URLOPEN = "src.api.geocoding.urllib.request.urlopen"
SLEEP = "src.api.geocoding.time.sleep"

SEARCH_PAYLOAD = [
    {"display_name": "Times Square, Manhattan, New York", "lat": "40.758", "lon": "-73.9855"},
    {"display_name": "Times Square Station, Manhattan", "lat": "40.7559", "lon": "-73.9870"},
]
REVERSE_PAYLOAD = {"display_name": "Times Square, Manhattan, New York, USA"}


def response_with(payload) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return response


def make_client(**kwargs) -> GeocodingClient:
    kwargs.setdefault("min_interval_seconds", 0.0)
    return GeocodingClient(**kwargs)


# ---------------------------------------------------------------------------
# 1. O User-Agent que o navegador descartava
# ---------------------------------------------------------------------------


class TestUserAgent:
    def test_identifies_the_application_to_nominatim(self):
        """Na Fetch API o User-Agent é header proibido e some silenciosamente. No servidor
        ele é definido de verdade - que é a razão de existir este proxy."""
        with patch(URLOPEN, return_value=response_with([])) as urlopen:
            make_client().search("times square")
        request = urlopen.call_args[0][0]
        assert request.get_header("User-agent") == NOMINATIM_USER_AGENT

    def test_the_user_agent_names_the_project(self):
        assert "nyc-taxi-fare-predictor" in NOMINATIM_USER_AGENT.lower()
