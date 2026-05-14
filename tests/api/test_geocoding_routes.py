from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.geocoding import Place

PLACES = [Place(display_name="Times Square, Manhattan", lat=40.758, lon=-73.9855)]


@pytest.fixture
def client():
    with patch("src.api.routes.load_models"):
        yield TestClient(app)


class TestSearchEndpoint:
    def test_returns_the_places_found(self, client):
        with patch("src.api.geocoding.GeocodingClient.search", return_value=PLACES):
            body = client.get("/geocode/search", params={"q": "times square"}).json()
        assert body["results"][0]["display_name"] == PLACES[0].display_name
        assert body["results"][0]["lat"] == pytest.approx(40.758)
