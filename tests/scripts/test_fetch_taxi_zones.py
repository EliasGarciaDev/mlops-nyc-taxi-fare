from scripts.fetch_taxi_zones import build_zone

# ---------------------------------------------------------------------------
# Helpers - recortes no formato publicado pela NYC TLC
# ---------------------------------------------------------------------------

SQUARE = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
HOLE = [[0.4, 0.4], [0.4, 0.6], [0.6, 0.6], [0.6, 0.4], [0.4, 0.4]]


def make_feature(location_id: int = 161, geometry: dict | None = None, **properties) -> dict:
    base = {
        "locationid": str(location_id),
        "zone": "Midtown Center",
        "borough": "Manhattan",
        "shape_area": "0.0007",
        "shape_leng": "0.11",
    }
    base.update(properties)
    return {
        "type": "Feature",
        "properties": base,
        "geometry": geometry or {"type": "MultiPolygon", "coordinates": [[SQUARE]]},
    }


def make_collection(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features or (make_feature(),))}


# ---------------------------------------------------------------------------
# 1. Uma zona vira o registro que o cliente consome
# ---------------------------------------------------------------------------


class TestBuildZone:
    def test_keeps_the_location_id_as_an_integer(self):
        assert build_zone(make_feature(132), tolerance=0.0001)["id"] == 132
