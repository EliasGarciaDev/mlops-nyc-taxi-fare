import math

# Tolerância de simplificação em graus. Cerca de onze metros na latitude de Nova York - abaixo
DEFAULT_SIMPLIFY_TOLERANCE: float = 0.0001

Point = tuple[float, float]
Ring = list[Point]


def perpendicular_distance(point: Point, start: Point, end: Point) -> float:
    """Distance from a point to the segment defined by start and end."""
    if start == end:
        return math.hypot(point[0] - start[0], point[1] - start[1])

    dx, dy = end[0] - start[0], end[1] - start[1]
    numerator = abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0])
    return numerator / math.hypot(dx, dy)


def simplify_ring(ring: Ring, tolerance: float) -> Ring:
    """Reduce a ring with Ramer-Douglas-Peucker, keeping it closed and valid.

    Um anel precisa de pelo menos quatro pontos para continuar sendo um polígono fechado.
    Quando a simplificação levaria abaixo disso, o anel original é preservado - reduzir mais
    trocaria peso por um buraco no mapa.
    """
    minimum_ring_size = 4
    if len(ring) <= minimum_ring_size:
        return list(ring)

    simplified = _douglas_peucker(list(ring), tolerance)
    if len(simplified) < minimum_ring_size:
        return list(ring)

    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    return simplified


def _douglas_peucker(points: Ring, tolerance: float) -> Ring:
    if len(points) < 3:  # noqa: PLR2004
        return points

    start, end = points[0], points[-1]
    distances = [perpendicular_distance(point, start, end) for point in points[1:-1]]
    if not distances:
        return points

    farthest = max(range(len(distances)), key=distances.__getitem__)
    if distances[farthest] <= tolerance:
        return [start, end]

    pivot = farthest + 1
    left = _douglas_peucker(points[: pivot + 1], tolerance)
    right = _douglas_peucker(points[pivot:], tolerance)
    return left[:-1] + right


def bounding_box(rings: list[Ring]) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) covering every ring."""
    longitudes = [point[0] for ring in rings for point in ring]
    latitudes = [point[1] for ring in rings for point in ring]
    if not longitudes:
        raise ValueError("Não é possível calcular o retângulo envolvente de uma geometria vazia.")
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def point_in_ring(longitude: float, latitude: float, ring: Ring) -> bool:
    """Ray casting containment test for a single closed ring."""
    inside = False
    previous = len(ring) - 1
    for current in range(len(ring)):
        cx, cy = ring[current]
        px, py = ring[previous]
        if (cy > latitude) != (py > latitude) and longitude < (px - cx) * (latitude - cy) / (
            py - cy
        ) + cx:
            inside = not inside
        previous = current
    return inside
