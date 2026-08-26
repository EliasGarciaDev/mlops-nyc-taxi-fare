'use strict';

// ════════════════════════════════════════════════════════════════════════════
// ZONAS DE TÁXI DA NYC TLC
//
// O índice é gerado por scripts/fetch_taxi_zones.py a partir do contorno oficial.
// As zonas cobrem exatamente o território atendido: fora delas não existe corrida,
// e é essa a fronteira que o mapa impõe.
// ════════════════════════════════════════════════════════════════════════════

const TAXI_ZONES_URL = 'data/taxi_zones.json';
const TRIP_DISTANCES_URL = 'data/trip_distances.json';

const AIRPORT_ZONE_IDS = new Set([1, 132, 138]);
const MANHATTAN_BOROUGH = 'Manhattan';

// Newark fica em Nova Jersey. A TLC classifica o aeroporto como zona porque um táxi de Nova
// York leva passageiro até lá, com sobretaxa própria - mas não pode pegar passageiro lá.
const OUT_OF_STATE_BOROUGH = 'EWR';

// Hail Exclusionary Zone: onde o boro taxi não pode aceitar street hail - Manhattan ao sul
// da W 110th St e da E 96th St. Acima dessa linha (Harlem, Washington Heights, Inwood) o
// embarque é legal, e bloqueá-lo negava serviço justamente nos bairros que mais dependem
// do boro taxi.
//
// O conjunto foi derivado do comportamento real, não transcrito: sobre 8,5 milhões de
// embarques de 2024-01 a 2024-03, estas zonas concentram menos de 0,25% de green apesar do
// movimento de yellow, enquanto as 11 zonas de Manhattan fora da lista ficam acima de 5%.
// O vão entre os dois grupos é de vinte vezes.
const HAIL_EXCLUSIONARY_ZONE_IDS = new Set([
  4, 12, 13, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 140, 141,
  142, 143, 144, 148, 151, 158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230,
  231, 232, 233, 234, 236, 237, 238, 239, 246, 249, 261, 262, 263,
]);

function isPointInRing(lng, lat, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if ((yi > lat) !== (yj > lat) && lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

// O primeiro anel é o contorno externo; os seguintes são recortes internos. Um ponto num
// recorte está fora da zona, mesmo estando dentro do contorno - é assim que ilhas e enclaves
// não são engolidos pela zona vizinha.
function isPointInPolygon(lng, lat, rings) {
  if (!rings.length || !isPointInRing(lng, lat, rings[0])) return false;
  for (let index = 1; index < rings.length; index++) {
    if (isPointInRing(lng, lat, rings[index])) return false;
  }
  return true;
}

// Simplificar polígonos vizinhos de forma independente rompe a topologia: a borda comum a
// duas zonas deixa de ser idêntica nas duas e sobram costuras de poucos metros onde ponto
// nenhum cai. Em vez de recusar o passageiro por causa disso, um ponto sem zona exata é
// atribuído à zona mais próxima dentro desta distância. O valor repara artefato de geometria
// e imprecisão de linha de costa - não estende a área atendida, porque qualquer travessia
// de água em Nova York é uma ordem de grandeza maior.
const SNAP_TOLERANCE_METERS = 100;

// Razão entre distância rodada e linha reta, medida sobre 2,8 milhões de corridas de 2024
// comparando a haversine entre centroides de zona com o que o taxímetro registrou. Manhattan
// é menor porque a malha em grade aproxima melhor a reta que uma travessia de ponte.
const MANHATTAN_DETOUR = 1.2040;
const DEFAULT_DETOUR = 1.4976;

const METERS_PER_DEGREE_LATITUDE = 111_320;

function metersPerDegreeLongitude(latitude) {
  return METERS_PER_DEGREE_LATITUDE * Math.cos((latitude * Math.PI) / 180);
}

function distanceToSegmentMeters(lng, lat, start, end) {
  const scaleX = metersPerDegreeLongitude(lat);
  const px = (lng - start[0]) * scaleX;
  const py = (lat - start[1]) * METERS_PER_DEGREE_LATITUDE;
  const vx = (end[0] - start[0]) * scaleX;
  const vy = (end[1] - start[1]) * METERS_PER_DEGREE_LATITUDE;

  const lengthSquared = vx * vx + vy * vy;
  const projection = lengthSquared === 0 ? 0 : Math.min(1, Math.max(0, (px * vx + py * vy) / lengthSquared));
  return Math.hypot(px - projection * vx, py - projection * vy);
}

function distanceToZoneMeters(zone, lng, lat) {
  let shortest = Infinity;
  for (const polygon of zone.polygons) {
    for (const ring of polygon) {
      for (let index = 1; index < ring.length; index++) {
        const distance = distanceToSegmentMeters(lng, lat, ring[index - 1], ring[index]);
        if (distance < shortest) shortest = distance;
      }
    }
  }
  return shortest;
}

function isInsideBbox(lng, lat, bbox) {
  const [minLng, minLat, maxLng, maxLat] = bbox;
  return lng >= minLng && lng <= maxLng && lat >= minLat && lat <= maxLat;
}

// O retângulo envolvente descarta a esmagadora maioria das 263 zonas antes de qualquer
// teste de polígono, o que mantém a resolução barata o bastante para rodar a cada quadro
// enquanto o marcador é arrastado.
function findZoneExact(zones, lng, lat) {
  for (const zone of zones) {
    if (!isInsideBbox(lng, lat, zone.bbox)) continue;
    for (const polygon of zone.polygons) {
      if (isPointInPolygon(lng, lat, polygon)) return zone;
    }
  }
  return null;
}

function isNearBbox(lng, lat, bbox, meters) {
  const padLat = meters / METERS_PER_DEGREE_LATITUDE;
  const padLng = meters / metersPerDegreeLongitude(lat);
  return isInsideBbox(lng, lat, [
    bbox[0] - padLng, bbox[1] - padLat, bbox[2] + padLng, bbox[3] + padLat,
  ]);
}

function nearestZone(zones, lng, lat, toleranceMeters) {
  let closest = null;
  let shortest = toleranceMeters;
  for (const zone of zones) {
    if (!isNearBbox(lng, lat, zone.bbox, toleranceMeters)) continue;
    const distance = distanceToZoneMeters(zone, lng, lat);
    if (distance <= shortest) {
      shortest = distance;
      closest = zone;
    }
  }
  return closest;
}

function findZone(zones, lng, lat, toleranceMeters = SNAP_TOLERANCE_METERS) {
  return findZoneExact(zones, lng, lat) ?? nearestZone(zones, lng, lat, toleranceMeters);
}

function isAirportZone(zone) {
  return zone !== null && AIRPORT_ZONE_IDS.has(zone.id);
}

function isManhattanZone(zone) {
  return zone !== null && zone.borough === MANHATTAN_BOROUGH;
}

function isPickupAllowed(zone) {
  return zone !== null && zone.borough !== OUT_OF_STATE_BOROUGH;
}

function isDropoffAllowed(zone) {
  return zone !== null;
}

// O boro taxi é barrado apenas na Hail Exclusionary Zone, não em Manhattan inteira. Fora
// dela valem as mesmas restrições de qualquer frota - Newark segue fechado para embarque.
function isGreenPickupAllowed(zone) {
  return isPickupAllowed(zone) && !HAIL_EXCLUSIONARY_ZONE_IDS.has(zone.id);
}

// O taxímetro cobra pela distância rodada; o mapa só sabe a linha reta entre os marcadores.
// Enviar a linha reta ao modelo custava US$ 4,26 de viés - ele foi treinado com a distância
// do taxímetro e interpretava a diferença como uma corrida mais curta do que a real.
//
// Como as zonas já estão resolvidas, a melhor estimativa não é um fator e sim a distância que
// as corridas daquele par de zonas de fato percorreram. O fator entra só quando o par é raro
// demais para ter mediana confiável.
function roadDistance(table, originZone, destZone, straightMiles) {
  if (table && originZone && destZone) {
    const median = table.pairs[`${originZone.id}-${destZone.id}`];
    if (median !== undefined) return median;
  }

  const bothInManhattan =
    isManhattanZone(originZone ?? null) && isManhattanZone(destZone ?? null);
  const detour = bothInManhattan
    ? (table?.manhattan_detour ?? MANHATTAN_DETOUR)
    : (table?.default_detour ?? DEFAULT_DETOUR);
  return straightMiles * detour;
}

function toLeafletOutline(zones) {
  return {
    type: 'FeatureCollection',
    features: zones.map((zone) => ({
      type: 'Feature',
      properties: { id: zone.id, zone: zone.zone, borough: zone.borough },
      geometry: { type: 'MultiPolygon', coordinates: zone.polygons },
    })),
  };
}

const taxiZones = {
  loaded: false,
  zones: [],
  bbox: null,
  distances: null,

  async load(url = TAXI_ZONES_URL) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Contornos das zonas indisponíveis: HTTP ${response.status}`);
    }
    const index = await response.json();
    if (!Array.isArray(index.zones) || index.zones.length === 0) {
      throw new Error('Índice de zonas vazio.');
    }
    this.zones = index.zones;
    this.bbox = index.bbox;
    this.loaded = true;

 // A tabela de distâncias é opcional: sem ela a estimativa cai no fator de área, que é
 // pior mas utilizável. Falhar aqui deixaria o app sem mapa por causa de um refinamento.
    try {
      const distances = await fetch(TRIP_DISTANCES_URL);
      if (distances.ok) this.distances = await distances.json();
    } catch {
      this.distances = null;
    }
    return index;
  },

  at(latlng) {
    return this.loaded ? findZone(this.zones, latlng.lng, latlng.lat) : null;
  },

  contains(latlng) {
    return this.at(latlng) !== null;
  },

 // Busca por identificador, e não por coordenada: quem já tem o LocationID resolvido não
 // precisa refazer o teste geométrico para descobrir a região.
  byId(id) {
    return this.zones.find((zone) => zone.id === id) ?? null;
  },

 // O embarque é mais restrito que o desembarque, e o mapa precisa refletir isso enquanto o
 // marcador é arrastado, não só quando a predição é pedida.
  allows(latlng, role) {
    const zone = this.at(latlng);
    return role === 'origin' ? isPickupAllowed(zone) : isDropoffAllowed(zone);
  },
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    DEFAULT_DETOUR,
    HAIL_EXCLUSIONARY_ZONE_IDS,
    MANHATTAN_DETOUR,
    SNAP_TOLERANCE_METERS,
    distanceToSegmentMeters,
    distanceToZoneMeters,
    findZone,
    findZoneExact,
    isAirportZone,
    isDropoffAllowed,
    isGreenPickupAllowed,
    isInsideBbox,
    isManhattanZone,
    isPickupAllowed,
    isPointInPolygon,
    isPointInRing,
    nearestZone,
    roadDistance,
    taxiZones,
    toLeafletOutline,
  };
}
