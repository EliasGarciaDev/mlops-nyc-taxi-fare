'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  HAIL_EXCLUSIONARY_ZONE_IDS,
  SNAP_TOLERANCE_METERS,
  distanceToSegmentMeters,
  distanceToZoneMeters,
  findZone,
  findZoneExact,
  isAirportZone,
  isDropoffAllowed,
  isGreenPickupAllowed,
  roadDistance,
  isManhattanZone,
  isPickupAllowed,
  isPointInPolygon,
  isPointInRing,
  nearestZone,
  toLeafletOutline,
} = require('../../src/web/zones.js');

// ---------------------------------------------------------------------------
// Recortes de teste: um quadrado com um recorte interno e um vizinho separado
// ---------------------------------------------------------------------------

const SQUARE = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]];
const HOLE = [[0.4, 0.4], [0.4, 0.6], [0.6, 0.6], [0.6, 0.4], [0.4, 0.4]];
const NEIGHBOUR = [[2, 2], [2, 3], [3, 3], [3, 2], [2, 2]];

const ZONES = [
  { id: 161, zone: 'Midtown Center', borough: 'Manhattan', bbox: [0, 0, 1, 1], polygons: [[SQUARE, HOLE]] },
  { id: 132, zone: 'JFK Airport', borough: 'Queens', bbox: [2, 2, 3, 3], polygons: [[NEIGHBOUR]] },
];

// ---------------------------------------------------------------------------
// 1. Containment em um anel
// ---------------------------------------------------------------------------

test('ponto interior está dentro do anel', () => {
  assert.equal(isPointInRing(0.5, 0.5, SQUARE), true);
});

test('ponto exterior está fora do anel', () => {
  assert.equal(isPointInRing(1.5, 0.5, SQUARE), false);
});

// ---------------------------------------------------------------------------
// 2. Recortes internos
// ---------------------------------------------------------------------------

test('ponto no recorte interno não pertence à zona', () => {
  assert.equal(isPointInPolygon(0.5, 0.5, [SQUARE, HOLE]), false);
});

test('ponto fora do recorte mas dentro do contorno pertence à zona', () => {
  assert.equal(isPointInPolygon(0.1, 0.1, [SQUARE, HOLE]), true);
});

test('polígono sem anéis não contém nada', () => {
  assert.equal(isPointInPolygon(0.5, 0.5, []), false);
});

// ---------------------------------------------------------------------------
// 3. Resolução da zona
// ---------------------------------------------------------------------------

test('resolve a zona que contém o ponto', () => {
  assert.equal(findZone(ZONES, 0.1, 0.1).id, 161);
});

test('resolve a zona vizinha correta', () => {
  assert.equal(findZone(ZONES, 2.5, 2.5).id, 132);
});

test('ponto fora de toda zona não resolve', () => {
  assert.equal(findZone(ZONES, 10, 10), null);
});

test('ponto no recorte interno não resolve para a zona que o contém', () => {
  assert.equal(findZone(ZONES, 0.5, 0.5), null);
});

test('ponto dentro do retângulo envolvente mas fora do polígono não resolve', () => {
  const triangle = { id: 7, zone: 'T', borough: 'Bronx', bbox: [0, 0, 1, 1], polygons: [[[[0, 0], [1, 0], [0, 1], [0, 0]]]] };
  assert.equal(findZone([triangle], 0.9, 0.9), null);
});

// ---------------------------------------------------------------------------
// 4. Reparo das costuras deixadas pela simplificação
// ---------------------------------------------------------------------------

// Duas zonas vizinhas cuja borda comum ficou 0.0002° (~17 m) diferente após simplificar.
const WEST = { id: 10, zone: 'Oeste', borough: 'Bronx', bbox: [0, 0, 1, 1], polygons: [[[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]] };
const EAST = { id: 11, zone: 'Leste', borough: 'Bronx', bbox: [1.0002, 0, 2, 1], polygons: [[[[1.0002, 0], [1.0002, 1], [2, 1], [2, 0], [1.0002, 0]]]] };
const SEAM = [WEST, EAST];

test('distância a um segmento é medida em metros', () => {
  // Um centésimo de grau de latitude são cerca de 1113 m.
  const distance = distanceToSegmentMeters(0, 0.01, [0, 0], [1, 0]);
  assert.ok(Math.abs(distance - 1113) < 5, `esperava ~1113 m, veio ${distance}`);
});

test('ponto sobre o segmento tem distância zero', () => {
  assert.equal(distanceToSegmentMeters(0.5, 0, [0, 0], [1, 0]), 0);
});

test('distância à zona considera todos os anéis', () => {
  assert.ok(distanceToZoneMeters(WEST, 1.001, 0.5) > 0);
});

test('ponto na costura entre duas zonas não fica sem zona', () => {
  assert.equal(findZoneExact(SEAM, 1.0001, 0.5), null);
  assert.notEqual(findZone(SEAM, 1.0001, 0.5), null);
});

test('ponto claramente fora continua sem zona', () => {
  assert.equal(findZone(SEAM, 40, 0.5), null);
});

test('a tolerância não estende a área além do previsto', () => {
  // Meio grau de longitude é muito maior que a tolerância de 100 m.
  assert.equal(nearestZone(SEAM, 2.5, 0.5, SNAP_TOLERANCE_METERS), null);
});

test('quando duas zonas estão perto, vence a mais próxima', () => {
  const zone = nearestZone(SEAM, 1.00019, 0.5, SNAP_TOLERANCE_METERS);
  assert.equal(zone.id, 11);
});

test('a tolerância é modesta o bastante para não cruzar água', () => {
  // A travessia mais estreita do East River passa de 600 m.
  assert.ok(SNAP_TOLERANCE_METERS < 600);
});

// ---------------------------------------------------------------------------
// 5. Classificações usadas pelas regras de negócio
// ---------------------------------------------------------------------------

test('zonas de aeroporto são reconhecidas', () => {
  assert.equal(isAirportZone(ZONES[1]), true);
  assert.equal(isAirportZone(ZONES[0]), false);
});

test('zona nula nunca é aeroporto nem Manhattan', () => {
  assert.equal(isAirportZone(null), false);
  assert.equal(isManhattanZone(null), false);
});

test('zonas de Manhattan são reconhecidas pelo borough', () => {
  assert.equal(isManhattanZone(ZONES[0]), true);
  assert.equal(isManhattanZone(ZONES[1]), false);
});

// ---------------------------------------------------------------------------
// 6. Newark só recebe desembarque
// ---------------------------------------------------------------------------

const NEWARK = { id: 1, zone: 'Newark Airport', borough: 'EWR', bbox: [4, 4, 5, 5], polygons: [[NEIGHBOUR]] };

test('embarque em Newark é recusado', () => {
  assert.equal(isPickupAllowed(NEWARK), false);
});

test('desembarque em Newark é permitido', () => {
  assert.equal(isDropoffAllowed(NEWARK), true);
});

test('embarque numa zona de Nova York é permitido', () => {
  assert.equal(isPickupAllowed(ZONES[0]), true);
});

test('fora de qualquer zona nada é permitido', () => {
  assert.equal(isPickupAllowed(null), false);
  assert.equal(isDropoffAllowed(null), false);
});

// ---------------------------------------------------------------------------
// 7. Conversão para desenho no mapa
// ---------------------------------------------------------------------------

test('o contorno vira uma FeatureCollection com uma feature por zona', () => {
  const outline = toLeafletOutline(ZONES);
  assert.equal(outline.type, 'FeatureCollection');
  assert.equal(outline.features.length, 2);
  assert.equal(outline.features[0].geometry.type, 'MultiPolygon');
  assert.equal(outline.features[0].properties.id, 161);
});

// ---------------------------------------------------------------------------
// Regra do Green Cab: a Hail Exclusionary Zone, não Manhattan inteira 
// ---------------------------------------------------------------------------

test('green cab é barrado nas zonas da Hail Exclusionary Zone', () => {
  // 161 Midtown Center, 234 Union Sq, 237 Upper East Side South: todas ao sul da linha.
  for (const id of [161, 234, 237, 13, 230]) {
    assert.equal(isGreenPickupAllowed({ id, borough: 'Manhattan' }), false);
  }
});

test('green cab embarca em Manhattan ao norte da W 110th / E 96th', () => {
  // Harlem, Washington Heights e Inwood são legais para street hail de boro taxi - e é
  // justamente onde o bloqueio anterior, de Manhattan inteira, negava serviço.
  for (const id of [41, 42, 74, 75, 116, 152, 166, 243, 244]) {
    assert.equal(isGreenPickupAllowed({ id, borough: 'Manhattan' }), true);
  }
});

test('green cab embarca livremente fora de Manhattan', () => {
  assert.equal(isGreenPickupAllowed({ id: 7, borough: 'Queens' }), true);
  assert.equal(isGreenPickupAllowed({ id: 33, borough: 'Brooklyn' }), true);
  assert.equal(isGreenPickupAllowed({ id: 3, borough: 'Bronx' }), true);
});

test('zona nula não autoriza embarque de green', () => {
  assert.equal(isGreenPickupAllowed(null), false);
});

test('a Hail Exclusionary Zone tem só zonas de Manhattan', () => {
  // Derivada do comportamento real: sobre 8,5 milhões de embarques de 2024-01 a 03, estas
  // zonas concentram menos de 0,25% de green apesar do movimento de yellow.
  const fs = require('node:fs');
  const path = require('node:path');
  const index = JSON.parse(
    fs.readFileSync(
      path.join(__dirname, '..', '..', 'src', 'web', 'data', 'taxi_zones.json'),
      'utf8'
    )
  );
  const manhattan = new Set(
    index.zones.filter((z) => z.borough === 'Manhattan').map((z) => z.id)
  );
  for (const id of HAIL_EXCLUSIONARY_ZONE_IDS) {
    assert.ok(manhattan.has(id), `zona ${id} não é de Manhattan`);
  }
});

test('o Newark segue barrado para embarque, independente da frota', () => {
  assert.equal(isGreenPickupAllowed({ id: 1, borough: 'EWR' }), false);
  assert.equal(isPickupAllowed({ id: 1, borough: 'EWR' }), false);
});

// ---------------------------------------------------------------------------
// Distância rodada: o taxímetro cobra pela rota, não pela linha reta 
// ---------------------------------------------------------------------------

test('usa a mediana histórica quando o par de zonas é conhecido', () => {
  const table = { pairs: { '161-237': 4.2 }, manhattan_detour: 1.2, default_detour: 1.5 };
  const origin = { id: 161, borough: 'Manhattan' };
  const dest = { id: 237, borough: 'Manhattan' };
  assert.equal(roadDistance(table, origin, dest, 3.0), 4.2);
});

test('cai no fator de Manhattan quando o par não está na tabela', () => {
  const table = { pairs: {}, manhattan_detour: 1.2, default_detour: 1.5 };
  const inside = { id: 161, borough: 'Manhattan' };
  // Tolerância de float: 3,0 x 1,2 não é exatamente 3,6 em ponto flutuante.
  const value = roadDistance(table, inside, { id: 234, borough: 'Manhattan' }, 3.0);
  assert.ok(Math.abs(value - 3.6) < 1e-9);
});

test('usa o fator geral quando alguma ponta está fora de Manhattan', () => {
  // Travessias de ponte desviam muito mais que a malha em grade da ilha.
  const table = { pairs: {}, manhattan_detour: 1.2, default_detour: 1.5 };
  const manhattan = { id: 161, borough: 'Manhattan' };
  const queens = { id: 7, borough: 'Queens' };
  assert.ok(Math.abs(roadDistance(table, manhattan, queens, 3.0) - 4.5) < 1e-9);
});

test('sobrevive à ausência da tabela', () => {
  // Se o arquivo não carregar, a estimativa piora mas o app continua utilizável.
  const straight = 3.0;
  const value = roadDistance(null, { id: 161, borough: 'Manhattan' }, { id: 7, borough: 'Queens' }, straight);
  assert.ok(value > straight);
});

test('sobrevive a zona desconhecida', () => {
  const table = { pairs: {}, manhattan_detour: 1.2, default_detour: 1.5 };
  assert.ok(roadDistance(table, null, null, 3.0) > 3.0);
});

test('a direção importa: ida e volta podem ter rotas diferentes', () => {
  const table = { pairs: { '161-237': 4.2, '237-161': 5.1 }, manhattan_detour: 1.2, default_detour: 1.5 };
  const a = { id: 161, borough: 'Manhattan' };
  const b = { id: 237, borough: 'Manhattan' };
  assert.equal(roadDistance(table, a, b, 3.0), 4.2);
  assert.equal(roadDistance(table, b, a, 3.0), 5.1);
});
