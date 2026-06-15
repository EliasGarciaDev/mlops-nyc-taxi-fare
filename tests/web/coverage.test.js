'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { findZone, findZoneExact } = require('../../src/web/zones.js');

const INDEX_PATH = path.join(__dirname, '..', '..', 'src', 'web', 'data', 'taxi_zones.json');

// ---------------------------------------------------------------------------
// Este arquivo cobra a promessa da interface: qualquer ponto de Nova York precisa
// resolver para uma zona, e nenhum ponto de fora da cidade pode resolver.
// ---------------------------------------------------------------------------

const NEIGHBOURHOODS = [
  ['Times Square', 'Manhattan', 40.758, -73.9855],
  ['Harlem', 'Manhattan', 40.809, -73.945],
  ['Inwood', 'Manhattan', 40.8677, -73.9212],
  ['Lower East Side', 'Manhattan', 40.715, -73.9843],
  ['Battery Park', 'Manhattan', 40.7033, -74.017],
  ['Roosevelt Island', 'Manhattan', 40.7614, -73.9505],
  ['Marble Hill', 'Manhattan', 40.876, -73.91],
  ['Governors Island', 'Manhattan', 40.6892, -74.0165],
  ['Astoria', 'Queens', 40.7644, -73.9235],
  ['Flushing', 'Queens', 40.7654, -73.8318],
  ['Jamaica', 'Queens', 40.702, -73.789],
  ['Far Rockaway', 'Queens', 40.6045, -73.755],
  ['Broad Channel', 'Queens', 40.606, -73.82],
  ['Glen Oaks', 'Queens', 40.747, -73.711],
  ['Breezy Point', 'Queens', 40.557, -73.925],
  ['Long Island City', 'Queens', 40.7447, -73.9485],
  ['Williamsburg', 'Brooklyn', 40.7081, -73.9571],
  ['Coney Island', 'Brooklyn', 40.5755, -73.9707],
  ['Bay Ridge', 'Brooklyn', 40.626, -74.03],
  ['Brownsville', 'Brooklyn', 40.663, -73.911],
  ['Bushwick', 'Brooklyn', 40.6944, -73.9213],
  ['Sheepshead Bay', 'Brooklyn', 40.586, -73.944],
  ['Riverdale', 'Bronx', 40.89, -73.91],
  ['Co-op City', 'Bronx', 40.874, -73.829],
  ['City Island', 'Bronx', 40.846, -73.787],
  ['Mott Haven', 'Bronx', 40.809, -73.9229],
  ['Throgs Neck', 'Bronx', 40.818, -73.82],
  ['Pelham Bay', 'Bronx', 40.85, -73.83],
  ['St. George', 'Staten Island', 40.6437, -74.074],
  ['Tottenville', 'Staten Island', 40.509, -74.24],
  ['Great Kills', 'Staten Island', 40.554, -74.15],
  ['New Springville', 'Staten Island', 40.59, -74.165],
  ['Todt Hill', 'Staten Island', 40.6, -74.1],
  ['JFK', 'Queens', 40.6413, -73.7781],
  ['LaGuardia', 'Queens', 40.7769, -73.874],
];

const OUTSIDE = [
  ['meio do rio Hudson', 40.73, -74.02],
  ['Hoboken, NJ', 40.744, -74.0324],
  ['Yonkers, NY', 40.9312, -73.8988],
  ['centro de Newark, NJ', 40.7357, -74.1724],
  ['oceano ao sul da cidade', 40.45, -73.95],
];

function loadZones() {
  if (!fs.existsSync(INDEX_PATH)) return null;
  return JSON.parse(fs.readFileSync(INDEX_PATH, 'utf8')).zones;
}

const zones = loadZones();
const missing = 'Rode `python scripts/fetch_taxi_zones.py` para gerar os contornos.';

test('todo bairro de Nova York resolve para uma zona', (t) => {
  if (zones === null) return t.skip(missing);

  const failures = NEIGHBOURHOODS.filter(([, , lat, lng]) => findZone(zones, lng, lat) === null);
  assert.deepEqual(failures.map(([name]) => name), []);
});

test('todos os cinco boroughs estão representados', (t) => {
  if (zones === null) return t.skip(missing);

  const boroughs = new Set(
    NEIGHBOURHOODS.map(([, , lat, lng]) => findZone(zones, lng, lat).borough),
  );
  for (const expected of ['Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island']) {
    assert.ok(boroughs.has(expected), `${expected} não apareceu`);
  }
});

test('nada de fora da cidade entra na área atendida', (t) => {
  if (zones === null) return t.skip(missing);

  const leaks = OUTSIDE.filter(([, lat, lng]) => findZone(zones, lng, lat) !== null);
  assert.deepEqual(leaks.map(([name]) => name), []);
});

test('cada LocationID aparece uma única vez', (t) => {
  if (zones === null) return t.skip(missing);

  const seen = new Set();
  const duplicated = zones.filter((zone) => (seen.has(zone.id) ? true : (seen.add(zone.id), false)));
  assert.deepEqual(duplicated.map((zone) => zone.id), []);
});

test('a aproximação recupera pontos que a geometria exata perde', (t) => {
  if (zones === null) return t.skip(missing);

  // Costuras conhecidas, medidas sobre o índice gerado: vazios de poucos metros cercados
  // por zona nos quatro lados, deixados pela simplificação de polígonos vizinhos.
  const seams = [
    [40.7003, -73.9714],
    [40.7018, -73.9924],
    [40.7045, -74.0143],
    [40.7057, -73.9507],
    [40.7075, -74.0041],
  ];

  for (const [lat, lng] of seams) {
    assert.equal(findZoneExact(zones, lng, lat), null, `${lat},${lng} deixou de ser costura`);
    assert.notEqual(findZone(zones, lng, lat), null, `${lat},${lng} continua sem zona`);
  }
});
