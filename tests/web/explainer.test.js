'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  FLAT_FARE_METER_USD,
  MANHATTAN_ZONE_IDS_FOR_RULES,
  isFlatFareTrip,
  predictLocally,
  PRICING_RULE_FLAT_FARE,
  buildExplanation,
  buildFeatureVector,
  buildFlatFareExplanation,
  marginFor,
  parsePickupComponents,
  splitDisplayTerms,
  toNycWallClock,
} = require('../../src/web/explainer.js');

// ---------------------------------------------------------------------------
// Fábricas - payload como o app.js envia e info como o /model-info responde
// ---------------------------------------------------------------------------

function makePayload(overrides = {}) {
  return Object.assign(
    {
      taxi_type: 'yellow',
      trip_distance: 3.5,
      passenger_count: 1,
      PULocationID: 161,
      DOLocationID: 237,
      RatecodeID: 1,
      pickup_datetime: '2024-06-03T14:30:00', // segunda-feira, 14h30 em Nova York
      trip_duration_minutes: 20.0,
    },
    overrides
  );
}

function makeModelInfo(overrides = {}) {
  return Object.assign(
    {
      taxi_type: 'yellow',
      intercept: 3.41,
      coefficients: {
        trip_distance: 2.76,
        hour_of_day: -0.0005,
        day_of_week: -0.011,
        is_weekend: -0.025,
        is_airport_trip: 0.034,
        is_congestion_zone: 0.0,
        is_rate_jfk: -0.68,
        is_rate_newark: 22.98,
        is_rate_nassau_westchester: 34.21,
        is_rate_negotiated: 41.67,
      },
      rmse: 4.13,
      training_samples: 5384002,
      model_version: 'yellow-test',
    },
    overrides
  );
}

// ---------------------------------------------------------------------------
// 1. O horário é hora de parede de Nova York, lido pelos componentes literais
// ---------------------------------------------------------------------------

test('reads the literal components of the naive timestamp', () => {
  assert.equal(parsePickupComponents('2024-06-03T14:30:00').getUTCHours(), 14);
});

test('ignores a timezone suffix instead of shifting the hour', () => {
  // O campo é hora de parede: deslocar por causa de um sufixo produziria uma hora que a
  // API não usou, que é exatamente o desvio que ocorrigiu.
  assert.equal(parsePickupComponents('2024-06-03T14:30:00Z').getUTCHours(), 14);
  assert.equal(parsePickupComponents('2024-06-03T14:30:00+02:00').getUTCHours(), 14);
});

// ---------------------------------------------------------------------------
// 1b. A conversão para a hora da cidade, que é o que o frontend envia
// ---------------------------------------------------------------------------

test('converts an instant to the New York wall clock', () => {
  // 2024-06-15T18:00Z é horário de verão em Nova York: UTC−4.
  assert.equal(toNycWallClock(new Date('2024-06-15T18:00:00Z')), '2024-06-15T14:00:00');
});

test('follows daylight saving time instead of a fixed offset', () => {
  assert.equal(toNycWallClock(new Date('2024-01-15T18:00:00Z')), '2024-01-15T13:00:00');
});

test('crosses the date boundary backwards, like the city does at night', () => {
  assert.equal(toNycWallClock(new Date('2024-06-16T02:00:00Z')), '2024-06-15T22:00:00');
});

test('produces a timestamp the explainer reads back unchanged', () => {
  // Ida e volta: o que o app envia é o que o painel decompõe. Se as duas pontas
  // divergirem, a explicação descreve um horário que a API não usou.
  const wall = toNycWallClock(new Date('2024-06-15T18:00:00Z'));
  assert.equal(parsePickupComponents(wall).getUTCHours(), 14);
});

// ---------------------------------------------------------------------------
// 2. O vetor reconstruído espelha a derivação de src/api/routes.py
// ---------------------------------------------------------------------------

test('derives the hour from the payload, never from the wall clock', () => {
  const vector = buildFeatureVector(makePayload({ pickup_datetime: '2024-06-03T05:00:00' }));
  assert.equal(vector.hour_of_day, 5);
});

test('converts the weekday to the Python convention, Monday as zero', () => {
  assert.equal(buildFeatureVector(makePayload()).day_of_week, 0); // 2024-06-03: segunda
  const sunday = makePayload({ pickup_datetime: '2024-06-09T12:00:00' });
  assert.equal(buildFeatureVector(sunday).day_of_week, 6);
});

test('marks the weekend exactly at the Saturday boundary', () => {
  const friday = makePayload({ pickup_datetime: '2024-06-07T12:00:00' });
  const saturday = makePayload({ pickup_datetime: '2024-06-08T12:00:00' });
  assert.equal(buildFeatureVector(friday).is_weekend, 0);
  assert.equal(buildFeatureVector(saturday).is_weekend, 1);
});

test('flags an airport trip from either endpoint of the ride', () => {
  assert.equal(buildFeatureVector(makePayload({ PULocationID: 132 })).is_airport_trip, 1);
  assert.equal(buildFeatureVector(makePayload({ DOLocationID: 1 })).is_airport_trip, 1);
  assert.equal(buildFeatureVector(makePayload()).is_airport_trip, 0);
});

test('raises exactly one rate indicator for a non-standard rate code', () => {
  const vector = buildFeatureVector(makePayload({ RatecodeID: 3 }));
  assert.equal(vector.is_rate_newark, 1);
  assert.equal(vector.is_rate_jfk + vector.is_rate_nassau_westchester + vector.is_rate_negotiated, 0);
});

test('keeps every rate indicator at zero for the reference codes', () => {
  for (const code of [1, 6]) {
    const vector = buildFeatureVector(makePayload({ RatecodeID: code }));
    const raised = Object.entries(vector).filter(
      ([name, value]) => name.startsWith('is_rate_') && value !== 0
    );
    assert.deepEqual(raised, []);
  }
});

test('flags the congestion zone from either endpoint, like the API', () => {
  // 161 = Midtown Center está na CRZ; 41 = Central Harlem e 200 = Riverdale, não.
  assert.equal(buildFeatureVector(makePayload({ PULocationID: 161 })).is_congestion_zone, 1);
  assert.equal(buildFeatureVector(makePayload({ PULocationID: 41, DOLocationID: 234 })).is_congestion_zone, 1);
  assert.equal(buildFeatureVector(makePayload({ PULocationID: 41, DOLocationID: 200 })).is_congestion_zone, 0);
});

// ---------------------------------------------------------------------------
// 3. A conta fecha: intercepto + Σ termos = predição do modelo linear
// ---------------------------------------------------------------------------

test('covers every coefficient exactly once', () => {
  const info = makeModelInfo();
  const explanation = buildExplanation(info, makePayload());
  assert.deepEqual(
    explanation.terms.map((term) => term.feature),
    Object.keys(info.coefficients)
  );
});

test('reconstructs the exact dot product of the linear model', () => {
  const info = makeModelInfo();
  const payload = makePayload({ RatecodeID: 2, PULocationID: 132 });
  const vector = buildFeatureVector(payload);

  let expected = info.intercept;
  for (const [feature, coefficient] of Object.entries(info.coefficients)) {
    expected += coefficient * (vector[feature] ?? 0);
  }
  // Tolerância de float, não igualdade estrita: a ordem da soma difere entre o teste
  // e a implementação, e ponto flutuante não é associativo.
  assert.ok(Math.abs(buildExplanation(info, payload).reconstructed - expected) < 1e-9);
});

test('keeps the account closed when the model grows a new feature', () => {
  // Um retreino pode adicionar features que este arquivo não conhece. A conta continua
  // fechando porque a iteração é sobre os coeficientes servidos, não sobre uma lista fixa.
  const info = makeModelInfo();
  info.coefficients.some_future_feature = 9.99;
  const explanation = buildExplanation(info, makePayload());
  const futureTerm = explanation.terms.find((t) => t.feature === 'some_future_feature');
  assert.equal(futureTerm.contribution, 0);
});

// ---------------------------------------------------------------------------
// 4. A divisão em linhas preserva o total na tela
// ---------------------------------------------------------------------------

test('splits negligible terms into a remainder that closes the sum', () => {
  const info = makeModelInfo();
  const payload = makePayload();
  const explanation = buildExplanation(info, payload);
  const fare = explanation.reconstructed;

  const { rows, others } = splitDisplayTerms(explanation, fare);
  const shown = rows.reduce((sum, term) => sum + term.contribution, 0);

  assert.ok(Math.abs(explanation.intercept + shown + others - fare) < 1e-9);
  for (const term of rows) {
    assert.ok(Math.abs(term.contribution) >= 0.005);
  }
});

test('absorbs the gap between client reconstruction and the served fare', () => {
  // O resto é derivado da tarifa exibida, então mesmo uma divergência entre o cliente
  // e o servidor não abre a conta na tela - ela aparece no resto, onde é auditável.
  const explanation = buildExplanation(makeModelInfo(), makePayload());
  const fareFromApi = explanation.reconstructed + 0.5;
  const { rows, others } = splitDisplayTerms(explanation, fareFromApi);
  const shown = rows.reduce((sum, term) => sum + term.contribution, 0);
  assert.ok(Math.abs(explanation.intercept + shown + others - fareFromApi) < 1e-9);
});


// ---------------------------------------------------------------------------
// Tarifa fixa: o painel troca de explicação em vez de somar coeficientes 
// ---------------------------------------------------------------------------

test('splits the flat fare into the regulated meter amount and the extras', () => {
  const { meter, extras } = buildFlatFareExplanation(94.06);
  assert.equal(meter, FLAT_FARE_METER_USD);
  assert.ok(Math.abs(extras - 24.06) < 1e-9);
});

test('the flat fare explanation always adds up to the served fare', () => {
  for (const fare of [88.0, 94.06, 110.5]) {
    const { meter, extras } = buildFlatFareExplanation(fare);
    assert.ok(Math.abs(meter + extras - fare) < 1e-9);
  }
});

test('the regulated amount does not move with the fare', () => {
  // É essa invariância que o painel precisa mostrar: a distância não entra na conta.
  const cheap = buildFlatFareExplanation(88.0);
  const pricey = buildFlatFareExplanation(110.5);
  assert.equal(cheap.meter, pricey.meter);
});

test('the coefficient decomposition would disagree with a flat fare, which is why it is skipped', () => {
  // Guarda o motivo da camada existir: com a regra aplicada, a soma dos coeficientes
  // descreve outro número. Somá-los na tela seria explicar um mecanismo que não agiu.
  const explanation = buildExplanation(makeModelInfo(), makePayload({
    PULocationID: 132,
    DOLocationID: 230,
    RatecodeID: 2,
  }));
  const servedByRule = FLAT_FARE_METER_USD + 24.06;
  assert.ok(Math.abs(explanation.reconstructed - servedByRule) > 1.0);
});

test('names the rule exactly as the API serializes it', () => {
  // O valor vem de PricingRule.JFK_FLAT_FARE em src/ml/fare_rules.py; se um lado renomear,
  // o painel volta silenciosamente a decompor por coeficiente.
  assert.equal(PRICING_RULE_FLAT_FARE, 'jfk_flat_fare');
});


// ---------------------------------------------------------------------------
// Modo estático: a mesma conta e a mesma regra que a API aplicaria
// ---------------------------------------------------------------------------

const STATIC_CONFIG = { flat_fare_amount: 70.0, minimum_total_amount: 4.5, excess: 24.06 };

function makeStaticInfo() {
  return makeModelInfo();
}

test('the flat fare rule fires on zones, in both directions', () => {
  const toJfk = { PULocationID: 230, DOLocationID: 132 };
  const fromJfk = { PULocationID: 132, DOLocationID: 230 };
  assert.equal(isFlatFareTrip(toJfk, STATIC_CONFIG), 94.06);
  assert.equal(isFlatFareTrip(fromJfk, STATIC_CONFIG), 94.06);
});

test('a forged rate code cannot buy the flat fare offline either', () => {
  // O payload nem é consultado quanto ao RatecodeID - a regra olha só as pontas.
  const forged = { PULocationID: 7, DOLocationID: 7, RatecodeID: 2 };
  assert.equal(isFlatFareTrip(forged, STATIC_CONFIG), null);
});

test('without calibration the offline rule abstains, like the API does', () => {
  const uncalibrated = { ...STATIC_CONFIG, excess: null };
  assert.equal(isFlatFareTrip({ PULocationID: 132, DOLocationID: 230 }, uncalibrated), null);
});

test('the offline prediction reuses the explainer reconstruction', () => {
  const payload = makePayload();
  const info = makeStaticInfo();
  const { reconstructed } = buildExplanation(info, payload);
  const local = predictLocally(info, payload, STATIC_CONFIG);
  assert.ok(Math.abs(local.predicted_fare - reconstructed) < 1e-9);
  assert.equal(local.pricing_rule, 'model');
});

test('the offline prediction raises an implausible value to the floor', () => {
  const info = { ...makeStaticInfo(), intercept: -100, coefficients: { trip_distance: 0 } };
  const local = predictLocally(info, makePayload(), STATIC_CONFIG);
  assert.equal(local.predicted_fare, STATIC_CONFIG.minimum_total_amount);
  assert.equal(local.pricing_rule, 'minimum_fare');
});

test('the Manhattan zone list matches the one the server uses', () => {
  // A lista está duplicada em JS porque o modo estático não tem servidor para consultar.
  // Este teste é o que impede as duas de divergirem em silêncio - se alguém alterar
  // MANHATTAN_LOCATION_IDS em src/core/constants.py e esquecer daqui, ele quebra.
  const fs = require('node:fs');
  const constants = fs.readFileSync(
    require('node:path').join(__dirname, '..', '..', 'src', 'core', 'constants.py'),
    'utf8'
  );
  const block = constants.match(/MANHATTAN_LOCATION_IDS: set\[int\] = \{([^}]*)\}/);
  assert.ok(block, 'bloco MANHATTAN_LOCATION_IDS não encontrado em constants.py');

  const fromPython = new Set(
    block[1].split(',').map((piece) => piece.trim()).filter(Boolean).map(Number)
  );
  assert.equal(fromPython.size, MANHATTAN_ZONE_IDS_FOR_RULES.size);
  for (const id of fromPython) {
    assert.ok(MANHATTAN_ZONE_IDS_FOR_RULES.has(id), `zona ${id} ausente no espelho JS`);
  }
});


// ---------------------------------------------------------------------------
// Margem de erro por região 
// ---------------------------------------------------------------------------

const INFO_WITH_REGIONS = {
  rmse: 6.81,
  rmse_by_borough: { Manhattan: 5.65, Queens: 13.17, 'Staten Island': 25.25 },
};

test('uses the measured margin of the dropoff region', () => {
  const { value, borough } = marginFor(INFO_WITH_REGIONS, 'Queens');
  assert.equal(value, 13.17);
  assert.equal(borough, 'Queens');
});

test('the periphery margin is far from the aggregate one', () => {
  // É o motivo do: exibir ±6,81 numa corrida para Staten Island promete uma
  // precisão que o sistema não tem, e o número certo é quase quatro vezes maior.
  assert.ok(marginFor(INFO_WITH_REGIONS, 'Staten Island').value > INFO_WITH_REGIONS.rmse * 3);
});

test('falls back to the aggregate margin for an unmeasured region', () => {
  const { value, borough } = marginFor(INFO_WITH_REGIONS, 'Bronx');
  assert.equal(value, 6.81);
  assert.equal(borough, null);
});

test('falls back when the artifact predates the measurement', () => {
  const legacy = { rmse: 6.81 };
  assert.equal(marginFor(legacy, 'Queens').value, 6.81);
  assert.equal(marginFor(legacy, 'Queens').borough, null);
});

test('falls back when the dropoff zone is unknown', () => {
  assert.equal(marginFor(INFO_WITH_REGIONS, null).value, 6.81);
  assert.equal(marginFor(INFO_WITH_REGIONS, undefined).value, 6.81);
});
