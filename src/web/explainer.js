'use strict';

// ════════════════════════════════════════════════════════════════════════════
// EXPLAINER - reconstrução do cálculo linear que a API executou
//
// O painel XAI só é honesto se explicar exatamente a conta que produziu o número
// na tela. Este módulo reconstrói o vetor de features do jeito que src/api/routes.py
// o monta a partir do payload - mesmos nomes, mesmas regras, mesma origem de dados -
// e decompõe a predição em intercepto + contribuição de cada coeficiente.
//
// A regra de ouro: iterar sobre TODOS os coeficientes que o /model-info devolve,
// nunca sobre uma lista fixa. Um retreino que mude o contrato de features continua
// fechando a conta sem tocar neste arquivo.
// ════════════════════════════════════════════════════════════════════════════

// Espelhos de src/core/constants.py. Divergência aqui quebra a conta do painel,
// e é por isso que os testes cobram o fechamento exato da soma.
const EXPLAINER_AIRPORT_IDS = new Set([1, 132, 138]);
const EXPLAINER_CRZ_IDS = new Set([
  4, 13, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 144, 148,
  158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234,
  246, 249, 261,
]);
const EXPLAINER_RATECODE_FEATURES = {
  2: 'is_rate_jfk',
  3: 'is_rate_newark',
  4: 'is_rate_nassau_westchester',
  5: 'is_rate_negotiated',
};
const EXPLAINER_WEEKEND_START = 5;
const NYC_TIMEZONE = 'America/New_York';

// Abaixo disso a contribuição arredonda para $0,00 e viraria uma linha vazia;
// esses termos são somados numa linha única de "outros fatores".
const DISPLAY_THRESHOLD_USD = 0.005;

// Converte um instante para a hora de parede de Nova York, que é a convenção do
// pickup_datetime e dos timestamps da TLC. O offset muda com o horário de verão, então
// ele é obtido do fuso e não de uma constante.
function toNycWallClock(date = new Date()) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: NYC_TIMEZONE,
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  }).formatToParts(date);
  const field = (type) => parts.find((part) => part.type === type).value;
 // hourCycle h23 devolve '24' à meia-noite em alguns runtimes; normalizar evita uma data inválida.
  const hour = field('hour') === '24' ? '00' : field('hour');
  return `${field('year')}-${field('month')}-${field('day')}T${hour}:${field('minute')}:${field('second')}`;
}

function parsePickupComponents(isoString) {
 // O horário é hora de parede de Nova York, sem fuso. Ler os componentes literais é o que
 // o Pydantic faz do outro lado - anexar um fuso aqui reintroduziria o desvio do.
  const naive = isoString.replace(/(?:Z|[+-]\d{2}:?\d{2})$/, '');
  return new Date(`${naive}Z`);
}

function buildFeatureVector(payload) {
  const pickup = parsePickupComponents(payload.pickup_datetime);
 // datetime.weekday() do Python: segunda = 0. getUTCDay() do JS: domingo = 0.
  const dayOfWeek = (pickup.getUTCDay() + 6) % 7;

  const vector = {
    trip_distance: payload.trip_distance,
    hour_of_day: pickup.getUTCHours(),
    day_of_week: dayOfWeek,
    is_weekend: dayOfWeek >= EXPLAINER_WEEKEND_START ? 1 : 0,
    is_airport_trip:
      EXPLAINER_AIRPORT_IDS.has(payload.PULocationID) ||
      EXPLAINER_AIRPORT_IDS.has(payload.DOLocationID)
        ? 1
        : 0,
    is_congestion_zone:
      EXPLAINER_CRZ_IDS.has(payload.PULocationID) ||
      EXPLAINER_CRZ_IDS.has(payload.DOLocationID)
        ? 1
        : 0,
  };

  for (const [code, featureName] of Object.entries(EXPLAINER_RATECODE_FEATURES)) {
    vector[featureName] = payload.RatecodeID === Number(code) ? 1 : 0;
  }
  return vector;
}

// Tarifa fixa regulada entre o JFK e Manhattan. Espelha JFK_FLAT_FARE_AMOUNT em
// src/core/constants.py - é a mesma constante da TLC, e as duas mudam juntas.
const FLAT_FARE_METER_USD = 70.00;

// Regras que a API pode ter aplicado no lugar do modelo, de src/ml/fare_rules.py.
const PRICING_RULE_FLAT_FARE = 'jfk_flat_fare';

function buildFlatFareExplanation(fare) {
 // Numa tarifa fixa a decomposição por coeficiente descreveria um mecanismo que não agiu:
 // o valor não vem de intercepto mais distância, vem da tabela da TLC. O painel troca de
 // explicação em vez de somar contribuições que não produziram o número ,.
  return {
    meter: FLAT_FARE_METER_USD,
    extras: fare - FLAT_FARE_METER_USD,
  };
}

function buildExplanation(info, payload) {
  const vector = buildFeatureVector(payload);

  const terms = Object.entries(info.coefficients).map(([feature, coefficient]) => {
    const value = vector[feature] ?? 0;
    return { feature, value, contribution: coefficient * value };
  });

  const reconstructed =
    info.intercept + terms.reduce((sum, term) => sum + term.contribution, 0);

  return { intercept: info.intercept, terms, reconstructed };
}

function splitDisplayTerms(explanation, fare, threshold = DISPLAY_THRESHOLD_USD) {
  const rows = explanation.terms.filter(
    (term) => Math.abs(term.contribution) >= threshold
  );
 // O resto é derivado do total exibido, não somado termo a termo: assim
 // intercepto + linhas + resto = tarifa SEMPRE, absorvendo ruído de ponto flutuante.
  const shown = rows.reduce((sum, term) => sum + term.contribution, 0);
  const others = fare - explanation.intercept - shown;
  return { rows, others };
}


// ════════════════════════════════════════════════════════════════════════════
// MODO ESTÁTICO
//
// O GitHub Pages serve arquivo e nada mais: não há processo para responder /predict. Como o
// modelo é linear, a predição é a mesma conta que `buildExplanation` já refaz para explicar
// cada estimativa - então o modo estático REUSA essa função em vez de reimplementar o modelo.
// Duas implementações da mesma conta é o defeito, e não é isso que está sendo feito aqui.
//
// A camada de regras doprecisa vir junto: sem ela, uma corrida JFK↔Manhattan
// mostraria a estimativa do modelo onde a API mostraria a tarifa regulada.
// ════════════════════════════════════════════════════════════════════════════

const STATIC_MODELS_URL = 'data/models.json';

function predictLocally(info, payload, staticConfig) {
  const flatFare = isFlatFareTrip(payload, staticConfig);
  if (flatFare !== null) {
    return { predicted_fare: flatFare, pricing_rule: PRICING_RULE_FLAT_FARE };
  }

  const { reconstructed } = buildExplanation(info, payload);
  const floor = staticConfig.minimum_total_amount;
  if (reconstructed < floor) {
    return { predicted_fare: floor, pricing_rule: 'minimum_fare' };
  }
  return { predicted_fare: reconstructed, pricing_rule: 'model' };
}

function isFlatFareTrip(payload, staticConfig) {
 // Deriva das zonas, nunca do RatecodeID - a mesma regra do servidor, pelo mesmo motivo.
  const ends = [payload.PULocationID, payload.DOLocationID];
  const touchesJfk = ends.includes(JFK_ZONE_ID_FOR_RULES);
  const touchesManhattan = ends.some(isManhattanZoneId);
  if (!touchesJfk || !touchesManhattan) return null;

  const excess = staticConfig.excess;
  return excess === null || excess === undefined
    ? null
    : staticConfig.flat_fare_amount + excess;
}

const JFK_ZONE_ID_FOR_RULES = 132;

// Espelha MANHATTAN_LOCATION_IDS de src/core/constants.py. As duas listas saem do mesmo
// índice oficial e mudam juntas - o teste que compara as duas existe para garantir isso.
const MANHATTAN_ZONE_IDS_FOR_RULES = new Set([
  4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 103, 107, 113,
  114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158,
  161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233,
  234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263,
]);

function isManhattanZoneId(zoneId) {
  return MANHATTAN_ZONE_IDS_FOR_RULES.has(zoneId);
}


// A margem de erro exibida é a da região do desembarque, não a do modelo inteiro. A análise de
// equidade da Fase 5 mediu, na frota amarela, de US$ 5,65 em Manhattan a US$ 25,25 em Staten
// Island: um número só, igual em toda corrida, promete na periferia uma precisão que o sistema
// não tem. Cai para a agregada quando a região não foi medida - artefato antigo, ou recorte sem
// corridas suficientes para o RMSE ser estável. Ver.
function marginFor(info, borough) {
  const measured = info.rmse_by_borough && borough ? info.rmse_by_borough[borough] : undefined;
  return {
    value: measured ?? info.rmse,
    borough: measured === undefined ? null : borough,
  };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    DISPLAY_THRESHOLD_USD,
    MANHATTAN_ZONE_IDS_FOR_RULES,
    STATIC_MODELS_URL,
    isFlatFareTrip,
    isManhattanZoneId,
    predictLocally,
    FLAT_FARE_METER_USD,
    PRICING_RULE_FLAT_FARE,
    buildFlatFareExplanation,
    marginFor,
    EXPLAINER_AIRPORT_IDS,
    EXPLAINER_CRZ_IDS,
    EXPLAINER_RATECODE_FEATURES,
    buildExplanation,
    buildFeatureVector,
    parsePickupComponents,
    splitDisplayTerms,
    toNycWallClock,
  };
}
