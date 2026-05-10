'use strict';

// ════════════════════════════════════════════════════════════════════════════
// CONSTANTES GEOGRÁFICAS
// ════════════════════════════════════════════════════════════════════════════

const NYC_CENTER = [40.7580, -73.9855];
const AVG_SPEED  = 0.28; // mi/min

// Limites usados apenas enquanto os contornos oficiais não chegam. Assim que o índice de
// zonas carrega, o mapa passa a ser limitado pelo território realmente atendido.
const PROVISIONAL_SW = [40.4774, -74.2591];
const PROVISIONAL_NE = [40.9176, -73.7004];

// Cada aeroporto tem um regime tarifário próprio: JFK tem tarifa fixa com Manhattan nos dois
// sentidos, Newark tem sobretaxa própria e LaGuardia é taxímetro comum.
const JFK_ZONE_ID       = 132;
const NEWARK_ZONE_ID    = 1;
const RATECODE_STANDARD = 1;
const RATECODE_JFK      = 2;
const RATECODE_NEWARK   = 3;

// ════════════════════════════════════════════════════════════════════════════
// I18N - TRADUÇÕES
// ════════════════════════════════════════════════════════════════════════════

const TRANSLATIONS = {
  pt: {
    title: 'NYC Taxi Fare', subtitle: 'Arraste os marcadores ou busque endereços',
    'btn-yellow': '🚕 Yellow Cab', 'btn-green': '🟢 Green Cab',
    'green-warning': 'Green Taxi não embarca em Manhattan ao sul da W 110th / E 96th. Mova o embarque para o norte ou outro bairro.',
    'too-close': 'Origem e destino estão muito próximos. Ajuste os pontos no mapa.',
    'origin-ph': 'Local de embarque', 'dest-ph': 'Para onde vamos?',
    miles: 'milhas', minutes: 'minutos',
    'calc-btn': 'Calcular Rota →', loading: 'Calculando...',
    'fare-label': 'Tarifa estimada',
    'explain-down': 'Entender esta estimativa ↓', 'explain-up': 'Entender esta estimativa ↑',
    hint: '📍 Arraste os marcadores para ajustar a rota',
    'jfk-badge': '✈ Tarifa fixa JFK', 'newark-badge': '✈ Tarifa fixa Newark',
    'not-found': 'Endereço não encontrado. Tente um bairro ou ponto turístico.',
    'error-msg': 'Serviço temporariamente indisponível. Tente novamente em alguns instantes.',
    'time-note': 'Estimativa para %TIME%%PEAK%.', peak: ' · horário de pico',
    'base-fare': 'Tarifa base', distance: 'Distância (%D% mi)', duration: 'Duração (~%T% min)',
    hour: 'Horário (%H%h)', weekend: 'Final de semana', weekday: 'Dia de semana',
    'airport-fee': 'Taxa de aeroporto', 'error-margin': 'Margem de erro',
    'congestion': 'Zona de congestionamento', 'rate-jfk': 'Tarifa fixa JFK', 'rate-newark': 'Tarifa Newark', 'rate-nassau': 'Tarifa Nassau/Westchester', 'rate-negotiated': 'Tarifa negociada', 'other-factors': 'Outros fatores',
    total: 'Total estimado', 'model-note': 'Modelo treinado com %N% corridas · RMSE $%R%',
    'fixed-rate': 'Tarifa fixa TLC - distância não altera o valor base.', 'lang-title': 'Idioma',
    'flat-fare-meter': 'Tarifa fixa JFK ↔ Manhattan', 'flat-fare-extras': 'Sobretaxas, pedágio e gorjeta (média)',
    'flat-fare-note': 'Valor regulado pela TLC: a distância não entra na conta.',
    'search-offline': 'Busca de endereços indisponível aqui. Arraste os marcadores no mapa.',
    'outside': 'Fora da área atendida pelos táxis de Nova York. Ajuste o ponto para dentro da cidade.',
    'zones-error': 'Não foi possível carregar os limites da cidade. Recarregue a página.',
    'no-pickup-here': 'Táxis de Nova York não embarcam passageiros em Newark. Só é possível desembarcar aqui.',
  },
  en: {
    title: 'NYC Taxi Fare', subtitle: 'Drag markers or search addresses',
    'btn-yellow': '🚕 Yellow Cab', 'btn-green': '🟢 Green Cab',
    'green-warning': 'Green Taxi cannot pick up in Manhattan south of W 110th / E 96th. Move pickup north or to another borough.',
    'too-close': 'Origin and destination are too close. Adjust the points on the map.',
    'origin-ph': 'Pickup location', 'dest-ph': 'Where to?',
    miles: 'miles', minutes: 'minutes',
    'calc-btn': 'Get Quote →', loading: 'Calculating...',
    'fare-label': 'Estimated fare',
    'explain-down': 'Understand this estimate ↓', 'explain-up': 'Understand this estimate ↑',
    hint: '📍 Drag the markers to adjust your route',
    'jfk-badge': '✈ Fixed rate JFK', 'newark-badge': '✈ Fixed rate Newark',
    'not-found': 'Address not found. Try a neighborhood or landmark.',
    'error-msg': 'Service temporarily unavailable. Please try again.',
    'time-note': 'Estimate for %TIME%%PEAK%.', peak: ' · peak hour',
    'base-fare': 'Base fare', distance: 'Distance (%D% mi)', duration: 'Duration (~%T% min)',
    hour: 'Time (%H%h)', weekend: 'Weekend', weekday: 'Weekday',
    'airport-fee': 'Airport fee', 'error-margin': 'Error margin',
    'congestion': 'Congestion zone', 'rate-jfk': 'JFK flat rate', 'rate-newark': 'Newark rate', 'rate-nassau': 'Nassau/Westchester rate', 'rate-negotiated': 'Negotiated rate', 'other-factors': 'Other factors',
    total: 'Estimated total', 'model-note': 'Model trained on %N% trips · RMSE $%R%',
    'fixed-rate': 'Fixed TLC rate - distance does not affect the base price.', 'lang-title': 'Language',
    'flat-fare-meter': 'JFK ↔ Manhattan flat fare', 'flat-fare-extras': 'Surcharges, tolls and tip (average)',
    'flat-fare-note': 'TLC-regulated amount: distance is not part of the calculation.',
    'search-offline': 'Address search unavailable here. Drag the markers on the map.',
    'outside': 'Outside the area served by New York City taxis. Move the point into the city.',
    'zones-error': 'Could not load the city boundaries. Please reload the page.',
    'no-pickup-here': 'New York City taxis cannot pick up passengers at Newark. Drop-off only.',
  },
};

function i18n(key) {
  const lang = (typeof state !== 'undefined' ? state.language : null)
            || localStorage.getItem('taxi-lang') || 'pt';
  const dict = TRANSLATIONS[lang] || TRANSLATIONS.pt;
  return dict[key] ?? TRANSLATIONS.pt[key] ?? key;
}

// ════════════════════════════════════════════════════════════════════════════
// MAPA
// ════════════════════════════════════════════════════════════════════════════

const map = L.map('map', {
  center: NYC_CENTER, zoom: 12,
  maxBounds: L.latLngBounds(L.latLng(...PROVISIONAL_SW), L.latLng(...PROVISIONAL_NE)),
  maxBoundsViscosity: 1.0,
  zoomControl: false,
});
L.control.zoom({ position: 'bottomright' }).addTo(map);

L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
  attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
  subdomains: 'abcd', maxZoom: 19,
}).addTo(map);

// ════════════════════════════════════════════════════════════════════════════
// ÍCONES
// ════════════════════════════════════════════════════════════════════════════

function pinIcon(fill) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="30" height="42" viewBox="0 0 30 42">
    <filter id="sh${fill.replace('#','')}"><feDropShadow dx="0" dy="2" stdDeviation="1.5" flood-opacity="0.25"/></filter>
    <path filter="url(#sh${fill.replace('#','')})" fill="${fill}" stroke="white" stroke-width="1.5"
      d="M15 2C8.37 2 3 7.37 3 14c0 9.75 12 26 12 26S27 23.75 27 14C27 7.37 21.63 2 15 2z"/>
    <circle fill="white" cx="15" cy="14" r="5"/>
  </svg>`;
  return L.divIcon({ className: '', html: svg, iconSize: [30, 42], iconAnchor: [15, 42] });
}

const originIcon = pinIcon('#22c55e');
const destIcon   = pinIcon('#ef4444');

// ════════════════════════════════════════════════════════════════════════════
// ESTADO
// ════════════════════════════════════════════════════════════════════════════

const state = {
  origin:   L.latLng(NYC_CENTER[0], NYC_CENTER[1]),
  dest:     L.latLng(NYC_CENTER[0] - 0.018, NYC_CENTER[1] + 0.025),
  taxiType: 'yellow',
  tripDistance: 0,
  tripDuration: 0,
  originZone: null,
  destZone: null,
  rateCode: RATECODE_STANDARD,
  lastFare: null,
  lastPayload: null,
  modelInfo: null,
  hintDismissed: false,
  language: localStorage.getItem('taxi-lang') || 'pt',
};

let isCalculating = false;

// ════════════════════════════════════════════════════════════════════════════
// MARCADORES E POLILINHA
// ════════════════════════════════════════════════════════════════════════════

const originMarker = L.marker(state.origin, { draggable: true, icon: originIcon }).addTo(map);
const destMarker   = L.marker(state.dest,   { draggable: true, icon: destIcon   }).addTo(map);
const polyline     = L.polyline([state.origin, state.dest], {
  color: '#3b82f6', weight: 3, dashArray: '7 5', opacity: 0.8,
}).addTo(map);

// ════════════════════════════════════════════════════════════════════════════
// FUNÇÕES GEOGRÁFICAS
// ════════════════════════════════════════════════════════════════════════════

function haversine(a, b) {
  const R = 3958.8;
  const dLat = (b.lat - a.lat) * Math.PI / 180;
  const dLng = (b.lng - a.lng) * Math.PI / 180;
  const s = Math.sin(dLat / 2) ** 2
          + Math.cos(a.lat * Math.PI / 180) * Math.cos(b.lat * Math.PI / 180)
          * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

// A tarifa fixa do JFK vale entre o aeroporto e Manhattan nos dois sentidos, então a regra
// olha as duas pontas da corrida - e não apenas o destino.
function detectRateCode(originZone, destZone) {
  const ends = [originZone, destZone];
  if (ends.some(zone => zone !== null && zone.id === NEWARK_ZONE_ID)) return RATECODE_NEWARK;

  const touchesJfk = ends.some(zone => zone !== null && zone.id === JFK_ZONE_ID);
  const touchesManhattan = ends.some(zone => isManhattanZone(zone));
  return touchesJfk && touchesManhattan ? RATECODE_JFK : RATECODE_STANDARD;
}

// ════════════════════════════════════════════════════════════════════════════
// ÁREA ATENDIDA
// ════════════════════════════════════════════════════════════════════════════

let outsideWarningTimer = null;

// Destaca o aviso por alguns segundos quando o usuário tenta sair da cidade. Um aviso
// permanente vira ruído; um que aparece no momento da tentativa é lido.
// `rejected` é a posição recusada, que precisa ser passada porque o marcador já voltou para a
// última posição válida quando este aviso aparece.
function flashOutsideWarning(role, rejected) {
  const zone = taxiZones.at(rejected);

  // Ponto dentro de uma zona onde apenas o embarque é proibido merece outra mensagem: o
  // usuário está vendo o aeroporto desenhado no mapa, e "fora da área" o deixaria sem entender.
  const message = role === 'origin' && zone !== null ? 'no-pickup-here' : 'outside';
  document.getElementById('outside-warning-text').textContent = i18n(message);

  const warning = document.getElementById('outside-warning');
  warning.classList.remove('hidden');
  clearTimeout(outsideWarningTimer);
  outsideWarningTimer = setTimeout(() => {
    if (isPickupAllowed(state.originZone) && isDropoffAllowed(state.destZone)) {
      warning.classList.add('hidden');
    }
  }, 4000);
}

function paintServiceArea(index) {
  const bounds = L.latLngBounds(
    L.latLng(index.bbox[1], index.bbox[0]),
    L.latLng(index.bbox[3], index.bbox[2]),
  );
  map.setMaxBounds(bounds);

  // Traço da mesma cor e opacidade do preenchimento: as divisórias entre zonas vizinhas
  // deixam de aparecer e a área atendida é lida como uma mancha única, o formato da cidade.
  L.geoJSON(toLeafletOutline(index.zones), {
    style: {
      color: '#3b82f6', weight: 1, opacity: 0.13,
      fillColor: '#3b82f6', fillOpacity: 0.13,
    },
    interactive: false,
  }).addTo(map);
}

async function loadServiceArea() {
  try {
    const index = await taxiZones.load();
    paintServiceArea(index);
  } catch {
    // Sem os contornos o app segue utilizável com os limites provisórios, mas não pode
    // afirmar em qual zona o passageiro está - e sem isso a predição seria inventada.
    document.getElementById('zones-warning').classList.remove('hidden');
  }
  checkState();
}

// ════════════════════════════════════════════════════════════════════════════
// ATUALIZAÇÃO DE ESTADO DA CORRIDA
// ════════════════════════════════════════════════════════════════════════════

function checkState() {
  // As zonas da TLC são a fonte da verdade: elas definem tanto onde há serviço quanto qual
  // regime tarifário se aplica, e são o que a API recebe como PULocationID e DOLocationID.
  // Resolver antes da distância é obrigatório - a conversão de linha reta para rota depende
  // do par de zonas, e usá-las desatualizadas mediria a corrida anterior.
  state.originZone = taxiZones.at(state.origin);
  state.destZone   = taxiZones.at(state.dest);
  state.rateCode   = detectRateCode(state.originZone, state.destZone);

  // A haversine é a linha reta entre os marcadores; o taxímetro cobra pela rota. A conversão
  // usa a distância que as corridas daquele par de zonas percorreram de fato, e cai num fator
  // de área quando o par é raro demais para ter mediana confiável.
  const straight = haversine(state.origin, state.dest);
  const dist = roadDistance(taxiZones.distances, state.originZone, state.destZone, straight);
  const dur  = dist / AVG_SPEED;
  state.tripDistance = dist;
  state.tripDuration = dur;

  document.getElementById('distance-display').textContent = dist.toFixed(1);
  document.getElementById('duration-display').textContent = Math.round(dur);

  polyline.setLatLngs([state.origin, state.dest]);

  const badge     = document.getElementById('airport-badge');
  const badgeText = document.getElementById('airport-badge-text');
  if (state.rateCode === RATECODE_JFK) {
    badgeText.textContent = i18n('jfk-badge');
    badge.classList.remove('hidden');
  } else if (state.rateCode === RATECODE_NEWARK) {
    badgeText.textContent = i18n('newark-badge');
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }

  const greenWarn = document.getElementById('green-warning');
  // O bloqueio é a Hail Exclusionary Zone, não Manhattan inteira: acima da W 110th / E 96th
  // o boro taxi embarca legalmente, e é onde o bloqueio anterior negava serviço.
  const greenBlocked =
    state.originZone !== null && !isGreenPickupAllowed(state.originZone);
  if (state.taxiType === 'green') {
    greenWarn.classList.remove('hidden');
    greenWarn.classList.toggle('warning-red', greenBlocked);
    greenWarn.classList.toggle('warning-amber', !greenBlocked);
  } else {
    greenWarn.classList.add('hidden');
  }

  const tooClose = dist < 0.1;
  document.getElementById('too-close-warning').classList.toggle('hidden', !tooClose);

  const routeAllowed = isPickupAllowed(state.originZone) && isDropoffAllowed(state.destZone);
  if (taxiZones.loaded && routeAllowed) {
    document.getElementById('outside-warning').classList.add('hidden');
  }

  if (!isCalculating) {
    const greenBlock = state.taxiType === 'green' && greenBlocked;
    document.getElementById('calc-btn').disabled =
      tooClose || greenBlock || !taxiZones.loaded || !routeAllowed;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Animação suave do marcador no mapa
// ════════════════════════════════════════════════════════════════════════════

function setCardOpacity(v) {
  document.getElementById('card').style.opacity = v;
}

function setMarkerActive(marker, active) {
  if (!marker._icon) return;
  marker._icon.classList.toggle('marker-dragging', active);
}

// ════════════════════════════════════════════════════════════════════════════
// GEOCODIFICAÇÃO
//
// Passa sempre pelo backend: o navegador não consegue cumprir a política de uso do
// Nominatim, porque `User-Agent` é forbidden header name na Fetch API . Sem
// backend - publicação estática - o recurso simplesmente não existe, e a interface diz isso.
// ════════════════════════════════════════════════════════════════════════════

let geocodingAvailable = true;

async function reverseGeocode(latlng) {
  // Numa publicação estática não há proxy de geocodificação. Depois do primeiro 404 as
  // tentativas param: cada arrasto de marcador dispararia uma, e o console encheria de erros
  // que não são erros - só a ausência de um recurso que esta versão não tem.
  if (!geocodingAvailable) return '';
  try {
    const r = await fetch(`/geocode/reverse?lat=${latlng.lat}&lon=${latlng.lng}`);
    if (r.status === 404) {
      geocodingAvailable = false;
      return '';
    }
    if (!r.ok) return '';
    const d = await r.json();
    return d.display_name?.split(',').slice(0, 2).join(', ') ?? '';
  } catch { return ''; }
}

// ════════════════════════════════════════════════════════════════════════════
// HINT INICIAL
// ════════════════════════════════════════════════════════════════════════════

function dismissHint() {
  if (state.hintDismissed) return;
  state.hintDismissed = true;
  const h = document.getElementById('drag-hint');
  h.style.opacity = '0';
  setTimeout(() => { h.style.display = 'none'; }, 700);
}

// ════════════════════════════════════════════════════════════════════════════
// EVENTOS DE DRAG DOS MARCADORES (bugs 1, 2, 12)
// ════════════════════════════════════════════════════════════════════════════

// Última posição válida (dentro do bounding box)
let lastValidOrigin = state.origin;
let lastValidDest   = state.dest;

function makeDragHandlers(marker, posKey, lastValidRef, inputId) {
  marker.on('dragstart', () => {
    setCardOpacity('0.15');
    setMarkerActive(marker, true);
    dismissHint();
  });

  marker.on('drag', (e) => {
    state[posKey] = e.latlng;
    checkState();
  });

  // dragend: valida bounds e reverse geocode (bug 1 corrigido: sem anônimas duplicadas)
  marker.on('dragend', async () => {
    const pos = marker.getLatLng();
    setMarkerActive(marker, false);
    setCardOpacity('1');

    // Fora da área atendida não existe corrida, então o marcador volta para a última posição
    // válida em vez de deixar o usuário montar um trajeto impossível. Embarque e desembarque
    // têm regras diferentes: um táxi de Nova York leva até Newark, mas não pega passageiro lá.
    if (taxiZones.loaded && !taxiZones.allows(pos, posKey)) {
      marker.setLatLng(lastValidRef.value);
      state[posKey] = lastValidRef.value;
      flashOutsideWarning(posKey, pos);
      checkState();
      return;
    }

    lastValidRef.value = pos;
    state[posKey] = pos;
    checkState();

    try {
      const addr = await reverseGeocode(pos);
      if (addr) document.getElementById(inputId).value = addr;
    } catch { /* silencia erro de rede */ }
  });
}

const lastValidOriginRef = { value: lastValidOrigin };
const lastValidDestRef   = { value: lastValidDest   };

makeDragHandlers(originMarker, 'origin', lastValidOriginRef, 'origin-input');
makeDragHandlers(destMarker,   'dest',   lastValidDestRef,   'dest-input');

// ════════════════════════════════════════════════════════════════════════════
// Autocomplete de busca de endereços
// ════════════════════════════════════════════════════════════════════════════

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// A busca passa pelo backend em vez de ir direto ao Nominatim. O motivo é que o navegador
// não consegue cumprir a política de uso do serviço: `User-Agent` é forbidden header name na
// Fetch API e some em silêncio, então as requisições chegavam anônimas. O proxy identifica a
// aplicação, respeita o limite de uma requisição por segundo e serve do cache o que se repete.
async function fetchPlaces(q) {
  // Sem backend - o caso do GitHub Pages - a busca não existe: o proxy é justamente o que
  // torna a chamada ao Nominatim legítima. Devolver vazio em silêncio faria o usuário digitar
  // e concluir que o endereço não existe, então a ausência do recurso é sinalizada.
  try {
    const r = await fetch(`/geocode/search?q=${encodeURIComponent(q)}`);
    if (r.status === 404) {
      // A rota não existe: é uma publicação estática, não uma busca sem resultado. Servidor
      // estático devolve 404 em vez de recusar a conexão, então o catch abaixo nunca veria.
      geocodingAvailable = false;
      return [];
    }
    if (!r.ok) return [];
    const body = await r.json();
    return body.results ?? [];
  } catch {
    geocodingAvailable = false;
    return [];
  }
}

async function searchNominatim(q) {
  // Tentativa 1: busca exata
  let results = await fetchPlaces(q);
  if (results.length) return results;

  // Tentativa 2: remove último caractere (erro de digitação no final, ex: "Underill" → "Underil")
  if (q.length > 4) {
    results = await fetchPlaces(q.slice(0, -1));
    if (results.length) return results;
  }

  // Tentativa 3: primeira palavra significativa ≥4 letras (ex: "Unerhill Avenue" → "Unerhill" → "Unerhi")
  const words = q.trim().split(/\s+/).filter(w => w.length >= 4);
  if (words.length && words[0].toLowerCase() !== q.trim().toLowerCase()) {
    results = await fetchPlaces(words[0]);
    if (results.length) return results;
  }

  return [];
}

function setupAutocomplete(inputId, dropdownId, validRef, posKey, markerRef) {
  const input    = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);

  const doSearch = debounce(async (q) => {
    if (q.length < 2) { dropdown.classList.add('hidden'); return; }
    try {
      const results = await searchNominatim(q);
      dropdown.innerHTML = '';
      if (!results.length) {
        const div = document.createElement('div');
        div.className = 'dropdown-item not-found';
        div.textContent = i18n(geocodingAvailable ? 'not-found' : 'search-offline');
        dropdown.appendChild(div);
      } else {
        results.forEach(item => {
          const div = document.createElement('div');
          div.className = 'dropdown-item';
          div.textContent = item.display_name.split(',').slice(0, 3).join(', ');
          div.addEventListener('click', () => {
            const ll = L.latLng(parseFloat(item.lat), parseFloat(item.lon));
            dropdown.classList.add('hidden');

            // O Nominatim devolve resultados de fora da cidade mesmo com o recorte pedido.
            if (taxiZones.loaded && !taxiZones.allows(ll, posKey)) {
              flashOutsideWarning(posKey, ll);
              return;
            }

            input.value = div.textContent;
            state[posKey] = ll;
            validRef.value = ll;
            markerRef.setLatLng(ll);
            map.panTo(ll);
            checkState();
          });
          dropdown.appendChild(div);
        });
      }
      dropdown.classList.remove('hidden');
    } catch { dropdown.classList.add('hidden'); }
  }, 400);

  input.addEventListener('input', (e) => doSearch(e.target.value));
  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.add('hidden');
    }
  });
}

setupAutocomplete('origin-input', 'origin-dropdown', lastValidOriginRef, 'origin', originMarker);
setupAutocomplete('dest-input',   'dest-dropdown',   lastValidDestRef,   'dest',   destMarker);

// ════════════════════════════════════════════════════════════════════════════
// TIPO DE TÁXI
// ════════════════════════════════════════════════════════════════════════════

function selectTaxi(type) {
  state.taxiType  = type;
  state.modelInfo = null;
  document.getElementById('btn-yellow').classList.toggle('taxi-active', type === 'yellow');
  document.getElementById('btn-yellow').classList.toggle('text-gray-500', type !== 'yellow');
  document.getElementById('btn-green').classList.toggle('taxi-active', type === 'green');
  document.getElementById('btn-green').classList.toggle('text-gray-500', type === 'yellow');
  checkState();
}

// ════════════════════════════════════════════════════════════════════════════
// Atualização de horário e fuso da cidade
// ════════════════════════════════════════════════════════════════════════════

function isPeakHour(h) { return (h >= 7 && h <= 9) || (h >= 17 && h <= 19); }

function updateClock() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, '0');
  const mm = String(now.getMinutes()).padStart(2, '0');
  const timeStr = `${hh}:${mm}`;
  const peak = isPeakHour(now.getHours()) ? i18n('peak') : '';

  document.getElementById('clock').textContent = timeStr;

  const timeNote = document.getElementById('time-note');
  if (timeNote && !document.getElementById('result').classList.contains('hidden')) {
    timeNote.textContent = i18n('time-note').replace('%TIME%', timeStr).replace('%PEAK%', peak);
  }
}

setInterval(updateClock, 1000);

// ════════════════════════════════════════════════════════════════════════════
// Seletor de idioma da interface
// ════════════════════════════════════════════════════════════════════════════

function toggleLangMenu() {
  document.getElementById('lang-menu').classList.toggle('hidden');
  document.getElementById('a11y-panel').classList.add('hidden');
}

function applyLanguage(lang) {
  state.language = lang;
  localStorage.setItem('taxi-lang', lang);

  // Aplica textContent para elementos data-i18n
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const val = TRANSLATIONS[lang]?.[el.dataset.i18n] ?? TRANSLATIONS.pt[el.dataset.i18n];
    if (val !== undefined) el.textContent = val;
  });
  // Aplica placeholder para elementos data-i18n-ph
  document.querySelectorAll('[data-i18n-ph]').forEach(el => {
    const val = TRANSLATIONS[lang]?.[el.dataset.i18nPh] ?? TRANSLATIONS.pt[el.dataset.i18nPh];
    if (val !== undefined) el.placeholder = val;
  });

  // Atualiza label do botão de idioma com código visível
  const LANG_LABELS = { pt: 'PT', en: 'EN' };
  const langCode = document.getElementById('lang-code');
  if (langCode) langCode.textContent = LANG_LABELS[lang] || lang.toUpperCase();

  document.getElementById('lang-menu').classList.add('hidden');
  checkState(); // re-check para atualizar warnings traduzidos

  // Re-renderiza explainer se estiver aberto (sem precisar fechar/abrir)
  const explainerPanel = document.getElementById('explainer');
  if (explainerPanel && !explainerPanel.classList.contains('hidden') && state.modelInfo && state.lastPayload) {
    renderExplainer(state.modelInfo, state.lastPayload, state.lastFare, state.lastRule);
    document.getElementById('explain-btn').textContent = i18n('explain-up');
  }
}

document.addEventListener('click', (e) => {
  const menu = document.getElementById('lang-menu');
  const btn  = document.getElementById('lang-btn');
  if (!menu.contains(e.target) && !btn.contains(e.target)) {
    menu.classList.add('hidden');
  }
});

// ════════════════════════════════════════════════════════════════════════════
// Cálculo e requisição de tarifa estimada
// ════════════════════════════════════════════════════════════════════════════

async function calculateFare() {
  if (isCalculating) return;
  if (!isPickupAllowed(state.originZone)) {
    flashOutsideWarning('origin', state.origin);
    return;
  }
  if (!isDropoffAllowed(state.destZone)) {
    flashOutsideWarning('dest', state.dest);
    return;
  }
  isCalculating = true;

  const btn = document.getElementById('calc-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>${i18n('loading')}`;

  // A API espera hora local de Nova York, a mesma convenção dos timestamps da TLC com que
  // o modelo foi treinado. Enviar UTC deslocava hour_of_day em quatro ou cinco horas .
  // Ajuste de timestamp para compatibilidade com a validação de data da API.
  const pickupIso = toNycWallClock(new Date(Date.now() - 60_000));

  const distMi = Math.max(0.1, parseFloat(state.tripDistance.toFixed(2)));
  const durMin = Math.max(1, parseFloat(state.tripDuration.toFixed(1)));

  const payload = {
    taxi_type: state.taxiType,
    trip_distance: distMi,
    passenger_count: 1,
    PULocationID: state.originZone.id,
    DOLocationID: state.destZone.id,
    RatecodeID: state.rateCode,
    pickup_datetime: pickupIso,
    trip_duration_minutes: durMin,
  };

  try {
    const data = await requestPrediction(payload);
    if (data === null) {
      showError();
      return;
    }

    state.lastFare    = data.predicted_fare;
    state.lastPayload = payload;
    state.lastRule    = data.pricing_rule;
    showResult(data.predicted_fare, distMi, durMin);

  } catch {
    showError();
  } finally {
    btn.disabled = false;
    btn.textContent = i18n('calc-btn');
    setTimeout(() => { isCalculating = false; }, 2000); // Debounce de 2s para evitar requisições repetidas
  }
}

function formatFare(v) {
  return '$' + v.toFixed(2).replace('.', ',');
}

function showResult(fare, distMi, durMin) {
  const now = new Date();
  const hh  = String(now.getHours()).padStart(2, '0');
  const mm  = String(now.getMinutes()).padStart(2, '0');
  const peak = isPeakHour(now.getHours()) ? i18n('peak') : '';

  document.getElementById('result').classList.remove('hidden');
  document.getElementById('fare-value').textContent  = formatFare(fare);
  document.getElementById('trip-summary').textContent = `${distMi.toFixed(1)} mi · ~${Math.round(durMin)} min`;
  document.getElementById('time-note').textContent   = i18n('time-note').replace('%TIME%', `${hh}:${mm}`).replace('%PEAK%', peak);

  document.getElementById('explainer').classList.add('hidden');
  document.getElementById('explain-btn').textContent = i18n('explain-down');
}

function showError() {
  document.getElementById('result').classList.remove('hidden');
  document.getElementById('fare-value').textContent  = '-';
  document.getElementById('trip-summary').textContent = i18n('error-msg');
  document.getElementById('time-note').textContent   = '';
  document.getElementById('explainer').classList.add('hidden');
}

// ════════════════════════════════════════════════════════════════════════════
// Painel explicativo de tarifa (Explainable AI)
// ════════════════════════════════════════════════════════════════════════════

async function toggleExplainer() {
  const panel = document.getElementById('explainer');
  const btn   = document.getElementById('explain-btn');

  if (!panel.classList.contains('hidden')) {
    panel.classList.add('hidden');
    btn.textContent = i18n('explain-down');
    return;
  }

  btn.textContent = '⟳';
  try {
    if (!state.modelInfo) {
      state.modelInfo = await requestModelInfo(state.taxiType);
      if (state.modelInfo === null) { btn.textContent = i18n('explain-down'); return; }
    }
    renderExplainer(state.modelInfo, state.lastPayload, state.lastFare, state.lastRule);
    panel.classList.remove('hidden');
    btn.textContent = i18n('explain-up');
  } catch {
    btn.textContent = i18n('explain-down');
  }
}


// ════════════════════════════════════════════════════════════════════════════
// PREDIÇÃO: API QUANDO EXISTE, CÁLCULO LOCAL QUANDO NÃO
//
// Publicado no GitHub Pages não há backend, e a estimativa é calculada no navegador com os
// coeficientes exportados por scripts/export_static_models.py. É a mesma conta e os mesmos
// coeficientes que a API usaria - quem calcula é `buildExplanation`, a função que o painel
// XAI já usava para reproduzir a resposta do servidor.
// ════════════════════════════════════════════════════════════════════════════

let staticModels = null;
let apiAvailable = null;

async function loadStaticModels() {
  if (staticModels === null) {
    const r = await fetch(STATIC_MODELS_URL);
    if (!r.ok) throw new Error('modelos estáticos indisponíveis');
    staticModels = await r.json();
  }
  return staticModels;
}

function staticConfigFor(models, taxiType) {
  return {
    flat_fare_amount: models.flat_fare_amount,
    minimum_total_amount: models.minimum_total_amount,
    excess: models.fleets[taxiType]?.flat_fare_excess ?? null,
  };
}

async function requestPrediction(payload) {
  if (apiAvailable !== false) {
    try {
      const r = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (r.ok) {
        apiAvailable = true;
        return await r.json();
      }
      // Resposta de erro é da API, não ausência dela: não vale cair para o modo estático.
      if (apiAvailable === true) return null;
    } catch {
      // Rede indisponível - é o caso do Pages, e o modo estático assume abaixo.
    }
    apiAvailable = false;
  }

  try {
    const models = await loadStaticModels();
    const info = models.fleets[payload.taxi_type];
    if (!info) return null;
    return predictLocally(info, payload, staticConfigFor(models, payload.taxi_type));
  } catch {
    return null;
  }
}

async function requestModelInfo(taxiType) {
  if (apiAvailable !== false) {
    try {
      const r = await fetch(`/model-info/${taxiType}`);
      if (r.ok) return await r.json();
    } catch {
      // Mesmo caminho da predição: sem API, os coeficientes vêm do arquivo estático.
    }
  }
  const models = await loadStaticModels();
  return models.fleets[taxiType] ?? null;
}

function marginRow(info, payload) {
  const zone = taxiZones.byId(payload.DOLocationID);
  const { value, borough } = marginFor(info, zone && zone.borough);
  const scope = borough === null ? '' : ` <span class="text-gray-400">(${borough})</span>`;
  return `<div class="explain-row">
    <span class="explain-label">${i18n('error-margin')}${scope}</span>
    <span class="explain-rmse">± $${value.toFixed(2)}</span>
  </div>`;
}

function renderFlatFareExplainer(fare, row) {
  const { meter, extras } = buildFlatFareExplanation(fare);

  let html = row(i18n('flat-fare-meter'), meter, 'explain-base');
  html += row(i18n('flat-fare-extras'), extras, extras >= 0 ? 'explain-pos' : 'explain-neg');

  html += `<div class="explain-row" style="border-top:2px solid #e5e7eb;margin-top:4px;padding-top:6px;">
    <span style="font-weight:700;color:#111827;">${i18n('total')}</span>
    <span style="font-weight:700;color:#111827;">${formatFare(fare)}</span>
  </div>`;

  html += `<div class="explain-row">
    <span class="explain-label" style="font-style:italic;color:#d97706;">${i18n('flat-fare-note')}</span>
  </div>`;

  document.getElementById('explainer').innerHTML = html;
}

function renderExplainer(info, payload, fare, pricingRule) {
  function row(label, value, cls) {
    const sign = value >= 0 ? '+$' : '-$';
    return `<div class="explain-row">
      <span class="explain-label">${label}</span>
      <span class="${cls}">${cls === 'explain-base' ? '$' + value.toFixed(2) : sign + Math.abs(value).toFixed(2)}</span>
    </div>`;
  }

  if (pricingRule === PRICING_RULE_FLAT_FARE) {
    renderFlatFareExplainer(fare, row);
    return;
  }

  // A decomposição itera sobre TODOS os coeficientes servidos pelo /model-info,
  // multiplicados pelas features reconstruídas do payload - a mesma derivação de
  // src/api/routes.py. É o que garante que intercepto + linhas + resto = total ,
  // e que a explicação descreve o horário que a API usou, não o do relógio local .
  const explanation = buildExplanation(info, payload);
  const { rows, others } = splitDisplayTerms(explanation, fare);

  if (Math.abs(explanation.reconstructed - fare) > 0.01) {
    // Cliente e servidor derivaram features diferentes - é o/regredindo.
    console.warn('explainer_mismatch', {
      reconstructed: explanation.reconstructed,
      served: fare,
      model_version: info.model_version,
    });
  }

  const pickupHour = buildFeatureVector(payload).hour_of_day;
  const peakMark = isPeakHour(pickupHour) ? ` <span class="text-orange-500">⚡</span>` : '';
  const labels = {
    trip_distance: i18n('distance').replace('%D%', payload.trip_distance.toFixed(1)),
    hour_of_day: i18n('hour').replace('%H%', pickupHour) + peakMark,
    day_of_week: i18n('weekday'),
    is_weekend: i18n('weekend'),
    is_airport_trip: i18n('airport-fee'),
    is_congestion_zone: i18n('congestion'),
    is_rate_jfk: i18n('rate-jfk'),
    is_rate_newark: i18n('rate-newark'),
    is_rate_nassau_westchester: i18n('rate-nassau'),
    is_rate_negotiated: i18n('rate-negotiated'),
  };

  let html = row(i18n('base-fare'), explanation.intercept, 'explain-base');
  for (const term of rows) {
    html += row(labels[term.feature] ?? term.feature, term.contribution,
      term.contribution >= 0 ? 'explain-pos' : 'explain-neg');
  }
  if (Math.abs(others) >= DISPLAY_THRESHOLD_USD) {
    html += row(i18n('other-factors'), others, others >= 0 ? 'explain-pos' : 'explain-neg');
  }

  if (payload.RatecodeID > 1) {
    html += `<div class="explain-row">
      <span class="explain-label" style="font-style:italic;color:#d97706;">${i18n('fixed-rate')}</span>
    </div>`;
  }

  html += `<div class="explain-row" style="border-top:2px solid #e5e7eb;margin-top:4px;padding-top:6px;">
    <span style="font-weight:700;color:#111827;">${i18n('total')}</span>
    <span style="font-weight:700;color:#111827;">${formatFare(fare)}</span>
  </div>`;

  // A margem de erro fica FORA do bloco aditivo, depois do total: ela não é uma parcela
  // da soma, e no meio das linhas fazia a coluna parecer não fechar.
  html += marginRow(info, payload);

  html += `<p style="margin-top:8px;font-size:10px;color:#9ca3af;line-height:1.4;">
    ${i18n('model-note').replace('%N%', info.training_samples.toLocaleString()).replace('%R%', info.rmse.toFixed(2))}
  </p>`;

  document.getElementById('explainer').innerHTML = html;
}

// ════════════════════════════════════════════════════════════════════════════
// ACESSIBILIDADE
// ════════════════════════════════════════════════════════════════════════════

const A11Y_STORAGE_KEY = 'taxi-a11y';

const a11yState = Object.assign(
  { fontSize: 0, highContrast: false, reduceMotion: false, largePointer: false },
  JSON.parse(localStorage.getItem(A11Y_STORAGE_KEY) || '{}')
);

function saveA11y() {
  localStorage.setItem(A11Y_STORAGE_KEY, JSON.stringify(a11yState));
}

function applyA11y() {
  const card = document.getElementById('card');

  // Tamanho de fonte via zoom no card (funciona com Tailwind rem fixo)
  const zooms = [1, 1.12, 1.24, 1.38];
  if (card) {
    card.style.zoom = zooms[a11yState.fontSize] ?? 1;
    // Compensa a posição do card quando ampliado para não vazar pela borda
    card.style.transformOrigin = 'top left';
  }

  // Alto contraste
  document.body.classList.toggle('a11y-contrast', a11yState.highContrast);

  // Reduzir movimento
  document.body.classList.toggle('a11y-reduce-motion', a11yState.reduceMotion);

  // Ponteiro grande
  document.body.classList.toggle('a11y-large-pointer', a11yState.largePointer);

  // Atualiza visuais dos botões do painel
  const panel = document.getElementById('a11y-panel');
  if (!panel) return;
  panel.querySelector('#a11y-font-display').textContent =
    ['A', 'A+', 'A++', 'A+++'][a11yState.fontSize] ?? 'A';

  ['contrast', 'motion', 'pointer'].forEach(key => {
    const btn = panel.querySelector(`[data-a11y="${key}"]`);
    if (!btn) return;
    const active = key === 'contrast' ? a11yState.highContrast
                 : key === 'motion'   ? a11yState.reduceMotion
                 : a11yState.largePointer;
    btn.classList.toggle('a11y-btn-active', active);
    btn.setAttribute('aria-pressed', String(active));
  });
}

function toggleA11yPanel() {
  const panel = document.getElementById('a11y-panel');
  panel.classList.toggle('hidden');
  document.getElementById('lang-menu').classList.add('hidden');
}

function a11yFontStep(delta) {
  a11yState.fontSize = Math.max(0, Math.min(3, a11yState.fontSize + delta));
  saveA11y(); applyA11y();
}

function a11yToggle(key) {
  if (key === 'contrast')  a11yState.highContrast  = !a11yState.highContrast;
  if (key === 'motion')    a11yState.reduceMotion  = !a11yState.reduceMotion;
  if (key === 'pointer')   a11yState.largePointer  = !a11yState.largePointer;
  saveA11y(); applyA11y();
}

// Fecha painel de acessibilidade ao clicar fora
document.addEventListener('click', (e) => {
  const panel = document.getElementById('a11y-panel');
  const btn   = document.getElementById('a11y-btn');
  if (panel && btn && !panel.contains(e.target) && !btn.contains(e.target)) {
    panel.classList.add('hidden');
  }
});

// ════════════════════════════════════════════════════════════════════════════
// INICIALIZAÇÃO
// ════════════════════════════════════════════════════════════════════════════

// Aplica idioma salvo
applyLanguage(state.language);

// Aplica preferências de acessibilidade salvas
applyA11y();

// Estado inicial. O botão de calcular fica desabilitado até os contornos chegarem, porque
// sem eles não há como saber a zona de embarque.
checkState();
updateClock();
loadServiceArea();

// Hint desaparece em 5s
setTimeout(() => { if (!state.hintDismissed) dismissHint(); }, 5000);

// Reverse geocode das posições iniciais
(async () => {
  const [oAddr, dAddr] = await Promise.all([
    reverseGeocode(state.origin),
    reverseGeocode(state.dest),
  ]);
  if (oAddr) document.getElementById('origin-input').value = oAddr;
  if (dAddr) document.getElementById('dest-input').value   = dAddr;
})();
