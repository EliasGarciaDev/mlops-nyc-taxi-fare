# NYC Taxi Fare Predictor - Sistema MLOps

Sistema web de MLOps para predição de tarifas de táxi em Nova York (NYC TLC), desenvolvido como Trabalho de Conclusão de Curso (TCC) no **Instituto Federal Sul-rio-grandense (IFSul)**.

**Orientador:** Prof. Alessandro da Silveira Dias  
**Autor:** Elias Garcia

---

## 1. Visão Geral

O projeto implementa o ciclo de vida completo de engenharia de Machine Learning (MLOps) aplicado à estimativa de tarifas de táxi em Nova York:
- **DataOps:** Extração, normalização e limpeza de arquivos Parquet mensais disponibilizados pela NYC TLC (Taxi & Limousine Commission).
- **Modelagem:** Treinamento de modelos lineares por frota (Yellow Cab e Green Cab) com validação por corte temporal estrito.
- **Serviço REST:** API em FastAPI com validação de contratos via Pydantic v2 e carregamento dinâmico de modelos.
- **Interface Web:** Aplicação interativa com mapa Leaflet.js, geocodificação, polígonos oficiais das 263 zonas de táxi e painel explicativo de tarifa (Explainable AI).
- **Monitoramento & Governança:** Detecção de Data Drift (KS-Test e PSI), cálculo contínuo de métricas reais e retreinamento com gate de promoção Champion/Challenger.

---

## 2. Tecnologias Utilizadas

- **Linguagem & Backend:** Python 3.12, FastAPI, Pydantic v2, Uvicorn
- **Machine Learning & Dados:** Scikit-Learn, Pandas, PyArrow, SciPy, Joblib
- **Frontend:** HTML5, CSS3, JavaScript vanilla, Leaflet.js, Tailwind CSS
- **Qualidade & Testes:** Pytest, Pytest-cov, Ruff, Mypy (strict)
- **Geocodificação:** Nominatim / OpenStreetMap com proxy backend

---

## 3. Estrutura do Repositório

```
nyc-taxi-fare-predictor/
├── src/
│   ├── core/           # Constantes da TLC, configurações, indexação espacial e logs
│   ├── pipeline/       # Extração Parquet, limpeza de outliers, features e pipeline de treino
│   ├── ml/             # Treinamento, split temporal, gate champion-challenger, regras de tarifa
│   ├── api/            # Servidor FastAPI, schemas Pydantic v2, rotas REST e proxy de geocoding
│   ├── monitoring/     # Detecção de drift (PSI/KS) e gatilhos de retreinamento
│   ├── cli/            # Interfaces de linha de comando (train, backtest, autopilot)
│   └── web/            # Interface do usuário (HTML, CSS, JS, Leaflet e dados estáticos)
├── tests/              # Suíte de testes unitários e de integração (Python e JavaScript)
├── scripts/            # Scripts utilitários de qualidade (check.py) e preparação de dados
├── pyproject.toml      # Configuração de linters (ruff, mypy) e ambiente de testes
└── requirements.txt    # Dependências do projeto
```

---

## 4. Como Executar o Projeto

### Pré-requisitos
- Python 3.12+
- Node.js (necessário apenas para os testes unitários do frontend)

### 1. Clonar o repositório e criar o ambiente virtual
```bash
git clone https://github.com/EliasGarciaDev/mlops-nyc-taxi-fare.git
cd mlops-nyc-taxi-fare

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Treinar os modelos iniciais
O pipeline baixa os dados públicos da TLC, aplica a limpeza, deriva features e gera os artefatos versionados em `models/`:

```bash
# Treinamento inicial com janela curta de demonstração (Jan a Mar/2024)
python -m src.cli.train --taxi-type yellow --from 2024-01 --to 2024-03
python -m src.cli.train --taxi-type green --from 2024-01 --to 2024-03
```

### 3. Iniciar o servidor da API
```bash
uvicorn src.api.app:app --reload --port 8000
```

Acesse no seu navegador:
* **Interface Web do Usuário:** `http://localhost:8000/app/`
* **Documentação OpenAPI (Swagger):** `http://localhost:8000/docs`

---

## 5. Endpoints da API

* `POST /predict` - Estima a tarifa total da corrida a partir dos dados de origem, destino e frota.
* `GET /model-info/{taxi_type}` - Retorna os coeficientes, versão ativa e métricas de validação do modelo.
* `GET /health` - Verificação de disponibilidade da aplicação.
* `GET /ready` - Verificação de prontidão dos modelos carregados em memória.

### Exemplo de Payload para `POST /predict`:
```json
{
  "taxi_type": "yellow",
  "trip_distance": 3.5,
  "passenger_count": 1,
  "PULocationID": 161,
  "DOLocationID": 237,
  "RatecodeID": 1,
  "pickup_datetime": "2024-01-15T08:30:00"
}
```

---

## 6. Execução da Suíte de Testes e Qualidade

Para rodar todos os testes unitários e de integração:

```bash
# Testes do backend e pipeline (Python)
pytest tests/ -v --cov=src

# Testes de regras de negócio e geometria da interface (JavaScript)
node --test tests/web/*.test.js

# Verificação estrita de linters e tipos (Ruff + Mypy Strict)
python scripts/check.py
```

---

## 7. Licença

Este projeto é desenvolvido para fins acadêmicos como Trabalho de Conclusão de Curso (TCC) no IFSul.
