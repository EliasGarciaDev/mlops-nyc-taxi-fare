# NYC Taxi Fare Predictor

Sistema web para estimativa de tarifas de táxi em Nova York (NYC TLC), desenvolvido como Trabalho de Conclusão de Curso (TCC) no Instituto Federal Sul-rio-grandense (IFSul).

Orientador: Prof. Alessandro da Silveira Dias
Autor: Elias Garcia

## Visão Geral

Aplicação web para estimativa de corridas das frotas Yellow e Green de Nova York, utilizando modelos de regressão treinados sobre o histórico de viagens disponibilizado pela Taxi & Limousine Commission (TLC).

## Funcionalidades Iniciais
- Extração e limpeza de dados mensais em Parquet da TLC
- Treinamento e avaliação de Regressão Linear com métricas de baseline
- API FastAPI com validação de dados via Pydantic
- Interface web interativa com mapa Leaflet para seleção de origem e destino

## Como Executar
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.app:app --reload
```
