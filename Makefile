# Atalhos do projeto. Todos os alvos usam o interpretador do ambiente virtual diretamente,
# então não é preciso ativá-lo antes de chamar `make`.

# Sobrescreva se você gerencia o ambiente de outro jeito: make test PYTHON=python3
PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

# Janela de treino. Sobrescreva na chamada: make train FROM=2024-01 TO=2024-11
FROM ?= 2024-01
TO ?= 2024-03
PORT ?= 8000
TAXI ?= green

.DEFAULT_GOAL := help
.PHONY: help venv install map train check test serve setup clean autopilot backtest demo-recovery series export-static

help: ## Lista os comandos disponíveis
	@echo "Comandos do nyc-taxi-fare-predictor:"
	@echo
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-9s %s\n", $$1, $$2}'
	@echo
	@echo "Janela de treino: $(FROM) a $(TO)     Porta da API: $(PORT)"
	@echo "Para mudar:  make train FROM=2024-01 TO=2024-11"

venv:
	@test -d .venv || python3 -m venv .venv

install: venv ## Cria o ambiente virtual e instala as dependências
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements-dev.txt
	@echo "Dependências instaladas."

map: ## Baixa e prepara os contornos das zonas de táxi
	$(PYTHON) scripts/fetch_taxi_zones.py

train: ## Treina os modelos das duas frotas na janela FROM..TO
	$(PYTHON) -m src.cli.train --taxi-type yellow --from $(FROM) --to $(TO)
	$(PYTHON) -m src.cli.train --taxi-type green  --from $(FROM) --to $(TO)

check: ## Roda o linter e a verificação de tipos
	$(PYTHON) scripts/check.py

test: ## Roda a suíte de testes completa, Python e JavaScript
	$(PYTHON) -m pytest tests/ -q --cov=src
	# O glob fica sem aspas de propósito: o test runner do Node só expande padrões a partir
	# da versão 21, então quem expande aqui é o shell - funciona em qualquer Node.
	node --test tests/web/*.test.js

autopilot: ## Roda o ciclo autônomo: avalia, reverte, retreina e promove sem operador
	$(PYTHON) -m src.cli.autopilot

backtest: ## Curva de degradação mês a mês da frota TAXI (padrão: green)
	$(PYTHON) -m src.cli.backtest --taxi-type $(TAXI) --from $(FROM) --to $(TO) --min-train-months 2

export-static: ## Exporta os modelos promovidos para a versão estática (GitHub Pages)
	$(PYTHON) scripts/export_static_models.py

demo-recovery: ## Demonstra rollback automático e gatilho por erro, em registro temporário
	$(PYTHON) scripts/demo_autopilot_recovery.py

series: ## Roda o piloto mês a mês na frota TAXI, mostrando o ciclo de vida completo
	$(PYTHON) scripts/simulate_autopilot_series.py --taxi-type $(TAXI)

serve: ## Sobe a API e o frontend em localhost, porta PORT
	$(PYTHON) -m uvicorn src.api.app:app --reload --port $(PORT)

# `map` não entra: o índice de zonas é versionado, e regerá-lo exige rede sem nenhum ganho.
# `export-static` entra porque a versão publicada calcula com os coeficientes exportados -
# sem ele o GitHub Pages sobe com o modelo anterior, ou sem modelo nenhum.
setup: install train export-static ## Prepara o projeto inteiro do zero, pronto para rodar
	@echo
	@echo "Pronto. Agora rode:  make serve"
	@echo "E abra:              http://localhost:$(PORT)/app/"

clean: ## Remove caches de build e de testes
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	@echo "Caches removidos."
