.PHONY: help install test lint format clean data notebook app all

VENV     := .venv
PYTHON   := python3.11
BIN      := $(VENV)/bin
PIP      := $(BIN)/pip
PY       := $(BIN)/python

help:
	@echo "zuri-kreislauf — common commands"
	@echo ""
	@echo "  make install     Create .venv and install package + dev extras"
	@echo "  make data        Fetch/cache the City of Zurich open-data CSVs"
	@echo "  make test        Run pytest"
	@echo "  make lint        ruff check + format check"
	@echo "  make format      Apply ruff format & autofixes"
	@echo "  make all         data + train/evaluate all layers"
	@echo "  make notebook    Launch Jupyter notebook server"
	@echo "  make app         Launch the Streamlit dashboard"
	@echo "  make clean       Remove build/test caches and virtualenv"

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel

install: $(BIN)/python
	$(PIP) install -e ".[app,dev]"

data:
	$(PY) scripts/download_data.py

test:
	$(BIN)/pytest -v

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

all: data
	$(PY) scripts/run_all.py

notebook:
	$(BIN)/jupyter notebook notebooks/

app:
	$(BIN)/streamlit run app/streamlit_app.py

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
