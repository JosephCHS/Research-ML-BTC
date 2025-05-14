PYTHON_VERSION = 3.12.3
VENV = .venv
REQUIREMENTS_DIR = source/requirements

.PHONY: all init dev clean update-deps

all: init run clean

init: $(VENV)/bin/activate

$(VENV)/bin/activate:
	pyenv install -s $(PYTHON_VERSION)
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	. $(VENV)/bin/activate && \
	pip install --upgrade pip && \
	pip install pip-tools && \
	pip-compile $(REQUIREMENTS_DIR)/requirements.txt -o $(REQUIREMENTS_DIR)/compiled-requirements.txt && \
	pip-compile $(REQUIREMENTS_DIR)/requirements-dev.txt -o $(REQUIREMENTS_DIR)/compiled-requirements-dev.txt && \
	pip-sync $(REQUIREMENTS_DIR)/compiled-requirements.txt $(REQUIREMENTS_DIR)/compiled-requirements-dev.txt

dev: init
	. $(VENV)/bin/activate && \
	pip install -e .

update-deps:
	. $(VENV)/bin/activate && \
	pip-compile --upgrade $(REQUIREMENTS_DIR)/requirements.txt -o $(REQUIREMENTS_DIR)/compiled-requirements.txt && \
	pip-compile --upgrade $(REQUIREMENTS_DIR)/requirements-dev.txt -o $(REQUIREMENTS_DIR)/compiled-requirements-dev.txt && \
	pip-sync $(REQUIREMENTS_DIR)/compiled-requirements.txt $(REQUIREMENTS_DIR)/compiled-requirements-dev.txt

run:
	. $(VENV)/bin/activate && python source/main.py

clean:
	rm -rf $(VENV)
	rm -rf build
	rm -rf documentation/*
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
