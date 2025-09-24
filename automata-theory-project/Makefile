.PHONY: install test clean lint format

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 src/ tests/

format:
	black src/ tests/

notebooks:
	jupyter lab notebooks/

help:
	@echo "Available commands:"
	@echo "  install  - Install package in development mode"
	@echo "  test     - Run tests"
	@echo "  clean    - Remove Python cache files"
	@echo "  lint     - Run linting"
	@echo "  format   - Format code"
	@echo "  notebooks - Start Jupyter Lab"
