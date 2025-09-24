# Makefile for Intelligent Systems Project
# Python executable
PYTHON := python3
PIP := pip3

# Project directories
SRC_DIR := src
TEST_DIR := tests
APPS_DIR := applications
DOCS_DIR := docs
NOTEBOOKS_DIR := notebooks

# Virtual environment
VENV := venv
VENV_BIN := $(VENV)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

# Coverage settings
COV_REPORT_DIR := htmlcov
COV_REPORT := coverage.xml

.PHONY: help install install-dev install-all clean test test-verbose test-coverage lint format check-format check-lint check-types check-all demo run-demos jupyter notebooks clean-notebooks docs serve-docs build package upload-test upload clean-build clean-pyc clean-test venv venv-clean all

# Default target
all: install-dev test lint

help: ## Show this help message
	@echo "Intelligent Systems Project - Available Commands:"
	@echo "================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install package in production mode
	$(PIP) install .

install-dev: ## Install package in development mode with dev dependencies
	$(PIP) install -e ".[dev]"

install-all: ## Install package with all optional dependencies
	$(PIP) install -e ".[all]"

# Virtual environment targets
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip setuptools wheel

venv-install: venv ## Create venv and install dev dependencies
	$(VENV_PIP) install -e ".[dev]"

venv-clean: ## Remove virtual environment
	rm -rf $(VENV)

# Testing targets
test: ## Run all tests
	pytest

test-verbose: ## Run tests with verbose output
	pytest -v

test-coverage: ## Run tests with coverage report
	pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=xml --cov-report=term-missing

test-fast: ## Run tests excluding slow tests
	pytest -m "not slow"

test-integration: ## Run only integration tests
	pytest -m integration

# Code quality targets
lint: ## Run linting checks
	flake8 $(SRC_DIR) $(TEST_DIR) $(APPS_DIR)

format: ## Format code using black and isort
	black $(SRC_DIR) $(TEST_DIR) $(APPS_DIR)
	isort $(SRC_DIR) $(TEST_DIR) $(APPS_DIR)

check-format: ## Check code formatting without making changes
	black --check $(SRC_DIR) $(TEST_DIR) $(APPS_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR) $(APPS_DIR)

check-lint: ## Check linting without making changes
	flake8 $(SRC_DIR) $(TEST_DIR) $(APPS_DIR)

check-types: ## Run type checking
	mypy $(SRC_DIR)

check-all: check-format check-lint check-types ## Run all code quality checks

# Demo and application targets
demo: ## Run pathfinding demo
	$(PYTHON) $(APPS_DIR)/pathfinding_demo.py

demo-game: ## Run game AI demo
	$(PYTHON) $(APPS_DIR)/game_ai_demo.py

demo-mdp: ## Run MDP robot navigation demo
	$(PYTHON) $(APPS_DIR)/mdp_robot_navigation.py

demo-rl: ## Run reinforcement learning trader demo
	$(PYTHON) $(APPS_DIR)/reinforcement_learning_trader.py

demo-bayes: ## Run medical diagnosis Bayes demo
	$(PYTHON) $(APPS_DIR)/medical_diagnosis_bayes.py

run-demos: demo demo-game demo-mdp ## Run all demos sequentially

# Jupyter notebook targets
jupyter: ## Start Jupyter notebook server
	jupyter notebook

notebooks: ## Start Jupyter and open main tutorial
	jupyter notebook $(NOTEBOOKS_DIR)/completed/07_pacman_assignment.ipynb

clean-notebooks: ## Clean notebook outputs
	jupyter nbconvert --clear-output --inplace $(NOTEBOOKS_DIR)/**/*.ipynb

# Documentation targets
docs: ## Generate documentation
	@echo "Opening documentation..."
	@if command -v open > /dev/null; then \
		open $(DOCS_DIR)/api_reference.md; \
	elif command -v xdg-open > /dev/null; then \
		xdg-open $(DOCS_DIR)/api_reference.md; \
	else \
		echo "Please manually open $(DOCS_DIR)/api_reference.md"; \
	fi

serve-docs: ## Serve documentation locally (if using mkdocs or similar)
	@echo "Serving documentation at http://localhost:8000"
	@if [ -f "mkdocs.yml" ]; then mkdocs serve; else echo "No documentation server configured"; fi

# Build and distribution targets
build: clean-build ## Build package
	$(PYTHON) -m build

package: build ## Create distribution packages
	@echo "Package built successfully!"
	@ls -la dist/

upload-test: build ## Upload to test PyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	$(PYTHON) -m twine upload dist/*

# Cleaning targets
clean: clean-build clean-pyc clean-test clean-notebooks ## Clean all artifacts

clean-build: ## Clean build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Clean Python cache files
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Clean test artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr $(COV_REPORT_DIR)/
	rm -f $(COV_REPORT)
	rm -fr .pytest_cache/
	rm -fr .mypy_cache/

# Performance and analysis targets
profile-search: ## Profile search algorithms
	$(PYTHON) -m cProfile -o search_profile.stats -s cumulative $(APPS_DIR)/pathfinding_demo.py
	@echo "Profile saved to search_profile.stats"

benchmark: ## Run performance benchmarks
	$(PYTHON) -m pytest tests/ -k "benchmark" -v

# Assignment and educational targets
assignment1: ## Run Assignment 1 autograder
	cd assignments/Assignment1 && $(PYTHON) submission_autograder.py

pacman-test: ## Test pacman implementation
	cd assignments/Assignment1 && $(PYTHON) pacman.py -l tinyMaze -p SearchAgent -a fn=dfs

# CI/CD simulation targets
ci: install-dev check-all test-coverage ## Simulate CI pipeline locally
	@echo "✅ CI pipeline completed successfully!"

release-check: ci build ## Check if ready for release
	@echo "🚀 Release check completed!"
	@echo "Built packages:"
	@ls -la dist/

# Development workflow targets
dev-setup: venv-install ## Complete development setup
	@echo "📚 Development environment ready!"
	@echo "Run 'source $(VENV_BIN)/activate' to activate virtual environment"
	@echo "Run 'make jupyter' to start Jupyter notebooks"
	@echo "Run 'make demo' to try the pathfinding demo"

dev-test: ## Quick development test cycle
	pytest tests/ -x -v --tb=short

# Git hooks (if pre-commit is used)
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit: ## Run pre-commit hooks manually
	pre-commit run --all-files

# Information targets
info: ## Show project information
	@echo "Intelligent Systems Project Information"
	@echo "======================================"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Project structure:"
	@find . -type f -name "*.py" | grep -E "(src|applications|tests)" | head -10
	@echo "..."
	@echo "Total Python files: $$(find . -name "*.py" | wc -l)"
	@echo "Test coverage: Run 'make test-coverage' to see coverage"

requirements: ## Generate requirements.txt from pyproject.toml
	pip-compile pyproject.toml

update-deps: ## Update all dependencies
	pip-compile --upgrade pyproject.toml

# Quick shortcuts
t: test ## Quick test shortcut
tc: test-coverage ## Quick test coverage shortcut
l: lint ## Quick lint shortcut
f: format ## Quick format shortcut
d: demo ## Quick demo shortcut
j: jupyter ## Quick jupyter shortcut