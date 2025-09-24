# Makefile for Automata Theory Library
# Run 'make help' for available commands

.PHONY: help install test clean build docs run-demo quickstart docker lint format coverage

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
VENV := venv
ACTIVATE := . $(VENV)/bin/activate
PROJECT := automata-theory

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(BLUE)╔══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║     Automata Theory Library - Make Commands         ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup & Installation:$(NC)"
	@echo "  $(GREEN)make install$(NC)      - Install library and dependencies"
	@echo "  $(GREEN)make quickstart$(NC)   - Run quick start script"
	@echo "  $(GREEN)make venv$(NC)         - Create virtual environment"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  $(GREEN)make test$(NC)         - Run all tests with pytest"
	@echo "  $(GREEN)make test-student$(NC) - Test student solutions only"
	@echo "  $(GREEN)make test-watch$(NC)   - Run tests in watch mode"
	@echo "  $(GREEN)make coverage$(NC)     - Run tests with coverage report"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  $(GREEN)make lint$(NC)         - Run code linting"
	@echo "  $(GREEN)make format$(NC)       - Format code with black"
	@echo "  $(GREEN)make clean$(NC)        - Clean build artifacts"
	@echo "  $(GREEN)make docs$(NC)         - Generate documentation"
	@echo ""
	@echo "$(YELLOW)Demos & Examples:$(NC)"
	@echo "  $(GREEN)make demo-loop$(NC)    - Run infinite loop demo"
	@echo "  $(GREEN)make demo-trace$(NC)   - Run TM with trace"
	@echo "  $(GREEN)make examples$(NC)     - Create all example machines"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  $(GREEN)make docker-build$(NC) - Build Docker image"
	@echo "  $(GREEN)make docker-run$(NC)   - Run in Docker container"
	@echo ""
	@echo "$(YELLOW)Student Exercises:$(NC)"
	@echo "  $(GREEN)make exercise-1$(NC)   - Test DFA even zeros"
	@echo "  $(GREEN)make exercise-2$(NC)   - Test DFA divisible by 3"
	@echo "  $(GREEN)make exercise-3$(NC)   - Test NFA patterns"
	@echo "  $(GREEN)make grade$(NC)        - Grade all exercises"

# Installation targets
install: venv
	@echo "$(BLUE)Installing automata-theory library...$(NC)"
	@$(ACTIVATE) && $(PIP) install -e .
	@$(ACTIVATE) && $(PIP) install pytest pytest-cov pytest-xdist rich click graphviz
	@echo "$(GREEN)✓ Installation complete!$(NC)"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(BLUE)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	fi

quickstart:
	@chmod +x quickstart.sh
	@./quickstart.sh

# Testing targets
test: venv
	@echo "$(BLUE)Running all tests...$(NC)"
	@$(ACTIVATE) && $(PYTEST) -v --tb=short --color=yes

test-student: venv
	@echo "$(BLUE)Testing student solutions...$(NC)"
	@$(ACTIVATE) && $(PYTEST) test_automata.py::TestDFAExercises -v --tb=short

test-watch: venv
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@$(ACTIVATE) && $(PYTEST) --looponfail

coverage: venv
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(ACTIVATE) && $(PYTEST) --cov=automata_lib --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

# Development targets
lint: venv
	@echo "$(BLUE)Running linters...$(NC)"
	@$(ACTIVATE) && flake8 src/ tests/ --max-line-length=100
	@$(ACTIVATE) && mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: venv
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(ACTIVATE) && black src/ tests/ --line-length=100
	@$(ACTIVATE) && isort src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean:
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -rf build/ dist/ *.egg-info
	@rm -rf htmlcov/ .coverage .pytest_cache/
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@echo "$(GREEN)✓ Clean complete$(NC)"

docs: venv
	@echo "$(BLUE)Generating documentation...$(NC)"
	@$(ACTIVATE) && sphinx-build -b html docs/ docs/_build
	@echo "$(GREEN)✓ Documentation built in docs/_build/index.html$(NC)"

# Demo targets
demo-loop: venv
	@echo "$(YELLOW)═══════════════════════════════════════════════════$(NC)"
	@echo "$(YELLOW)     UNDECIDABILITY DEMONSTRATION$(NC)"
	@echo "$(YELLOW)═══════════════════════════════════════════════════$(NC)"
	@$(ACTIVATE) && automata run examples/loop.tm.json '111' --trace --max-steps 30

demo-trace: venv
	@echo "$(BLUE)Running Turing Machine with trace...$(NC)"
	@$(ACTIVATE) && automata run examples/increment.tm.json '1011' --trace --delay 0.1

examples: venv
	@echo "$(BLUE)Creating example machines...$(NC)"
	@mkdir -p examples
	@$(ACTIVATE) && python -c "from automata_lib import *; \
		import json; \
		with open('examples/generated.dfa.json', 'w') as f: \
			json.dump(create_binary_palindrome_dfa().to_dict(), f, indent=2)"
	@echo "$(GREEN)✓ Examples created in examples/$(NC)"

# Docker targets
docker-build:
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t $(PROJECT):latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run:
	@echo "$(BLUE)Running in Docker container...$(NC)"
	@docker run -it --rm -v $(PWD):/app $(PROJECT):latest

# Student exercise targets
exercise-1: venv
	@echo "$(YELLOW)Testing Exercise 1: DFA Even Zeros$(NC)"
	@$(ACTIVATE) && $(PYTEST) test_automata.py::TestDFAExercises::test_student_dfa_even_zeros -v

exercise-2: venv
	@echo "$(YELLOW)Testing Exercise 2: DFA Divisible by 3$(NC)"
	@$(ACTIVATE) && $(PYTEST) test_automata.py::TestDFAExercises::test_student_dfa_divisible_by_3 -v

exercise-3: venv
	@echo "$(YELLOW)Testing Exercise 3: NFA Patterns$(NC)"
	@$(ACTIVATE) && $(PYTEST) test_automata.py::TestNFAExercises -v

grade: venv
	@echo "$(YELLOW)═══════════════════════════════════════════════════$(NC)"
	@echo "$(YELLOW)           GRADING ALL EXERCISES$(NC)"
	@echo "$(YELLOW)═══════════════════════════════════════════════════$(NC)"
	@$(ACTIVATE) && $(PYTEST) test_automata.py -v --tb=no --quiet | grep -E "(PASSED|FAILED)" | \
		awk '{pass+=gsub(/PASSED/,""); fail+=gsub(/FAILED/,"")} \
		END {print "$(GREEN)Passed: "pass"$(NC)  $(RED)Failed: "fail"$(NC)  Score: "int(pass/(pass+fail)*100)"%"}'

# Continuous Integration
ci: lint test coverage
	@echo "$(GREEN)✓ CI checks complete$(NC)"

# Watch for changes and auto-test
watch: venv
	@echo "$(BLUE)Watching for changes...$(NC)"
	@$(ACTIVATE) && watchmedo auto-restart --patterns="*.py" --recursive -- $(PYTEST) -x

# Interactive Python with library loaded
repl: venv
	@$(ACTIVATE) && $(PYTHON) -i -c "from automata_lib import *; \
		print('Automata Theory Library loaded!'); \
		print('Available: DFA, NFA, TuringMachine'); \
		print('Try: dfa = create_binary_palindrome_dfa()'); \
		print('     dfa.accepts(\"00\", trace=True)')"

# Generate student template files
templates: venv
	@echo "$(BLUE)Generating student templates...$(NC)"
	@mkdir -p student_solutions
	@for i in 1 2 3 4 5 6 7 8; do \
		echo '{"type": "DFA", "states": [], "alphabet": [], "transitions": {}, "start_state": "", "accept_states": []}' \
			> student_solutions/exercise_$$i.json; \
	done
	@echo "$(GREEN)✓ Templates created in student_solutions/$(NC)"

# Performance profiling
profile: venv
	@echo "$(BLUE)Profiling performance...$(NC)"
	@$(ACTIVATE) && $(PYTHON) -m cProfile -s time test_automata.py

# Memory profiling
memcheck: venv
	@echo "$(BLUE)Checking memory usage...$(NC)"
	@$(ACTIVATE) && $(PYTHON) -m memory_profiler test_automata.py

# Security check
security: venv
	@echo "$(BLUE)Running security checks...$(NC)"
	@$(ACTIVATE) && safety check
	@$(ACTIVATE) && bandit -r src/

# Package distribution
dist: clean venv
	@echo "$(BLUE)Building distribution packages...$(NC)"
	@$(ACTIVATE) && $(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)✓ Distribution packages built in dist/$(NC)"

# Upload to PyPI (test)
upload-test: dist
	@echo "$(BLUE)Uploading to TestPyPI...$(NC)"
	@$(ACTIVATE) && twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to PyPI (production)
upload: dist
	@echo "$(RED)Uploading to PyPI (production)...$(NC)"
	@$(ACTIVATE) && twine upload dist/*

.PHONY: all test clean install help