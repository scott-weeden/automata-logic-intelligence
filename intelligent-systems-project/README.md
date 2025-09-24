# Intelligent Systems Project

A compact teaching library for CS 5368 that implements classic AI algorithms:
state-space search, adversarial game playing, Markov decision processes, and
reinforcement learning.  The repository also ships assignment templates,
realistic demo applications, and an end-to-end test suite.

## Features
- **Search** – BFS, DFS, UCS, A*, greedy and iterative deepening over custom
  problem definitions.
- **Games** – Tic-Tac-Toe domain with minimax, alpha-beta, and expectimax
  agents.
- **MDPs** – Finite MDP framework with value and policy iteration solvers.
- **Learning** – Tabular Q-learning and SARSA agents ready for experimentation.
- **Applications** – Ready-to-run demos for pathfinding, perfect Tic-Tac-Toe,
  grid-world planning, trading with RL, and naïve Bayes diagnosis.
- **Templates & Tests** – Starter files for coursework plus comprehensive
  pytest suites covering every module.

## Project Layout
```
.
├── applications/                # Demo entry points (Phase 9)
├── docs/                        # API reference and guides
├── src/                         # Library packages: search, games, mdp, learning
├── templates/                   # Assignment templates (Phase 7)
├── tests/                       # Pytest suites for all modules (Phase 8)
├── requirements.txt             # Runtime dependencies
└── setup.py                     # Packaging metadata (Phase 10)
```

## Installation
It is recommended to work inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```

The optional `dev` extra installs pytest and Jupyter notebooks.

## Running the Test Suite
Every stage of the project is validated with pytest:

```bash
pytest tests
```

To inspect basic coverage without installing extra tooling you can leverage the
standard library trace module:

```bash
python -m trace --count --module pytest tests
```

If you prefer detailed reports, install `coverage` and run
`python -m coverage run -m pytest tests` followed by `python -m coverage report`.

## Demo Applications
Each application can be executed as a module or via the console scripts exposed
by the package:

| Demo | Module invocation | Console script |
|------|------------------|----------------|
| Grid pathfinding comparison | `python -m applications.pathfinding_demo` | `is-pathfinding-demo` |
| Perfect-play Tic-Tac-Toe | `python -m applications.game_ai_demo` | `is-game-demo` |
| Grid-world MDP analysis | `python -m applications.mdp_robot_navigation` | `is-mdp-demo` |
| Q-learning trading agent | `python -m applications.reinforcement_learning_trader` | `is-rl-trader` |
| Naïve Bayes diagnosis | `python -m applications.medical_diagnosis_bayes` | `is-medical-bayes` |

All demos print their results to the console and are safe to run out of the box.

## Templates
Reusable scaffolding for assignments lives under `templates/`:

- `search_agent_template.py` – customise a search strategy.
- `game_agent_template.py` – build minimax-style agents with your own heuristics.
- `mdp_agent_template.py` – extend value iteration for bespoke planners.
- `test_template.py` – starting point for writing pytest-based assessments.

Copy the relevant file into your workspace and fill in the TODO sections.

## Documentation
An API reference covering each module is available in `docs/api_reference.md`.
Use it alongside the tests to understand the expected behaviour of every class
and function.

## Contributing
The project follows the original eleven-phase roadmap shown in `AGENTS.md`.  If
you add new functionality, remember to:

1. Include unit tests under `tests/`.
2. Update the documentation when interfaces change.
3. Run `pytest tests` before submitting changes.

Happy hacking!
