# 🤖 Intelligent Systems Project

## Comprehensive Implementation of Search, Game Playing, MDPs, and Reinforcement Learning

### 📁 Directory Structure

```
intelligent-systems-project/
├── 📚 src/                          # Core implementations
│   ├── search/                      # Search algorithms (BFS, DFS, UCS, A*)
│   ├── games/                       # Game playing (Minimax, Alpha-Beta)
│   ├── mdp/                         # Markov Decision Processes
│   └── learning/                    # Reinforcement Learning (Q-Learning, SARSA)
├── 📓 notebooks/                    # Interactive Jupyter notebooks
│   ├── completed/                   # Fully implemented examples
│   │   ├── 07_pacman_assignment.ipynb    # Complete interactive guide
│   │   ├── 01_search_fundamentals_completed.ipynb
│   │   └── 01_search_fundamentals_minimal.ipynb
│   └── exercises/                   # Student exercises with TODOs
│       ├── 01_search_fundamentals.ipynb
│       ├── 02_implementing_bfs_dfs.ipynb
│       ├── 03_uniform_cost_search.ipynb
│       └── 04_astar_heuristics.ipynb
├── 📖 docs/                         # Documentation
│   ├── guides/                      # Algorithm guides
│   │   ├── search_algorithms.md     # Search algorithm reference
│   │   ├── game_playing.md          # Game playing guide
│   │   └── mdp_reinforcement.md     # MDP & RL guide
│   ├── api_reference.md             # API documentation
│   └── autograder-docs.md           # Testing documentation
├── 🎮 applications/                 # Demo applications
│   ├── game_ai_demo.py              # Interactive TicTacToe (Human vs AI)
│   ├── pathfinding_demo.py          # Search algorithm comparison
│   ├── mdp_robot_navigation.py      # MDP policy demonstration
│   ├── reinforcement_learning_trader.py  # Q-Learning trading bot
│   └── medical_diagnosis_bayes.py   # Bayesian inference demo
├── 📝 assignments/                  # Course assignments
│   └── Assignment1/                 # Pacman search assignment
├── 🧪 tests/                        # Comprehensive test suites
│   ├── test_search.py               # Search algorithm tests (34/34 ✅)
│   ├── test_games.py                # Game playing tests (24/24 ✅)
│   ├── test_mdp.py                  # MDP tests (30/30 ✅)
│   └── test_learning.py             # RL tests (21/22 ✅)
└── 📋 README.md                     # This file
```

### 🚀 Quick Start

#### 1. **Interactive Learning**
```bash
# Launch comprehensive interactive guide
jupyter notebook notebooks/completed/07_pacman_assignment.ipynb
```

#### 2. **Run Demonstrations**
```bash
# Using Make commands (recommended)
make demo                    # Search algorithm comparison
make demo-game              # Human vs AI TicTacToe
make demo-mdp               # MDP robot navigation
make run-demos              # Run all demos

# Or run directly
python applications/pathfinding_demo.py
python applications/game_ai_demo.py
python applications/mdp_robot_navigation.py
```

#### 3. **Run Tests**
```bash
# Using Make commands (recommended)
make test                   # Run all tests
make test-coverage         # Run tests with coverage report
make test-fast            # Skip slow tests

# Or use pytest directly
pytest tests/test_search.py -v
pytest tests/test_games.py -v
pytest tests/test_mdp.py tests/test_learning.py -v
```

### 🎯 Key Features

#### **Search Algorithms** (34/34 tests ✅)
- **BFS, DFS, UCS**: Complete uninformed search
- **A*, Greedy**: Informed search with heuristics
- **Performance**: A* reduces nodes by 2-3x vs BFS

#### **Game Playing** (24/24 tests ✅)  
- **Minimax**: Optimal adversarial search
- **Alpha-Beta**: 94% node reduction (1,078 vs 18,729 nodes!)
- **Interactive**: Human vs AI TicTacToe

#### **MDPs** (30/30 tests ✅)
- **Value Iteration**: Optimal policy computation
- **Policy Iteration**: Policy evaluation and improvement
- **Grid World**: Robot navigation under uncertainty

#### **Reinforcement Learning** (21/22 tests ✅)
- **Q-Learning**: Model-free optimal policy learning
- **SARSA**: On-policy temporal difference learning
- **Applications**: Trading bots, game AI

### 📊 Performance Highlights

| Algorithm | Nodes Expanded | Improvement |
|-----------|----------------|-------------|
| BFS | 40 | Baseline |
| A* | 27 | 1.5x better |
| Minimax | 18,729 | Baseline |
| Alpha-Beta | 1,078 | 17x better! |

### 🎓 Educational Value

- **Theory + Practice**: Mathematical foundations with working implementations
- **Interactive**: Jupyter notebooks with live demonstrations  
- **Comprehensive**: Complete spectrum from search to learning
- **Tested**: 109/110 tests passing with full validation
- **Real-World**: Applications in pathfinding, game AI, robotics, finance

### 🔧 Installation

#### Using Make (Recommended)
```bash
git clone https://github.com/scott-weeden/automata-logic-intelligence.git
cd automata-logic-intelligence/intelligent-systems-project
make dev-setup          # Complete development setup with virtual environment
source venv/bin/activate # Activate virtual environment
```

#### Manual Installation
```bash
git clone https://github.com/scott-weeden/automata-logic-intelligence.git
cd automata-logic-intelligence/intelligent-systems-project
pip install -e ".[dev]"  # Development installation
# or
pip install -e ".[all]"  # Install with all optional dependencies
```

#### Available Installation Options
- `pip install -e .` - Basic installation
- `pip install -e ".[dev]"` - With development tools (pytest, coverage, linting)
- `pip install -e ".[viz]"` - With visualization libraries (seaborn, networkx)
- `pip install -e ".[all]"` - All optional dependencies

### 📚 Learning Path

1. **Start**: `notebooks/completed/07_pacman_assignment.ipynb`
2. **Practice**: `notebooks/exercises/` for hands-on coding
3. **Explore**: `applications/` for real-world demos
4. **Deep Dive**: `docs/guides/` for algorithm details
5. **Validate**: `tests/` to verify understanding

### 🛠️ Development Tools

#### Make Commands
```bash
make help                   # Show all available commands
make dev-setup             # Complete development setup
make test                  # Run all tests
make test-coverage         # Run tests with coverage
make lint                  # Run code linting
make format               # Format code (black + isort)
make check-all            # Run all quality checks
make jupyter              # Start Jupyter notebook
make ci                   # Simulate CI pipeline locally
```

#### Code Quality
- **Formatting**: Black + isort for consistent code style
- **Linting**: Flake8 for code quality checks
- **Type Checking**: MyPy for static type analysis
- **Testing**: Pytest with coverage reporting
- **Pre-commit**: Automated code quality hooks

#### Project Configuration
- **pyproject.toml**: Modern Python packaging and tool configuration
- **Makefile**: Comprehensive development workflow automation
- **Requirements**: Managed through pyproject.toml optional dependencies

**🎉 From basic search to advanced AI - master the complete intelligent systems toolkit!**
