# Intelligent Systems Project

## Overview
This project implements core artificial intelligence algorithms including search strategies, game playing, Markov Decision Processes (MDPs), and reinforcement learning. Based on CS 5368: Intelligent Systems coursework.

## Project Structure
- `src/` - Core algorithm implementations
- `applications/` - Real-world AI applications and demos
- `notebooks/` - Educational Jupyter notebooks
- `tests/` - Comprehensive test suites
- `templates/` - Student assignment templates
- `docs/` - API documentation and guides

## Key Components

### Search Algorithms
- Breadth-First Search (BFS)
- Depth-First Search (DFS) 
- Uniform Cost Search (UCS)
- A* Search with heuristics

### Game Playing
- Minimax algorithm
- Alpha-Beta pruning
- Game state representation

### Decision Making
- Markov Decision Processes
- Value iteration
- Policy iteration
- Q-Learning and TD Learning

## Installation
```bash
pip install -r requirements.txt
python setup.py install
```

# Implementation Order (with dependency reasoning):
## Phase 1: Core Foundation (No dependencies)

src/__init__.py - Empty or with version info
src/search/__init__.py, src/games/__init__.py, src/mdp/__init__.py, src/learning/__init__.py - Package initializers
src/search/problem.py - Define base SearchProblem class (no dependencies)
src/games/game_state.py - Define base GameState and GameAgent classes

## Phase 2: Utilities and Helpers

src/search/utils.py - Implement Node and PriorityQueue classes
src/search/heuristics.py - Distance functions and heuristic classes

## Phase 3: Search Algorithms (depends on problem.py and utils.py)

src/search/algorithms.py - Implement all search algorithms:

SearchAgent base class first
Then BreadthFirstSearch, DepthFirstSearch
Then UniformCostSearch (needs PriorityQueue)
Finally AStarSearch, GreedyBestFirstSearch (need heuristics)

## Phase 4: Game Playing (depends on game_state.py)

src/games/minimax.py - Basic Minimax implementation
src/games/alphabeta.py - Alpha-beta pruning (extends minimax concepts)

## Phase 5: MDPs (mostly independent)

src/mdp/mdp.py - Base MDP classes (MarkovDecisionProcess, MDP, GridMDP)
src/mdp/value_iteration.py - Value iteration (depends on mdp.py)
src/mdp/policy_iteration.py - Policy iteration (depends on mdp.py)

## Phase 6: Learning (depends on MDP concepts)

src/learning/td_learning.py - Base TD learning concepts
src/learning/qlearning.py - Q-Learning and SARSA implementations

## Phase 7: Templates (depends on core implementations)

templates/search_agent_template.py
templates/game_agent_template.py
templates/mdp_agent_template.py
templates/test_template.py

## Phase 8: Tests (depends on all src/ files)

tests/__init__.py
tests/test_search.py
tests/test_games.py
tests/test_mdp.py
tests/test_learning.py

## Phase 9: Applications (depends on core modules)

applications/__init__.py
applications/pathfinding_demo.py - Uses search module
applications/game_ai_demo.py - Uses games module
applications/mdp_robot_navigation.py - Uses MDP module
applications/reinforcement_learning_trader.py - Uses learning module
applications/medical_diagnosis_bayes.py - May use multiple modules

## Phase 10: Documentation and Setup

requirements.txt - Can be created early
setup.py - Package configuration
README.md - Project overview
docs/api_reference.md - After implementation
Other docs/ - Module-specific documentation

## Phase 11: Notebooks (depends on working implementations)

All notebooks/ - Interactive demonstrations

Key Principles I'm Following:

Bottom-up approach: Start with base classes and interfaces
Resolve dependencies first: Never implement something that depends on unimplemented code
Test as you go: Implement tests right after the corresponding module
Documentation last: Write docs after implementation is stable

Critical Dependencies to Remember:

Search algorithms need SearchProblem, Node, and PriorityQueue
A and Greedy* need heuristic functions
Game algorithms need GameState base class
MDP solvers need MDP class definitions
Q-Learning benefits from understanding MDP structure
All tests need their corresponding source modules
Applications need fully working core modules

This order ensures you never hit a "missing dependency" error and can test each component as soon as it's built!

## Usage
See individual module documentation and notebooks for detailed examples.
