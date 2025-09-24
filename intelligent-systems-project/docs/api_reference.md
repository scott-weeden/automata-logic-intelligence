# API Reference - Intelligent Systems Library

## Overview
Complete documentation for all Python classes in the intelligent systems project, organized by module.

---

## Search Module (`src/search/`)

### Core Classes

#### `SearchProblem`
**Base class for search problems**

```python
class SearchProblem:
    def get_start_state(self) -> Any
    def is_goal_state(self, state: Any) -> bool
    def get_successors(self, state: Any) -> List[Tuple[Any, str, float]]
    def get_cost_of_actions(self, actions: List[str]) -> float
```

**Methods:**
- `get_start_state()`: Returns the initial state
- `is_goal_state(state)`: Returns True if state is a goal
- `get_successors(state)`: Returns list of (successor, action, cost) tuples
- `get_cost_of_actions(actions)`: Returns total cost of action sequence

#### `GridSearchProblem(SearchProblem)`
**Grid-based pathfinding problem**

```python
class GridSearchProblem(SearchProblem):
    def __init__(self, grid: List[List], start: Tuple[int, int], goal: Tuple[int, int])
```

**Parameters:**
- `grid`: 2D array where 0/space=free, 1/#=obstacle
- `start`: Starting position (row, col)
- `goal`: Goal position (row, col)

### Search Algorithms

#### `SearchAgent`
**Base class for search algorithms**

```python
class SearchAgent:
    def __init__(self)
    def search(self, problem: SearchProblem) -> List[str]
    
    # Attributes:
    nodes_expanded: int  # Number of nodes explored
```

#### `BreadthFirstSearch(SearchAgent)`
**Breadth-first search implementation**

- **Completeness**: Yes (if branching factor finite)
- **Optimality**: Yes (for unit costs)
- **Time Complexity**: O(b^d)
- **Space Complexity**: O(b^d)

#### `DepthFirstSearch(SearchAgent)`
**Depth-first search implementation**

- **Completeness**: No (infinite paths)
- **Optimality**: No
- **Time Complexity**: O(b^m)
- **Space Complexity**: O(bm)

#### `UniformCostSearch(SearchAgent)`
**Uniform cost search implementation**

- **Completeness**: Yes (if step costs ≥ ε > 0)
- **Optimality**: Yes
- **Time Complexity**: O(b^(1+⌊C*/ε⌋))
- **Space Complexity**: O(b^(1+⌊C*/ε⌋))

#### `AStarSearch(SearchAgent)`
**A* search with heuristic function**

```python
class AStarSearch(SearchAgent):
    def __init__(self, heuristic: Callable[[Any, SearchProblem], float] = None)
```

**Parameters:**
- `heuristic`: Function returning estimated cost to goal

- **Completeness**: Yes
- **Optimality**: Yes (if heuristic is admissible)
- **Time Complexity**: O(b^d)
- **Space Complexity**: O(b^d)

#### `GreedyBestFirstSearch(SearchAgent)`
**Greedy best-first search**

```python
class GreedyBestFirstSearch(SearchAgent):
    def __init__(self, heuristic: Callable[[Any, SearchProblem], float] = None)
```

- **Completeness**: No
- **Optimality**: No
- **Time Complexity**: O(b^m)
- **Space Complexity**: O(b^m)

#### `IterativeDeepeningSearch(SearchAgent)`
**Iterative deepening search**

```python
def search(self, problem: SearchProblem, max_depth: int = 50) -> List[str]
```

- **Completeness**: Yes
- **Optimality**: Yes (for unit costs)
- **Time Complexity**: O(b^d)
- **Space Complexity**: O(bd)

### Utility Classes

#### `Node`
**Search tree node representation**

```python
class Node:
    def __init__(self, state: Any, parent: Node = None, action: str = None, path_cost: float = 0)
    
    # Attributes:
    state: Any          # Current state
    parent: Node        # Parent node
    action: str         # Action from parent
    path_cost: float    # Cost from root
    depth: int          # Depth in tree
    
    # Methods:
    def solution(self) -> List[str]  # Action sequence from root
    def path(self) -> List[Any]      # State sequence from root
```

#### `PriorityQueue`
**Priority queue for search algorithms**

```python
class PriorityQueue:
    def __init__(self, order: str = 'min', f: Callable = lambda x: x)
    def append(self, item: Any) -> None
    def pop(self) -> Any
    def __len__(self) -> int
    def __contains__(self, key: Any) -> bool
```

---

## Games Module (`src/games/`)

### Base Classes

#### `GameState`
**Abstract base class for game states**

```python
class GameState:
    def get_legal_actions(self, agent_index: int = 0) -> List[Any]
    def generate_successor(self, agent_index: int, action: Any) -> GameState
    def is_terminal(self) -> bool
    def get_utility(self, agent_index: int = 0) -> float
```

#### `GameAgent`
**Base class for game-playing agents**

```python
class GameAgent:
    def __init__(self, index: int = 0)
    def get_action(self, game_state: GameState) -> Any
    
    # Attributes:
    index: int              # Agent index (0 or 1)
    nodes_explored: int     # Nodes explored in search
```

### Game Algorithms

#### `MinimaxAgent(GameAgent)`
**Minimax algorithm implementation**

```python
class MinimaxAgent(GameAgent):
    def __init__(self, index: int = 0, depth: int = 2)
```

**Parameters:**
- `index`: Agent index (0=maximizing, 1=minimizing)
- `depth`: Maximum search depth

**Algorithm**: Perfect play assuming rational opponent
- **Time Complexity**: O(b^m)
- **Space Complexity**: O(bm)

#### `AlphaBetaAgent(GameAgent)`
**Alpha-beta pruning optimization**

```python
class AlphaBetaAgent(GameAgent):
    def __init__(self, index: int = 0, depth: int = 2)
```

**Algorithm**: Minimax with branch pruning
- **Best Case Time**: O(b^(m/2))
- **Worst Case Time**: O(b^m)
- **Space Complexity**: O(bm)

#### `ExpectimaxAgent(GameAgent)`
**Expectimax for stochastic games**

```python
class ExpectimaxAgent(GameAgent):
    def __init__(self, index: int = 0, depth: int = 2)
```

**Algorithm**: Expected value for chance nodes
- **Time Complexity**: O(b^m)
- **Space Complexity**: O(bm)

### Game Implementations

#### `Game`
**Abstract game interface**

```python
class Game:
    def actions(self, state: Any) -> List[Any]
    def result(self, state: Any, action: Any) -> Any
    def terminal_test(self, state: Any) -> bool
    def utility(self, state: Any, player: Any) -> float
    def to_move(self, state: Any) -> Any
    def display(self, state: Any) -> None
```

#### `TicTacToe(Game)`
**Tic-Tac-Toe game implementation**

```python
class TicTacToe(Game):
    def __init__(self)
    
    # Attributes:
    initial: GameState  # Starting game state
```

---

## MDP Module (`src/mdp/`)

### Base Classes

#### `MarkovDecisionProcess`
**Abstract MDP interface**

```python
class MarkovDecisionProcess:
    def get_states(self) -> List[Any]
    def get_start_state(self) -> Any
    def get_possible_actions(self, state: Any) -> List[Any]
    def get_transition_states_and_probs(self, state: Any, action: Any) -> List[Tuple[Any, float]]
    def get_reward(self, state: Any, action: Any, next_state: Any) -> float
    def is_terminal(self, state: Any) -> bool
```

#### `MDP`
**General MDP implementation**

```python
class MDP:
    def __init__(self, init: Any, actlist: Any, terminals: Set[Any], 
                 transitions: Dict = None, reward: Any = None, gamma: float = 0.9)
    
    # Methods:
    def R(self, state: Any) -> float                    # Reward function
    def T(self, state: Any, action: Any) -> List[Tuple] # Transition model
    def actions(self, state: Any) -> List[Any]          # Available actions
```

**Parameters:**
- `init`: Initial state
- `actlist`: Actions or action function
- `terminals`: Set of terminal states
- `transitions`: Transition probabilities P(s'|s,a)
- `reward`: Reward function R(s)
- `gamma`: Discount factor [0,1]

#### `GridMDP(MDP)`
**Grid-based MDP for navigation**

```python
class GridMDP(MDP):
    def __init__(self, grid: List[List], terminals: Set[Tuple], 
                 init: Tuple = (0, 0), gamma: float = 0.9)
    
    # Methods:
    def to_grid(self, mapping: Dict) -> List[List]  # Convert to 2D visualization
```

**Features:**
- Stochastic actions (80% intended, 10% each perpendicular)
- Grid obstacles and rewards
- Bounded movement

### MDP Algorithms

#### `ValueIterationAgent`
**Value iteration solver**

```python
class ValueIterationAgent:
    def __init__(self, mdp: MarkovDecisionProcess, discount: float = 0.9, iterations: int = 100)
    
    # Methods:
    def get_value(self, state: Any) -> float        # State value V(s)
    def get_policy(self, state: Any) -> Any         # Optimal action π(s)
    
    # Attributes:
    values: Dict[Any, float]  # State values
```

**Algorithm**: Bellman equation iteration
- **Convergence**: Guaranteed for finite MDPs
- **Time per iteration**: O(|S| × |A| × |S|)

#### `PolicyIterationAgent`
**Policy iteration solver**

```python
class PolicyIterationAgent:
    def __init__(self, mdp: MarkovDecisionProcess, discount: float = 0.9)
    
    # Methods:
    def get_value(self, state: Any) -> float
    def get_policy(self, state: Any) -> Any
    
    # Attributes:
    values: Dict[Any, float]
    policy: Dict[Any, Any]
```

**Algorithm**: Policy evaluation + improvement
- **Convergence**: Finite iterations for finite MDPs
- **Often faster**: Than value iteration in practice

---

## Learning Module (`src/learning/`)

### Reinforcement Learning

#### `QLearningAgent`
**Model-free Q-learning**

```python
class QLearningAgent:
    def __init__(self, action_fn: Callable, discount: float = 0.9, 
                 alpha: float = 0.5, epsilon: float = 0.1)
    
    # Methods:
    def get_q_value(self, state: Any, action: Any) -> float
    def get_max_q_value(self, state: Any) -> float
    def get_action(self, state: Any) -> Any
    def update(self, state: Any, action: Any, next_state: Any, reward: float) -> None
    def start_episode(self) -> None
    def stop_training(self) -> None
    
    # Attributes:
    q_values: Dict[Tuple, float]  # Q(s,a) values
    training: bool                # Training mode flag
```

**Parameters:**
- `action_fn`: Function returning legal actions for state
- `discount`: Discount factor γ [0,1]
- `alpha`: Learning rate [0,1]
- `epsilon`: Exploration rate [0,1]

**Algorithm**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- **Convergence**: To optimal Q* under conditions
- **Off-policy**: Learns optimal while exploring

#### `SARSAAgent(QLearningAgent)`
**On-policy SARSA learning**

```python
class SARSAAgent(QLearningAgent):
    def __init__(self, action_fn: Callable, discount: float = 0.9,
                 alpha: float = 0.5, epsilon: float = 0.1)
    
    # Additional attributes:
    next_action: Any  # Next action for SARSA update
```

**Algorithm**: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
- **On-policy**: Learns policy being followed
- **More conservative**: Than Q-learning

---

## Heuristics Module (`src/search/heuristics.py`)

### Distance Functions

#### `manhattan_distance(pos1: Tuple, pos2: Tuple) -> float`
**L1 distance for grid worlds**
- **Admissible for**: 4-directional movement
- **Formula**: |x₁-x₂| + |y₁-y₂|

#### `euclidean_distance(pos1: Tuple, pos2: Tuple) -> float`
**L2 distance for continuous spaces**
- **Admissible for**: Straight-line movement
- **Formula**: √[(x₁-x₂)² + (y₁-y₂)²]

#### `chebyshev_distance(pos1: Tuple, pos2: Tuple) -> float`
**L∞ distance for 8-directional movement**
- **Admissible for**: Diagonal movement
- **Formula**: max(|x₁-x₂|, |y₁-y₂|)

### Heuristic Classes

#### `GridHeuristic`
**Adaptive grid heuristic**

```python
class GridHeuristic:
    def __init__(self, goal: Tuple, movement_type: str = '4-way')
    def __call__(self, state: Any) -> float
```

**Movement types:**
- `'4-way'`: Manhattan distance
- `'8-way'`: Chebyshev distance  
- `'continuous'`: Euclidean distance

#### `null_heuristic(state: Any) -> float`
**Zero heuristic (reduces A* to UCS)**

---

## Usage Examples

### Search Example
```python
from search import BreadthFirstSearch, GridSearchProblem

# Create problem
grid = [[0, 0, 1], [0, 1, 0], [0, 0, 0]]
problem = GridSearchProblem(grid, (0,0), (2,2))

# Solve with BFS
bfs = BreadthFirstSearch()
solution = bfs.search(problem)
print(f"Path: {solution}")
```

### Game Example
```python
from games import MinimaxAgent, TicTacToe

# Create game and agent
game = TicTacToe()
agent = MinimaxAgent(depth=9)

# Get best move
action = agent.get_action(game.initial)
```

### MDP Example
```python
from mdp import GridMDP, ValueIterationAgent

# Create MDP
grid = [[0, 0, 1], [0, 0, -1]]
mdp = GridMDP(grid, terminals={(1,2)})

# Solve with value iteration
agent = ValueIterationAgent(mdp)
policy = {s: agent.get_policy(s) for s in mdp.get_states()}
```

### Q-Learning Example
```python
from learning import QLearningAgent

# Create agent
agent = QLearningAgent(
    action_fn=lambda s: ['up', 'down', 'left', 'right'],
    alpha=0.1, epsilon=0.1
)

# Training loop
for episode in range(1000):
    state = env.reset()
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, next_state, reward)
        state = next_state
```

---

## Performance Characteristics

### Search Algorithms
| Algorithm | Complete | Optimal | Time | Space |
|-----------|----------|---------|------|-------|
| BFS | Yes* | Yes** | O(b^d) | O(b^d) |
| DFS | No | No | O(b^m) | O(bm) |
| UCS | Yes* | Yes | O(b^C*/ε) | O(b^C*/ε) |
| A* | Yes* | Yes*** | O(b^d) | O(b^d) |

*if branching factor finite, **if step costs equal, ***if heuristic admissible

### Game Algorithms
| Algorithm | Time | Space | Pruning |
|-----------|------|-------|---------|
| Minimax | O(b^m) | O(bm) | None |
| Alpha-Beta | O(b^m/2) | O(bm) | Yes |
| Expectimax | O(b^m) | O(bm) | None |

### MDP Algorithms
| Algorithm | Time/Iteration | Convergence | Memory |
|-----------|----------------|-------------|--------|
| Value Iteration | O(\|S\|²\|A\|) | Guaranteed | O(\|S\|) |
| Policy Iteration | O(\|S\|³+\|S\|²\|A\|) | Finite steps | O(\|S\|) |
| Q-Learning | O(\|A\|) | Asymptotic* | O(\|S\|\|A\|) |

*under exploration conditions
