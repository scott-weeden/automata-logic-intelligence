# API Reference – Intelligent Systems Project

This reference summarises the core interfaces implemented in Phases 1 and 2 of
the project.  These modules provide the reusable abstractions and utilities that
later algorithms (search strategies, MDP solvers, reinforcement learning) build
upon.

---

## Search Module (`src/search/`)

### Problem Definitions

#### `class SearchProblem`
Base class describing the contract every search problem must implement.

```python
class SearchProblem:
    def get_start_state(self) -> Any
    def is_goal_state(self, state: Any) -> bool
    def get_successors(self, state: Any) -> Iterable[Tuple[Any, Any, float]]
    def get_cost_of_actions(self, actions: Sequence[Any]) -> float
```

* `get_start_state()` – return the initial state.
* `is_goal_state(state)` – true when `state` satisfies the goal condition.
* `get_successors(state)` – iterable of `(successor, action, step_cost)` triples.
* `get_cost_of_actions(actions)` – total path cost for a proposed action list
  (defaults to unit cost per action and performs input validation).

#### `@dataclass class GridSearchProblem(SearchProblem)`
Concrete 4-connected grid world.

```python
GridSearchProblem(
    grid: Sequence[Sequence[Any]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
)
```

* Accepts any rectangular grid; values equal to `1` or `'#'` are obstacles.
* Validates that `start` and `goal` reside inside the grid and are traversable.
* `get_successors(state)` returns legal neighbours with action names
  `"North"`, `"South"`, `"West"`, `"East"` and unit step cost.

### Tree Nodes and Queues

#### `@dataclass class Node`
Search tree node storing a state, optional parent/action, accumulated cost, and
depth.  Convenience helpers expedite common algorithmic patterns.

```python
node.state      # state value
node.parent     # parent Node or None
node.action     # action from parent into this node
node.path_cost  # cumulative g(n)
node.depth      # depth in the tree (root = 0)

node.expand(problem)    -> List[Node]
node.solution()         -> List[Any]   # actions from root to node
node.path()             -> List[Any]   # states from root to node
```

`Node` comparisons order by `path_cost`, allowing use inside priority queues.

#### `class PriorityQueue`
Heap-backed priority queue compatible with the Berkeley Pac-Man projects.

```python
PriorityQueue(order: Literal['min','max'] = 'min', f: Callable[[Any], float] = lambda x: x)

pq.append(item)
pq.extend(iterable)
pq.pop() -> item
len(pq) -> int

key in pq           # membership check
pq[key]             # retrieve matching item
del pq[key]         # delete matching item
```

Membership tests understand common patterns (tuples like `(priority, state)` or
objects exposing a `state` attribute) so queues remain ergonomic in search code.

### Heuristics

```python
manhattan_distance(p: Iterable[float], q: Iterable[float]) -> float
chebyshev_distance(p: Iterable[float], q: Iterable[float]) -> float
euclidean_distance(p: Iterable[float], q: Iterable[float]) -> float
null_heuristic(state, problem=None) -> float
```

#### `class GridHeuristic`
Callable wrapper that selects the appropriate distance metric for grid-based
problems.

```python
GridHeuristic(goal, movement_type='4-way')(state, problem=None) -> float
```

* `'4-way'` → Manhattan distance
* `'8-way'` → Chebyshev distance
* `'continuous'` → Euclidean distance

---

## Games Module (`src/games/`)

### Core Abstractions

#### `class GameState`
Abstract interface required by adversarial search algorithms.

```python
class GameState:
    def get_legal_actions(self, agent_index: int = 0) -> Sequence[Any]
    def generate_successor(self, agent_index: int, action: Any) -> GameState
    def is_terminal(self) -> bool
    def get_utility(self, agent_index: int = 0) -> float
```

#### `class GameAgent`
Base class providing the `index` of the controlled player and a
`nodes_explored` counter for instrumentation.

```python
class GameAgent:
    index: int
    nodes_explored: int
    def reset_statistics(self) -> None
    def get_action(self, game_state: GameState): ...
```

#### `class Game`
Lightweight wrapper used by demo applications.

```python
class Game:
    def actions(self, state: GameState) -> Sequence[Any]
    def result(self, state: GameState, action: Any) -> GameState
    def terminal_test(self, state: GameState) -> bool
    def utility(self, state: GameState, player: int) -> float
    def to_move(self, state: GameState) -> int
```

### Tic-Tac-Toe Reference Implementation

#### `@dataclass class TicTacToeState(GameState)`
Immutable 3×3 board representation.

* `board`: tuple of tuples holding `'X'`, `'O'`, or `' '`.
* `to_move`: `'X'` or `'O'` indicating whose turn it is.
* `last_move`: optional `(row, col)` of the move that produced this state.
* Provides concrete implementations of all `GameState` methods.

#### `class TicTacToe(Game)`
Exposes the Tic-Tac-Toe domain through the `Game` interface.

* `initial`: starting `TicTacToeState` with an empty board.
* `actions(state)` delegates to `state.get_legal_actions`.
* `result(state, action)` validates the move and returns a new state.
* `utility(state, player)` forwards to the state's utility function.

### Search-Based Agents

The minimax family of agents operate on any `GameState` implementation.

* `MinimaxAgent(index: int = 0, depth: int = 2)` – deterministic optimal play.
* `AlphaBetaAgent(index: int = 0, depth: int = 2)` – minimax with alpha-beta
  pruning for improved efficiency.
* `ExpectimaxAgent(index: int = 0, depth: int = 2)` – handles chance/opponent
  randomness via expectation instead of minimisation.

All agents expose `get_action(game_state)` and maintain `nodes_explored` for
profiling.

---

## Usage Examples

### Solving a Grid Pathfinding Task
```python
from src.search import GridSearchProblem, BreadthFirstSearch, manhattan_distance

problem = GridSearchProblem(
    grid=[
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ],
    start=(0, 0),
    goal=(2, 2),
)

agent = BreadthFirstSearch()
solution = agent.search(problem)
print("Solution actions:", solution)
```

### Playing Tic-Tac-Toe with Minimax
```python
from src.games import TicTacToe, MinimaxAgent

game = TicTacToe()
agent = MinimaxAgent(index=0, depth=4)

state = game.initial
while not game.terminal_test(state):
    action = agent.get_action(state)
    state = game.result(state, action)
    # Here you could alternate with a human or another agent

print("Utility for X:", game.utility(state, player=0))
```
