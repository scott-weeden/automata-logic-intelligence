# API Reference – Intelligent Systems Project

Phases 1–6 establish the reusable infrastructure and flagship algorithms for the
project.  The modules below now include problem abstractions, heuristic helpers,
search strategies, adversarial agents, Markov decision process utilities, and
reinforcement learning agents that later coursework builds upon.

---

## Search Module (`src/search/`)

### Problem Definitions

#### `class SearchProblem`
Abstract contract implemented by every domain-specific search problem.

```python
class SearchProblem:
    def get_start_state(self) -> Any
    def is_goal_state(self, state: Any) -> bool
    def get_successors(self, state: Any) -> Iterable[Tuple[Any, Any, float]]
    def get_cost_of_actions(self, actions: Sequence[Any]) -> float
```

#### `@dataclass class GridSearchProblem(SearchProblem)`
Grid-based navigation with 4-directional movement.

```python
GridSearchProblem(
    grid: Sequence[Sequence[Any]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
)
```

* Square cells equal to `1` or `'#'` are blocked.
* Validates rectangular shape plus accessible start/goal.
* Successors return `(state, action, 1.0)` where actions are one of
  `"UP"`, `"DOWN"`, `"LEFT"`, `"RIGHT"`.

### Tree Nodes and Queues

#### `@dataclass class Node`
Immutable search tree node with parent linkage, accumulated path cost, and
helpers for reconstructing solutions.

```python
node.state
node.parent
node.action
node.path_cost
node.depth

node.solution() -> List[Any]
node.path()     -> List[Any]
```

#### `class PriorityQueue`
Heap-backed queue supporting membership tests and keyed deletion—compatible with
the Berkeley Pac-Man infrastructure.

### Heuristics

```python
manhattan_distance(p: Iterable[float], q: Iterable[float]) -> float
chebyshev_distance(p: Iterable[float], q: Iterable[float]) -> float
euclidean_distance(p: Iterable[float], q: Iterable[float]) -> float
null_heuristic(state, problem=None) -> float
```

#### `class GridHeuristic`
Callable that selects the right metric for grid domains:
`'4-way'` → Manhattan, `'8-way'` → Chebyshev, `'continuous'` → Euclidean.

### Algorithms

Each agent stores `nodes_expanded` for instrumentation and exposes
`search(problem) -> Optional[List[Any]]`.

* `BreadthFirstSearch()` – complete and optimal for uniform step costs; explores
  level by level using a FIFO frontier.
* `DepthFirstSearch()` – space efficient depth-first tree search; may cycle on
  infinite graphs but prunes repeated states along the current path.
* `UniformCostSearch()` – Dijkstra-style search that minimises cumulative cost
  for strictly positive step costs.
* `AStarSearch(heuristic)` – best-first search ordered by `g + h`; optimal with
  admissible, consistent heuristics.
* `GreedyBestFirstSearch(heuristic)` – heuristic-only ordering (fast, not
  optimal).
* `IterativeDeepeningSearch()` – depth-first search with increasing depth limits
  that retains BFS optimality while using linear space.

---

## Games Module (`src/games/`)

### Core Abstractions

#### `class GameState`
Interface implemented by deterministic, perfect-information games.

```python
class GameState:
    def get_legal_actions(self, agent_index: int = 0) -> Sequence[Any]
    def generate_successor(self, agent_index: int, action: Any) -> GameState
    def is_terminal(self) -> bool
    def get_utility(self, agent_index: int = 0) -> float
```

#### `class GameAgent`
Stores the controlled player index plus `nodes_explored`; `reset_statistics()`
resets counters before each search.

#### `class Game`
Wrapper used in demos with methods `actions`, `result`, `terminal_test`,
`utility`, and `to_move`.

### Tic-Tac-Toe Reference Implementation

* `TicTacToeState` – immutable board state tracking last move and active player.
* `TicTacToe` – wires the state into the `Game` interface for sample
  applications and tests.

### Search-Based Agents

* `MinimaxAgent(index: int = 0, depth: int = 2)` – optimal two-player search.
* `AlphaBetaAgent(index: int = 0, depth: int = 2)` – minimax with pruning that
  reduces node expansions when move ordering is favourable.
* `ExpectimaxAgent(index: int = 0, depth: int = 2)` – models non-deterministic
  opponents via expectation instead of minimisation.

All agents invoke `reset_statistics()` at the start of `get_action()` and can be
used directly with `GameState` subclasses without additional wrappers.

---

## MDP Module (`src/mdp/`)

### Core Interfaces

#### `class MarkovDecisionProcess`
Abstract base class specifying the API consumed by planning algorithms.  Custom
MDPs inherit and override all six methods (`get_states`, `get_start_state`,
`get_possible_actions`, `get_transition_states_and_probs`, `get_reward`,
`is_terminal`).

#### `class MDP(MarkovDecisionProcess)`
Concrete helper for finite MDPs with optional stochastic transitions.

```python
MDP(
    init: State,
    actlist: Sequence | Mapping | Callable,
    terminals: Iterable[State],
    transitions: Mapping,
    reward: Mapping | Callable,
    gamma: float = 0.9,
)
```

* Accepts transition models either as `{state: {action: [(next, prob), ...]}}`
  or `{(state, action): [(next, prob), ...]}`.
* Rewards can be supplied as a mapping (`state` or `(state, action, next)` keys)
  or as a callable.
* Exposes convenience methods `actions(state)`, `T(state, action)`, and `R(state)`.

#### `@dataclass class GridMDP(MDP)`
Stochastic grid world mirroring the example from Russell & Norvig.

* Treats `None` cells as obstacles and numbers as per-state rewards.
* Actions are `(dr, dc)` offsets for the four cardinal directions; terminal
  states expose `[None]` as the only action.
* Transition model: 80% intended direction, 10% for each perpendicular deviation
  (falling back to the current cell if the move hits an obstacle).

### Dynamic-Programming Algorithms

#### `value_iteration(mdp, discount=0.9, iterations=100, tolerance=1e-6)`
Iteratively refines the state-value function until convergence or the iteration
limit is reached.  Returns a dictionary `{state: V(state)}`.

#### `policy_evaluation(mdp, policy, discount=0.9, tolerance=1e-6)`
Computes the value function for a fixed policy mapping `state -> action`.

#### `extract_policy(mdp, values, discount=0.9)`
Returns the greedy policy corresponding to a state-value function.

#### `ValueIterationAgent`
Runs `value_iteration` during initialization and caches both the value function
and the greedy policy.

```python
agent = ValueIterationAgent(mdp, discount=0.95, iterations=200)
agent.get_value(state)
agent.get_policy(state)
```

#### `PolicyIterationAgent`
Alternates between policy evaluation and improvement until convergence.
Stores both the converged value function and policy.

```python
agent = PolicyIterationAgent(mdp, discount=0.9)
agent.get_policy(state)
agent.get_value(state)
```

---

## Learning Module (`src/learning/`)

### Agents

#### `QLearningAgent`
Tabular Q-learning agent with epsilon-greedy exploration.

```python
agent = QLearningAgent(
    action_fn=lambda state: ['up', 'down', 'left', 'right'],
    discount=0.9,
    alpha=0.5,
    epsilon=0.1,
)

agent.start_episode()
action = agent.get_action(state)
agent.update(state, action, next_state, reward)
agent.stop_training()
```

* `q_values[(state, action)]` stores learned estimates.
* `get_action(state)` explores with probability `epsilon` while training and acts
  greedily otherwise.
* `episode_rewards` accumulates per-episode reward; reset via `start_episode()`.

#### `SARSAAgent`
On-policy temporal-difference control mirroring `QLearningAgent` but updating
with the value of the action actually taken next.  Set `agent.next_action`
before calling `agent.update(...)` when simulating the policy.

---

## Usage Examples

### Solving a Grid Pathfinding Task
```python
from src.search import GridSearchProblem, BreadthFirstSearch

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

print("Utility for X:", game.utility(state, player=0))
```

### Planning with Value Iteration
```python
from src.mdp import GridMDP, ValueIterationAgent

grid = [
    [-0.04, -0.04, -0.04, 1],
    [-0.04, None, -0.04, -1],
    [-0.04, -0.04, -0.04, -0.04],
]

mdp = GridMDP(grid, terminals={(0, 3), (1, 3)})
agent = ValueIterationAgent(mdp, discount=0.9, iterations=100)
print(agent.get_policy((0, 0)))
```

### Training a Q-Learning Agent
```python
from src.learning import QLearningAgent

actions = lambda state: ['up', 'down', 'left', 'right']
agent = QLearningAgent(actions, discount=0.9, alpha=0.5, epsilon=0.1)

state = (0, 0)
agent.start_episode()
for _ in range(100):
    action = agent.get_action(state)
    next_state = ...  # environment transition
    reward = ...
    agent.update(state, action, next_state, reward)
    state = next_state
```
