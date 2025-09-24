# Search Algorithms

## Overview

Search algorithms find paths from initial states to goal states. This module implements uninformed and informed search strategies with performance tracking.

## Uninformed Search

### Breadth-First Search (BFS)
- **Strategy**: FIFO queue, explores shallowest nodes first
- **Complete**: Yes (finite branching factor)
- **Optimal**: Yes (unit step costs)
- **Time**: O(b^d), **Space**: O(b^d)

```python
from search.algorithms import BreadthFirstSearch

bfs = BreadthFirstSearch()
solution = bfs.search(problem)
print(f"Nodes expanded: {bfs.nodes_expanded}")
```

### Depth-First Search (DFS)
- **Strategy**: LIFO stack, explores deepest nodes first
- **Complete**: No (infinite paths)
- **Optimal**: No
- **Time**: O(b^m), **Space**: O(bm)

```python
from search.algorithms import DepthFirstSearch

dfs = DepthFirstSearch()
solution = dfs.search(problem)
```

### Uniform Cost Search (UCS)
- **Strategy**: Priority queue by path cost g(n)
- **Complete**: Yes (step costs ≥ ε > 0)
- **Optimal**: Yes
- **Time**: O(b^(1+⌊C*/ε⌋)), **Space**: O(b^(1+⌊C*/ε⌋))

```python
from search.algorithms import UniformCostSearch

ucs = UniformCostSearch()
solution = ucs.search(problem)
```

## Informed Search

### A* Search
- **Strategy**: f(n) = g(n) + h(n) evaluation function
- **Complete**: Yes
- **Optimal**: Yes (admissible heuristic)
- **Performance**: Optimal among optimal algorithms

```python
from search.algorithms import AStarSearch
from search.heuristics import manhattan_distance

astar = AStarSearch(heuristic=manhattan_distance)
solution = astar.search(problem)
```

### Greedy Best-First Search
- **Strategy**: f(n) = h(n) only
- **Complete**: No
- **Optimal**: No
- **Performance**: Fast but can be misled

```python
from search.algorithms import GreedyBestFirstSearch

greedy = GreedyBestFirstSearch(heuristic=manhattan_distance)
solution = greedy.search(problem)
```

## Heuristic Functions

### Manhattan Distance
```python
def manhattan_distance(state, problem):
    x1, y1 = state
    x2, y2 = problem.goal
    return abs(x1 - x2) + abs(y1 - y2)
```

### Euclidean Distance
```python
def euclidean_distance(state, problem):
    x1, y1 = state
    x2, y2 = problem.goal
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
```

## Performance Comparison

| Algorithm | Complete | Optimal | Time | Space | Best Use Case |
|-----------|----------|---------|------|-------|---------------|
| BFS | Yes | Yes* | O(b^d) | O(b^d) | Shortest path |
| DFS | No | No | O(b^m) | O(bm) | Memory limited |
| UCS | Yes | Yes | O(b^C*) | O(b^C*) | Variable costs |
| A* | Yes | Yes* | O(b^d) | O(b^d) | With good heuristic |
| Greedy | No | No | O(b^m) | O(b^m) | Fast approximation |

*Optimal under specific conditions

## Example Usage

```python
from search.problem import GridSearchProblem
from search.algorithms import AStarSearch
from search.heuristics import manhattan_distance

# Create problem
grid = [[0, 0, 1], [0, 1, 0], [0, 0, 0]]
problem = GridSearchProblem(grid, (0,0), (2,2))

# Solve with A*
astar = AStarSearch(manhattan_distance)
solution = astar.search(problem)

print(f"Solution: {solution}")
print(f"Nodes expanded: {astar.nodes_expanded}")
```

## Key Concepts

- **Admissible heuristic**: h(n) ≤ h*(n) (never overestimates)
- **Consistent heuristic**: h(n) ≤ c(n,a,n') + h(n') (monotonic)
- **Node expansion**: Generating successors from current node
- **Frontier**: Set of nodes waiting to be expanded
- **Explored set**: Nodes already expanded (graph search)
