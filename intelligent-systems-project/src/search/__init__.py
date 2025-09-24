"""Search package exposing foundational problem definitions, utilities, heuristics,
   and search agents.

The first two implementation phases focus on reusable components that higher-level
algorithms rely on:

* :mod:`src.search.problem` supplies abstract base classes describing the search
  interface and a concrete grid example used in tests and demos.
* :mod:`src.search.utils` contains the :class:`Node` helper and a robust
  :class:`PriorityQueue` abstraction used by informed searches.
* :mod:`src.search.heuristics` implements standard distance heuristics together
  with a configurable :class:`GridHeuristic` wrapper.

Later phases add algorithmic agents which we re-export here for convenience once
available.
"""

from .problem import GridSearchProblem, SearchProblem
from .utils import Node, PriorityQueue
from .heuristics import (
    manhattan_distance,
    euclidean_distance,
    chebyshev_distance,
    GridHeuristic,
    null_heuristic,
)
from .algorithms import (
    SearchAgent,
    BreadthFirstSearch,
    DepthFirstSearch,
    UniformCostSearch,
    AStarSearch,
    GreedyBestFirstSearch,
    IterativeDeepeningSearch,
)

__all__ = [
    "SearchProblem",
    "GridSearchProblem",
    "Node",
    "PriorityQueue",
    "manhattan_distance",
    "euclidean_distance",
    "chebyshev_distance",
    "GridHeuristic",
    "null_heuristic",
    "SearchAgent",
    "BreadthFirstSearch",
    "DepthFirstSearch",
    "UniformCostSearch",
    "AStarSearch",
    "GreedyBestFirstSearch",
    "IterativeDeepeningSearch",
]
