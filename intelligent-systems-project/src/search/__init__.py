"""
Search Module

Implements uninformed and informed search algorithms for AI problem solving.
Based on CS 5368 Weeks 1-4 covering search problem formulation and strategies.
"""

from .algorithms import (
    BreadthFirstSearch, DepthFirstSearch, UniformCostSearch,
    AStarSearch, GreedyBestFirstSearch, IterativeDeepeningSearch
)

from .problem import SearchProblem, GridSearchProblem

from .heuristics import (
    manhattan_distance,
    euclidean_distance,
    chebyshev_distance,
    GridHeuristic,
    null_heuristic
)

from .utils import Node, PriorityQueue

__all__ = [
    'BreadthFirstSearch', 'DepthFirstSearch', 'UniformCostSearch',
    'AStarSearch', 'GreedyBestFirstSearch', 'IterativeDeepeningSearch',
    'SearchProblem', 'GridSearchProblem',
    'manhattan_distance', 'euclidean_distance', 'chebyshev_distance',
    'GridHeuristic', 'null_heuristic',
    'Node', 'PriorityQueue'
]
