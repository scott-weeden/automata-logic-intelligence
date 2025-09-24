"""
Search Module

Implements uninformed and informed search algorithms for AI problem solving.
Based on CS 5368 Weeks 1-4 covering search problem formulation and strategies.
"""

from .algorithms import (
    breadth_first_search,
    depth_first_search, 
    uniform_cost_search,
    astar_search,
    greedy_best_first_search
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
    'breadth_first_search', 'depth_first_search', 'uniform_cost_search',
    'astar_search', 'greedy_best_first_search',
    'SearchProblem', 'GridSearchProblem',
    'manhattan_distance', 'euclidean_distance', 'chebyshev_distance',
    'GridHeuristic', 'null_heuristic',
    'Node', 'PriorityQueue'
]
