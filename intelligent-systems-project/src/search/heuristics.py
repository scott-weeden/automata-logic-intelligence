"""
Heuristic Functions for Informed Search

Implements common heuristics for A* and greedy search.
Based on CS 5368 Week 3-4 material on admissible and consistent heuristics.

Key properties:
- Admissible: h(n) ≤ h*(n) (never overestimate)
- Consistent: h(n) ≤ c(n,a,n') + h(n') (triangle inequality)
"""

import math

def manhattan_distance(pos1, pos2):
    """
    Manhattan (L1) distance between two positions.
    Admissible for grid worlds with 4-directional movement.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance(pos1, pos2):
    """
    Euclidean (L2) distance between two positions.
    Admissible for continuous spaces with straight-line movement.
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def chebyshev_distance(pos1, pos2):
    """
    Chebyshev (L∞) distance between two positions.
    Admissible for grid worlds with 8-directional movement.
    """
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

class GridHeuristic:
    """
    Heuristic functions for grid-based pathfinding problems.
    Automatically selects appropriate distance metric.
    """
    
    def __init__(self, goal, movement_type='4-way'):
        """
        Initialize grid heuristic.
        goal: target position (row, col)
        movement_type: '4-way', '8-way', or 'continuous'
        """
        self.goal = goal
        self.movement_type = movement_type
    
    def __call__(self, state):
        """Compute heuristic value for given state."""
        if self.movement_type == '4-way':
            return manhattan_distance(state, self.goal)
        elif self.movement_type == '8-way':
            return chebyshev_distance(state, self.goal)
        elif self.movement_type == 'continuous':
            return euclidean_distance(state, self.goal)
        else:
            raise ValueError(f"Unknown movement type: {self.movement_type}")

def null_heuristic(state):
    """
    Null heuristic that always returns 0.
    Reduces A* to uniform cost search.
    """
    return 0

def max_heuristic(*heuristics):
    """
    Return maximum of multiple heuristics.
    Result is admissible if all input heuristics are admissible.
    """
    def combined_heuristic(state):
        return max(h(state) for h in heuristics)
    return combined_heuristic

def weighted_heuristic(heuristic, weight=1.0):
    """
    Scale heuristic by constant weight.
    weight > 1 makes search faster but potentially suboptimal.
    """
    def weighted_h(state):
        return weight * heuristic(state)
    return weighted_h

class PatternDatabase:
    """
    Pattern database heuristic for complex domains.
    Precomputes exact distances for simplified problem instances.
    """
    
    def __init__(self, pattern_problem):
        """
        Initialize pattern database.
        pattern_problem: simplified version of original problem
        """
        self.pattern_problem = pattern_problem
        self.database = {}
        self._build_database()
    
    def _build_database(self):
        """Build pattern database using backward search from goal."""
        from .algorithms import breadth_first_search
        # Implementation would depend on specific pattern abstraction
        pass
    
    def __call__(self, state):
        """Look up heuristic value in precomputed database."""
        pattern_state = self._abstract_state(state)
        return self.database.get(pattern_state, 0)
    
    def _abstract_state(self, state):
        """Convert full state to pattern state."""
        # Implementation depends on specific abstraction
        return state
