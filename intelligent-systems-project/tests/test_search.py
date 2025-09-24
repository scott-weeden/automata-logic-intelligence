"""
Test Suite for Search Algorithms

Tests uninformed and informed search strategies on various problems.
Validates correctness, optimality, and completeness properties.
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from search import (
    breadth_first_search, depth_first_search, uniform_cost_search, astar_search,
    GridSearchProblem, manhattan_distance, GridHeuristic
)

class TestSearchAlgorithms:
    """Test search algorithms on grid pathfinding problems."""
    
    def setup_method(self):
        """Set up test grid and problems."""
        # Simple 3x3 grid: 0=free, 1=obstacle
        self.grid = [
            [0, 0, 1],
            [0, 1, 0], 
            [0, 0, 0]
        ]
        self.start = (0, 0)
        self.goal = (2, 2)
        self.problem = GridSearchProblem(self.grid, self.start, self.goal)
    
    def test_bfs_finds_solution(self):
        """Test that BFS finds a solution."""
        solution = breadth_first_search(self.problem)
        assert solution is not None
        assert len(solution) > 0
    
    def test_dfs_finds_solution(self):
        """Test that DFS finds a solution.""" 
        solution = depth_first_search(self.problem)
        assert solution is not None
        assert len(solution) > 0
    
    def test_ucs_finds_solution(self):
        """Test that UCS finds optimal solution."""
        solution = uniform_cost_search(self.problem)
        assert solution is not None
        # UCS should find optimal path length
        assert len(solution) == 4  # Shortest path in this grid
    
    def test_astar_finds_optimal_solution(self):
        """Test that A* with admissible heuristic finds optimal solution."""
        heuristic = GridHeuristic(self.goal, '4-way')
        solution = astar_search(self.problem, heuristic)
        assert solution is not None
        assert len(solution) == 4  # Should match UCS
    
    def test_no_solution_case(self):
        """Test behavior when no solution exists."""
        # Grid with goal blocked off
        blocked_grid = [
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 0]  # Goal at (2,2) is unreachable
        ]
        blocked_problem = GridSearchProblem(blocked_grid, (0,0), (2,2))
        
        assert breadth_first_search(blocked_problem) is None
        assert depth_first_search(blocked_problem) is None
        assert uniform_cost_search(blocked_problem) is None

class TestHeuristics:
    """Test heuristic functions for admissibility."""
    
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        assert manhattan_distance((0,0), (3,4)) == 7
        assert manhattan_distance((1,1), (1,1)) == 0
        assert manhattan_distance((0,0), (0,5)) == 5
    
    def test_grid_heuristic_admissible(self):
        """Test that grid heuristic is admissible."""
        goal = (2, 2)
        heuristic = GridHeuristic(goal, '4-way')
        
        # Heuristic should never overestimate
        assert heuristic((0, 0)) <= 4  # True distance is 4
        assert heuristic((1, 1)) <= 2  # True distance is 2
        assert heuristic((2, 2)) == 0  # At goal

if __name__ == '__main__':
    pytest.main([__file__])
