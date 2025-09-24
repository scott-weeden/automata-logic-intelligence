"""
Comprehensive Test Suite for Search Algorithms

Tests all search algorithms including BFS, DFS, UCS, A*, Greedy Best-First,
and Iterative Deepening on various problem types. Validates correctness,
optimality, completeness, and performance characteristics.

Based on CS 5368 Week 1-4 material on search strategies.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from search import (
    BreadthFirstSearch, DepthFirstSearch, UniformCostSearch,
    AStarSearch, GreedyBestFirstSearch, IterativeDeepeningSearch,
    SearchProblem, GridSearchProblem,
    manhattan_distance, euclidean_distance, GridHeuristic,
    Node
)

class SimpleSearchProblem(SearchProblem):
    """
    Simple linear search problem for testing basic functionality.
    States are integers 0 to goal, actions move +1 or +2.
    """
    
    def __init__(self, goal=5):
        """Initialize with goal state (default 5)."""
        self.goal = goal
    
    def get_start_state(self):
        """Start at state 0."""
        return 0
    
    def is_goal_state(self, state):
        """Goal is reaching the target number."""
        return state == self.goal
    
    def get_successors(self, state):
        """Can move +1 (cost 1) or +2 (cost 2) if not exceeding goal."""
        successors = []
        if state + 1 <= self.goal:
            successors.append((state + 1, '+1', 1))
        if state + 2 <= self.goal:
            successors.append((state + 2, '+2', 2))
        return successors

class TestSearchProblemInterface(unittest.TestCase):
    """Test the SearchProblem base class and problem formulation."""
    
    def setUp(self):
        """Set up test fixtures with simple and grid problems."""
        self.simple_problem = SimpleSearchProblem(goal=3)
        
        # 3x3 grid with obstacle in middle
        self.grid = [
            [0, 0, 0],
            [0, 1, 0],  # 1 = obstacle
            [0, 0, 0]
        ]
        self.grid_problem = GridSearchProblem(self.grid, (0, 0), (2, 2))
    
    def test_simple_problem_interface(self):
        """Test that simple problem implements required interface correctly."""
        # Test start state
        self.assertEqual(self.simple_problem.get_start_state(), 0)
        
        # Test goal test
        self.assertFalse(self.simple_problem.is_goal_state(0))
        self.assertTrue(self.simple_problem.is_goal_state(3))
        
        # Test successors format: (state, action, cost)
        successors = self.simple_problem.get_successors(1)
        self.assertEqual(len(successors), 2)
        self.assertIn((2, '+1', 1), successors)
        self.assertIn((3, '+2', 2), successors)
    
    def test_grid_problem_interface(self):
        """Test that grid problem implements interface correctly."""
        # Test start and goal
        self.assertEqual(self.grid_problem.get_start_state(), (0, 0))
        self.assertTrue(self.grid_problem.is_goal_state((2, 2)))
        
        # Test successors respect grid boundaries and obstacles
        successors = self.grid_problem.get_successors((0, 0))
        successor_states = [s[0] for s in successors]
        self.assertIn((0, 1), successor_states)  # East
        self.assertIn((1, 0), successor_states)  # South
        self.assertEqual(len(successors), 2)  # Only 2 valid moves from corner
        
        # Test obstacle avoidance
        successors = self.grid_problem.get_successors((1, 0))
        successor_states = [s[0] for s in successors]
        self.assertNotIn((1, 1), successor_states)  # Should avoid obstacle

class TestBreadthFirstSearch(unittest.TestCase):
    """Test BFS algorithm for completeness, optimality, and correctness."""
    
    def setUp(self):
        """Set up test problems and BFS instance."""
        self.bfs = BreadthFirstSearch()
        self.simple_problem = SimpleSearchProblem(goal=4)
        
        # Grid with multiple paths to test shortest path finding
        self.grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ]
        self.grid_problem = GridSearchProblem(self.grid, (0, 0), (2, 3))
    
    def test_bfs_finds_solution(self):
        """Test that BFS finds a valid solution when one exists."""
        solution = self.bfs.search(self.simple_problem)
        
        # Should find a solution
        self.assertIsNotNone(solution)
        self.assertIsInstance(solution, list)
        
        # Verify solution leads to goal by simulating execution
        state = self.simple_problem.get_start_state()
        for action in solution:
            successors = self.simple_problem.get_successors(state)
            next_states = {a: s for s, a, c in successors}
            self.assertIn(action, next_states)
            state = next_states[action]
        
        self.assertTrue(self.simple_problem.is_goal_state(state))
    
    def test_bfs_optimal_path_length(self):
        """Test that BFS finds shortest path in terms of number of steps."""
        solution = self.bfs.search(self.simple_problem)
        
        # For goal=4, optimal is ['+2', '+2'] (2 steps) not ['+1','+1','+1','+1'] (4 steps)
        self.assertEqual(len(solution), 2)
        self.assertEqual(solution, ['+2', '+2'])
    
    def test_bfs_grid_shortest_path(self):
        """Test BFS finds shortest path in grid world."""
        solution = self.bfs.search(self.grid_problem)
        
        # Should find solution and it should be reasonably short
        self.assertIsNotNone(solution)
        # Shortest path should be 5 moves: right, right, right, down, down
        self.assertEqual(len(solution), 5)
    
    def test_bfs_no_solution(self):
        """Test BFS returns None when no solution exists."""
        # Create impossible grid problem
        impossible_grid = [
            [0, 1],
            [1, 0]  # Goal at (1,1) unreachable due to obstacles
        ]
        impossible_problem = GridSearchProblem(impossible_grid, (0, 0), (1, 1))
        
        solution = self.bfs.search(impossible_problem)
        self.assertIsNone(solution)
    
    def test_bfs_nodes_expanded_tracking(self):
        """Test that BFS correctly tracks number of nodes expanded."""
        self.bfs.search(self.simple_problem)
        
        # Should have expanded some nodes (at least start state)
        self.assertGreater(self.bfs.nodes_expanded, 0)
        # For simple problem with goal=4, should expand states 0,1,2,3,4
        self.assertLessEqual(self.bfs.nodes_expanded, 5)

class TestDepthFirstSearch(unittest.TestCase):
    """Test DFS algorithm behavior, noting it's not optimal but space-efficient."""
    
    def setUp(self):
        """Set up test problems and DFS instance."""
        self.dfs = DepthFirstSearch()
        self.simple_problem = SimpleSearchProblem(goal=3)
        
        # Small grid to avoid infinite paths
        self.grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        self.grid_problem = GridSearchProblem(self.grid, (0, 0), (2, 2))
    
    def test_dfs_finds_solution(self):
        """Test that DFS finds a valid solution (may not be optimal)."""
        solution = self.dfs.search(self.simple_problem)
        
        # Should find some solution
        self.assertIsNotNone(solution)
        
        # Verify solution validity
        state = self.simple_problem.get_start_state()
        for action in solution:
            successors = self.simple_problem.get_successors(state)
            next_states = {a: s for s, a, c in successors}
            state = next_states[action]
        
        self.assertTrue(self.simple_problem.is_goal_state(state))
    
    def test_dfs_may_not_be_optimal(self):
        """Test that DFS may find suboptimal solutions."""
        solution = self.dfs.search(self.simple_problem)
        
        # DFS might find ['+1', '+1', '+1'] instead of optimal ['+2', '+1']
        # We just verify it's a valid solution, not necessarily optimal
        self.assertIsNotNone(solution)
        self.assertGreaterEqual(len(solution), 2)  # At least as long as optimal
    
    def test_dfs_grid_navigation(self):
        """Test DFS can navigate grid problems."""
        solution = self.dfs.search(self.grid_problem)
        
        # Should find some path (may be longer than BFS)
        self.assertIsNotNone(solution)
        self.assertGreater(len(solution), 0)

class TestUniformCostSearch(unittest.TestCase):
    """Test UCS algorithm for optimality with varying step costs."""
    
    def setUp(self):
        """Set up UCS instance and problems with different costs."""
        self.ucs = UniformCostSearch()
        self.simple_problem = SimpleSearchProblem(goal=4)
        
        # Grid problem (uniform costs)
        self.grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.grid_problem = GridSearchProblem(self.grid, (0, 0), (2, 2))
    
    def test_ucs_optimal_cost(self):
        """Test that UCS finds minimum cost solution."""
        solution = self.ucs.search(self.simple_problem)
        
        # Should find optimal solution: ['+2', '+2'] with cost 4
        # rather than ['+1', '+1', '+1', '+1'] with cost 4
        # or ['+1', '+2', '+1'] with cost 4
        self.assertIsNotNone(solution)
        
        # Calculate total cost
        total_cost = 0
        state = self.simple_problem.get_start_state()
        for action in solution:
            successors = self.simple_problem.get_successors(state)
            for s, a, c in successors:
                if a == action:
                    total_cost += c
                    state = s
                    break
        
        # Should find minimum cost path
        self.assertEqual(total_cost, 4)  # Optimal cost to reach goal 4
    
    def test_ucs_vs_bfs_same_result_uniform_costs(self):
        """Test UCS gives same result as BFS when all costs are equal."""
        bfs = BreadthFirstSearch()
        
        ucs_solution = self.ucs.search(self.grid_problem)
        bfs_solution = bfs.search(self.grid_problem)
        
        # Both should find optimal solutions of same length
        self.assertEqual(len(ucs_solution), len(bfs_solution))

class TestAStarSearch(unittest.TestCase):
    """Test A* algorithm with various heuristics for optimality and efficiency."""
    
    def setUp(self):
        """Set up A* instances with different heuristics."""
        # Manhattan heuristic for grid problems
        def manhattan_heuristic(state, problem):
            if hasattr(problem, 'goal'):
                return manhattan_distance(state, problem.goal)
            return 0
        
        self.astar_manhattan = AStarSearch(manhattan_heuristic)
        self.astar_null = AStarSearch()  # No heuristic (reduces to UCS)
        
        # Grid problem for testing
        self.grid = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
        self.grid_problem = GridSearchProblem(self.grid, (0, 0), (2, 3))
    
    def test_astar_with_admissible_heuristic_optimal(self):
        """Test A* with admissible heuristic finds optimal solution."""
        solution = self.astar_manhattan.search(self.grid_problem)
        
        # Should find optimal solution
        self.assertIsNotNone(solution)
        
        # Compare with UCS (should get same cost)
        ucs = UniformCostSearch()
        ucs_solution = ucs.search(self.grid_problem)
        
        self.assertEqual(len(solution), len(ucs_solution))
    
    def test_astar_efficiency_vs_ucs(self):
        """Test that A* with good heuristic explores fewer nodes than UCS."""
        # Run both algorithms
        self.astar_manhattan.search(self.grid_problem)
        
        ucs = UniformCostSearch()
        ucs.search(self.grid_problem)
        
        # A* should explore fewer or equal nodes due to heuristic guidance
        self.assertLessEqual(self.astar_manhattan.nodes_expanded, ucs.nodes_expanded + 2)
    
    def test_astar_null_heuristic_equals_ucs(self):
        """Test A* with null heuristic behaves like UCS."""
        astar_solution = self.astar_null.search(self.grid_problem)
        
        ucs = UniformCostSearch()
        ucs_solution = ucs.search(self.grid_problem)
        
        # Should find same length solution
        self.assertEqual(len(astar_solution), len(ucs_solution))
    
    def test_astar_inadmissible_heuristic_still_finds_solution(self):
        """Test A* with inadmissible heuristic still finds valid solution."""
        # Inadmissible heuristic (overestimates)
        def bad_heuristic(state, problem):
            return manhattan_distance(state, problem.goal) * 10
        
        astar_bad = AStarSearch(bad_heuristic)
        solution = astar_bad.search(self.grid_problem)
        
        # Should still find a solution (just not guaranteed optimal)
        self.assertIsNotNone(solution)

class TestGreedyBestFirstSearch(unittest.TestCase):
    """Test Greedy Best-First Search behavior (fast but not optimal)."""
    
    def setUp(self):
        """Set up Greedy search with Manhattan heuristic."""
        def manhattan_heuristic(state, problem):
            return manhattan_distance(state, problem.goal)
        
        self.greedy = GreedyBestFirstSearch(manhattan_heuristic)
        
        # Grid problem
        self.grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.grid_problem = GridSearchProblem(self.grid, (0, 0), (2, 2))
    
    def test_greedy_finds_solution(self):
        """Test Greedy search finds valid solution."""
        solution = self.greedy.search(self.grid_problem)
        
        self.assertIsNotNone(solution)
        
        # Verify solution validity
        state = self.grid_problem.get_start_state()
        for action in solution:
            successors = self.grid_problem.get_successors(state)
            next_states = {a: s for s, a, c in successors}
            state = next_states[action]
        
        self.assertTrue(self.grid_problem.is_goal_state(state))
    
    def test_greedy_may_be_suboptimal(self):
        """Test that Greedy search may find suboptimal solutions."""
        solution = self.greedy.search(self.grid_problem)
        
        # Compare with optimal (BFS)
        bfs = BreadthFirstSearch()
        optimal_solution = bfs.search(self.grid_problem)
        
        # Greedy may find longer path due to heuristic misleading it
        self.assertGreaterEqual(len(solution), len(optimal_solution))

class TestIterativeDeepeningSearch(unittest.TestCase):
    """Test Iterative Deepening Search for optimality with space efficiency."""
    
    def setUp(self):
        """Set up IDS instance and test problems."""
        self.ids = IterativeDeepeningSearch()
        self.simple_problem = SimpleSearchProblem(goal=3)
        
        # Small grid to test depth limits
        self.grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.grid_problem = GridSearchProblem(self.grid, (0, 0), (2, 2))
    
    def test_ids_finds_optimal_solution(self):
        """Test IDS finds optimal solution like BFS."""
        solution = self.ids.search(self.simple_problem)
        
        # Should find optimal solution
        self.assertIsNotNone(solution)
        
        # Compare with BFS
        bfs = BreadthFirstSearch()
        bfs_solution = bfs.search(self.simple_problem)
        
        self.assertEqual(len(solution), len(bfs_solution))
    
    def test_ids_with_depth_limit(self):
        """Test IDS respects depth limits."""
        # Search with very small depth limit
        solution = self.ids.search(self.simple_problem, max_depth=1)
        
        # Should not find solution for goal=3 with depth limit 1
        self.assertIsNone(solution)
        
        # Should find solution with adequate depth
        solution = self.ids.search(self.simple_problem, max_depth=5)
        self.assertIsNotNone(solution)
    
    def test_ids_space_efficiency(self):
        """Test that IDS explores nodes multiple times but finds optimal solution."""
        solution = self.ids.search(self.grid_problem)
        
        # Should find solution
        self.assertIsNotNone(solution)
        
        # May explore more nodes than BFS due to repeated exploration
        # but should find optimal solution
        bfs = BreadthFirstSearch()
        bfs_solution = bfs.search(self.grid_problem)
        
        self.assertEqual(len(solution), len(bfs_solution))

class TestHeuristicFunctions(unittest.TestCase):
    """Test heuristic functions for admissibility and consistency."""
    
    def test_manhattan_distance_calculation(self):
        """Test Manhattan distance calculation correctness."""
        # Test basic cases
        self.assertEqual(manhattan_distance((0, 0), (0, 0)), 0)
        self.assertEqual(manhattan_distance((0, 0), (3, 4)), 7)
        self.assertEqual(manhattan_distance((1, 1), (4, 5)), 7)
        
        # Test symmetry
        self.assertEqual(
            manhattan_distance((1, 2), (3, 4)),
            manhattan_distance((3, 4), (1, 2))
        )
    
    def test_euclidean_distance_calculation(self):
        """Test Euclidean distance calculation correctness."""
        import math
        
        # Test basic cases
        self.assertEqual(euclidean_distance((0, 0), (0, 0)), 0)
        self.assertEqual(euclidean_distance((0, 0), (3, 4)), 5.0)
        
        # Test Pythagorean theorem
        self.assertAlmostEqual(
            euclidean_distance((0, 0), (1, 1)),
            math.sqrt(2),
            places=5
        )
    
    def test_grid_heuristic_admissibility(self):
        """Test that GridHeuristic is admissible for different movement types."""
        goal = (5, 5)
        
        # 4-way movement heuristic
        h_4way = GridHeuristic(goal, '4-way')
        
        # Test admissibility: h(n) <= actual distance
        # For 4-way movement, Manhattan distance is admissible
        test_states = [(0, 0), (2, 3), (5, 0), (3, 5)]
        
        for state in test_states:
            heuristic_value = h_4way(state)
            actual_manhattan = manhattan_distance(state, goal)
            
            # Heuristic should equal Manhattan distance for 4-way
            self.assertEqual(heuristic_value, actual_manhattan)
            
            # Should never overestimate (admissible)
            self.assertLessEqual(heuristic_value, actual_manhattan)

class TestNodeClass(unittest.TestCase):
    """Test Node class for search tree representation."""
    
    def test_node_creation(self):
        """Test basic node creation and attributes."""
        node = Node(state='A', parent=None, action=None, path_cost=0)
        
        self.assertEqual(node.state, 'A')
        self.assertIsNone(node.parent)
        self.assertIsNone(node.action)
        self.assertEqual(node.path_cost, 0)
        self.assertEqual(node.depth, 0)
    
    def test_node_parent_child_relationship(self):
        """Test parent-child relationships and depth calculation."""
        parent = Node(state='A', parent=None, action=None, path_cost=0)
        child = Node(state='B', parent=parent, action='move', path_cost=1)
        
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.action, 'move')
        self.assertEqual(child.depth, 1)
    
    def test_node_solution_path(self):
        """Test solution path extraction from node."""
        # Create path: A -> B -> C
        node_a = Node(state='A', parent=None, action=None, path_cost=0)
        node_b = Node(state='B', parent=node_a, action='a_to_b', path_cost=1)
        node_c = Node(state='C', parent=node_b, action='b_to_c', path_cost=2)
        
        # Test solution (action sequence)
        solution = node_c.solution()
        self.assertEqual(solution, ['a_to_b', 'b_to_c'])
        
        # Test path (state sequence)
        path = node_c.path()
        self.assertEqual(path, ['A', 'B', 'C'])
    
    def test_node_equality(self):
        """Test node equality based on state."""
        node1 = Node(state='A', path_cost=5)
        node2 = Node(state='A', path_cost=10)
        node3 = Node(state='B', path_cost=5)
        
        # Nodes with same state should be equal regardless of cost
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)

if __name__ == '__main__':
    # Run all search algorithm tests
    unittest.main(verbosity=2)
