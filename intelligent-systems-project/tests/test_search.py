"""
Comprehensive test suite for search module
Tests all search algorithms and utility classes
"""

import pytest
import sys
import os
from typing import List, Tuple, Any
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search import (
    SearchProblem, GridSearchProblem, SearchAgent,
    BreadthFirstSearch, DepthFirstSearch, UniformCostSearch,
    AStarSearch, GreedyBestFirstSearch, IterativeDeepeningSearch,
    Node, PriorityQueue,
    manhattan_distance, euclidean_distance, chebyshev_distance,
    null_heuristic, GridHeuristic
)


# Test Problems
class SimpleProblem(SearchProblem):
    """Simple graph problem for testing"""
    
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal
    
    def get_start_state(self):
        return self.start
    
    def is_goal_state(self, state):
        return state == self.goal
    
    def get_successors(self, state):
        return self.graph.get(state, [])
    
    def get_cost_of_actions(self, actions):
        return len(actions)


class TestSearchProblem:
    """Test SearchProblem base class and implementations"""
    
    def test_grid_search_problem_initialization(self):
        """Test GridSearchProblem initialization"""
        grid = [[0, 0, 1], [0, 1, 0], [0, 0, 0]]
        problem = GridSearchProblem(grid, (0, 0), (2, 2))
        
        assert problem.get_start_state() == (0, 0)
        assert problem.is_goal_state((2, 2)) == True
        assert problem.is_goal_state((0, 0)) == False
    
    def test_grid_successors_basic(self):
        """Test basic successor generation"""
        grid = [[0, 0], [0, 0]]
        problem = GridSearchProblem(grid, (0, 0), (1, 1))
        
        successors = problem.get_successors((0, 0))
        states = [s[0] for s in successors]
        
        assert (1, 0) in states  # DOWN
        assert (0, 1) in states  # RIGHT
        assert len(states) == 2
    
    def test_grid_successors_with_obstacles(self):
        """Test successor generation with obstacles"""
        grid = [[0, 1], [0, 0]]
        problem = GridSearchProblem(grid, (0, 0), (1, 1))
        
        successors = problem.get_successors((0, 0))
        states = [s[0] for s in successors]
        
        assert (0, 1) not in states  # Blocked by obstacle
        assert (1, 0) in states
    
    def test_grid_successors_boundary(self):
        """Test successor generation at boundaries"""
        grid = [[0, 0], [0, 0]]
        problem = GridSearchProblem(grid, (0, 0), (1, 1))
        
        # Corner position
        successors = problem.get_successors((1, 1))
        states = [s[0] for s in successors]
        
        assert (0, 1) in states  # UP
        assert (1, 0) in states  # LEFT
        assert len(states) == 2
    
    def test_cost_of_actions(self):
        """Test action cost calculation"""
        grid = [[0, 0], [0, 0]]
        problem = GridSearchProblem(grid, (0, 0), (1, 1))
        
        actions = ['DOWN', 'RIGHT']
        cost = problem.get_cost_of_actions(actions)
        assert cost == 2


class TestSearchAlgorithms:
    """Test all search algorithm implementations"""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph"""
        return {
            'A': [('B', 'go_B', 1), ('C', 'go_C', 2)],
            'B': [('D', 'go_D', 3)],
            'C': [('D', 'go_D', 1)],
            'D': []
        }
    
    @pytest.fixture
    def grid_maze(self):
        """Create a grid maze for testing"""
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        return GridSearchProblem(grid, (0, 0), (3, 3))
    
    def test_bfs_finds_shortest_path(self, simple_graph):
        """Test BFS finds shortest path (by steps)"""
        problem = SimpleProblem(simple_graph, 'A', 'D')
        bfs = BreadthFirstSearch()
        solution = bfs.search(problem)
        
        assert solution == ['go_C', 'go_D']  # Shortest by steps
        assert bfs.nodes_expanded > 0
    
    def test_bfs_on_grid(self, grid_maze):
        """Test BFS on grid problem"""
        bfs = BreadthFirstSearch()
        solution = bfs.search(grid_maze)
        
        assert solution is not None
        assert len(solution) == 6  # Optimal path length
        assert all(action in ['UP', 'DOWN', 'LEFT', 'RIGHT'] for action in solution)
    
    def test_dfs_finds_solution(self, simple_graph):
        """Test DFS finds a solution (may not be optimal)"""
        problem = SimpleProblem(simple_graph, 'A', 'D')
        dfs = DepthFirstSearch()
        solution = dfs.search(problem)
        
        assert solution is not None
        assert dfs.nodes_expanded > 0
    
    def test_dfs_on_grid(self, grid_maze):
        """Test DFS on grid problem"""
        dfs = DepthFirstSearch()
        solution = dfs.search(grid_maze)
        
        assert solution is not None
        # DFS may not find optimal path
        assert len(solution) >= 6
    
    def test_ucs_finds_optimal_cost(self):
        """Test UCS finds path with minimum cost"""
        graph = {
            'A': [('B', 'expensive', 10), ('C', 'cheap', 1)],
            'B': [('D', 'to_D', 1)],
            'C': [('B', 'to_B', 1), ('D', 'to_D', 10)],
            'D': []
        }
        problem = SimpleProblem(graph, 'A', 'D')
        
        ucs = UniformCostSearch()
        solution = ucs.search(problem)
        
        # Should take A->C->B->D (cost 3) not A->B->D (cost 11)
        assert solution == ['cheap', 'to_B', 'to_D']
    
    def test_ucs_on_grid(self, grid_maze):
        """Test UCS on uniform cost grid"""
        ucs = UniformCostSearch()
        solution = ucs.search(grid_maze)
        
        assert solution is not None
        assert len(solution) == 6  # Should find optimal
    
    def test_astar_with_null_heuristic(self, grid_maze):
        """Test A* with null heuristic (should behave like UCS)"""
        astar = AStarSearch(null_heuristic)
        solution = astar.search(grid_maze)
        
        assert solution is not None
        assert len(solution) == 6
    
    def test_astar_with_manhattan(self, grid_maze):
        """Test A* with Manhattan heuristic"""
        def manhattan_h(state, problem):
            return manhattan_distance(state, problem.goal)
        
        astar = AStarSearch(manhattan_h)
        solution = astar.search(grid_maze)
        
        assert solution is not None
        assert len(solution) == 6
        assert astar.nodes_expanded > 0
    
    def test_astar_faster_than_ucs(self, grid_maze):
        """Test that A* with good heuristic expands fewer nodes than UCS"""
        def manhattan_h(state, problem):
            return manhattan_distance(state, problem.goal)
        
        ucs = UniformCostSearch()
        astar = AStarSearch(manhattan_h)
        
        ucs.search(grid_maze)
        astar.search(grid_maze)
        
        # A* should expand fewer nodes with a good heuristic
        assert astar.nodes_expanded <= ucs.nodes_expanded
    
    def test_greedy_best_first(self, grid_maze):
        """Test Greedy Best-First Search"""
        def manhattan_h(state, problem):
            return manhattan_distance(state, problem.goal)
        
        greedy = GreedyBestFirstSearch(manhattan_h)
        solution = greedy.search(grid_maze)
        
        assert solution is not None
        # Greedy may not find optimal path
        assert len(solution) >= 6
    
    def test_iterative_deepening(self, grid_maze):
        """Test Iterative Deepening Search"""
        ids = IterativeDeepeningSearch()
        solution = ids.search(grid_maze, max_depth=10)
        
        assert solution is not None
        assert len(solution) == 6  # Should find optimal
    
    def test_no_solution_exists(self):
        """Test algorithms handle unsolvable problems"""
        grid = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]
        problem = GridSearchProblem(grid, (0, 0), (2, 2))
        
        bfs = BreadthFirstSearch()
        solution = bfs.search(problem)
        assert solution is None
    
    def test_start_is_goal(self, grid_maze):
        """Test when start state is goal"""
        problem = GridSearchProblem(
            [[0, 0], [0, 0]], 
            (0, 0), 
            (0, 0)
        )
        
        bfs = BreadthFirstSearch()
        solution = bfs.search(problem)
        assert solution == []


class TestNode:
    """Test Node class functionality"""
    
    def test_node_creation(self):
        """Test basic node creation"""
        node = Node('A')
        
        assert node.state == 'A'
        assert node.parent is None
        assert node.action is None
        assert node.path_cost == 0
        assert node.depth == 0
    
    def test_node_with_parent(self):
        """Test node with parent"""
        parent = Node('A')
        child = Node('B', parent, 'move', 5)
        
        assert child.state == 'B'
        assert child.parent == parent
        assert child.action == 'move'
        assert child.path_cost == 5
        assert child.depth == 1
    
    def test_solution_path(self):
        """Test solution extraction"""
        n1 = Node('A')
        n2 = Node('B', n1, 'go_B', 1)
        n3 = Node('C', n2, 'go_C', 2)
        
        assert n3.solution() == ['go_B', 'go_C']
        assert n3.path() == ['A', 'B', 'C']
    
    def test_single_node_solution(self):
        """Test solution for single node"""
        node = Node('A')
        
        assert node.solution() == []
        assert node.path() == ['A']


class TestPriorityQueue:
    """Test PriorityQueue implementation"""
    
    def test_basic_operations(self):
        """Test basic push and pop"""
        pq = PriorityQueue()
        
        pq.append((5, 'A'))
        pq.append((2, 'B'))
        pq.append((8, 'C'))
        
        assert len(pq) == 3
        assert pq.pop() == (2, 'B')
        assert pq.pop() == (5, 'A')
        assert pq.pop() == (8, 'C')
        assert len(pq) == 0
    
    def test_custom_priority_function(self):
        """Test with custom priority function"""
        pq = PriorityQueue(order='max', f=lambda x: x[0])
        
        pq.append((5, 'A'))
        pq.append((2, 'B'))
        pq.append((8, 'C'))
        
        assert pq.pop() == (8, 'C')  # Max priority first
    
    def test_contains_operation(self):
        """Test membership checking"""
        pq = PriorityQueue(f=lambda x: x[0])
        
        pq.append((5, 'A'))
        
        assert 'A' in pq
        assert 'B' not in pq
        
        pq.pop()
        assert 'A' not in pq


class TestHeuristics:
    """Test heuristic functions"""
    
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation"""
        assert manhattan_distance((0, 0), (3, 4)) == 7
        assert manhattan_distance((1, 1), (1, 1)) == 0
        assert manhattan_distance((-1, -1), (1, 1)) == 4
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation"""
        assert euclidean_distance((0, 0), (3, 4)) == 5.0
        assert euclidean_distance((1, 1), (1, 1)) == 0.0
        assert abs(euclidean_distance((0, 0), (1, 1)) - math.sqrt(2)) < 0.001
    
    def test_chebyshev_distance(self):
        """Test Chebyshev distance calculation"""
        assert chebyshev_distance((0, 0), (3, 4)) == 4
        assert chebyshev_distance((1, 1), (1, 1)) == 0
        assert chebyshev_distance((-2, -2), (2, 2)) == 4
    
    def test_null_heuristic(self):
        """Test null heuristic always returns 0"""
        assert null_heuristic('any_state') == 0
        assert null_heuristic((1, 2, 3)) == 0
        assert null_heuristic(None) == 0
    
    def test_grid_heuristic(self):
        """Test GridHeuristic class"""
        goal = (5, 5)
        
        # 4-way movement
        h4 = GridHeuristic(goal, '4-way')
        assert h4((0, 0)) == manhattan_distance((0, 0), goal)
        
        # 8-way movement
        h8 = GridHeuristic(goal, '8-way')
        assert h8((0, 0)) == chebyshev_distance((0, 0), goal)
        
        # Continuous movement
        hc = GridHeuristic(goal, 'continuous')
        assert hc((0, 0)) == euclidean_distance((0, 0), goal)
    
    def test_heuristic_admissibility(self):
        """Test that heuristics are admissible for appropriate movement"""
        # For 4-directional movement, Manhattan is admissible
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        problem = GridSearchProblem(grid, (0, 0), (2, 2))
        
        # Manhattan should never overestimate
        for i in range(3):
            for j in range(3):
                h_value = manhattan_distance((i, j), (2, 2))
                # Actual shortest path in 4-directional movement
                actual = abs(i - 2) + abs(j - 2)
                assert h_value <= actual + 0.001


class TestComplexScenarios:
    """Test complex scenarios and edge cases"""
    
    def test_large_grid(self):
        """Test algorithms on larger grid"""
        size = 10
        grid = [[0] * size for _ in range(size)]
        # Add some obstacles
        for i in range(1, size-1):
            grid[i][size//2] = 1
        grid[size//2][size//2] = 0  # Opening in wall
        
        problem = GridSearchProblem(grid, (0, 0), (size-1, size-1))
        
        bfs = BreadthFirstSearch()
        solution = bfs.search(problem)
        assert solution is not None
    
    def test_multiple_optimal_paths(self):
        """Test when multiple optimal paths exist"""
        grid = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        problem = GridSearchProblem(grid, (0, 0), (2, 2))
        
        # All algorithms should find an optimal path
        for Algorithm in [BreadthFirstSearch, UniformCostSearch, AStarSearch]:
            if Algorithm == AStarSearch:
                agent = Algorithm(lambda s, p: manhattan_distance(s, p.goal))
            else:
                agent = Algorithm()
            
            solution = agent.search(problem)
            assert len(solution) == 4  # Optimal length
    
    def test_variable_costs(self):
        """Test UCS and A* with variable edge costs"""
        class VariableCostProblem(SearchProblem):
            def __init__(self):
                self.graph = {
                    'A': [('B', 'cheap', 1), ('C', 'expensive', 10)],
                    'B': [('D', 'normal', 5)],
                    'C': [('D', 'cheap', 1)],
                    'D': []
                }
            
            def get_start_state(self):
                return 'A'
            
            def is_goal_state(self, state):
                return state == 'D'
            
            def get_successors(self, state):
                return self.graph.get(state, [])
        
        problem = VariableCostProblem()
        
        # UCS should find cheapest path
        ucs = UniformCostSearch()
        solution = ucs.search(problem)
        assert solution == ['cheap', 'normal']  # A->B->D, cost 6
        
        # A* with admissible heuristic should also find optimal
        def h(state, problem):
            distances = {'A': 2, 'B': 1, 'C': 1, 'D': 0}
            return distances.get(state, 0)
        
        astar = AStarSearch(h)
        solution = astar.search(problem)
        assert solution == ['cheap', 'normal']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
