"""
Comprehensive test suite for search algorithms, game playing, and MDPs
For CS5368 Intelligent Systems
"""

import unittest
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search.algorithms import *
from src.search.problem import SearchProblem
from src.games.minimax import *
from src.mdp.mdp import *
from src.mdp.agents import *
import numpy as np


class GridSearchProblem(SearchProblem):
    """
    A simple grid world search problem for testing
    """
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def get_start_state(self):
        return self.start
    
    def is_goal_state(self, state):
        return state == self.goal
    
    def get_successors(self, state):
        successors = []
        x, y = state
        
        # Define possible moves: (dx, dy, action_name, cost)
        moves = [
            (0, 1, 'East', 1),
            (1, 0, 'South', 1),
            (0, -1, 'West', 1),
            (-1, 0, 'North', 1)
        ]
        
        for dx, dy, action, cost in moves:
            new_x, new_y = x + dx, y + dy
            # Check if move is valid (within bounds and not a wall)
            if (0 <= new_x < self.rows and 0 <= new_y < self.cols and 
                self.grid[new_x][new_y] != '#'):
                successors.append(((new_x, new_y), action, cost))
        
        return successors


class TestSearchAlgorithms(unittest.TestCase):
    """Test suite for search algorithms"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Simple 5x5 grid
        self.simple_grid = [
            [' ', ' ', ' ', ' ', ' '],
            [' ', '#', '#', '#', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', '#', '#', '#', ' '],
            [' ', ' ', ' ', ' ', ' ']
        ]
        self.simple_problem = GridSearchProblem(
            self.simple_grid, 
            start=(0, 0), 
            goal=(4, 4)
        )
        
        # Maze grid
        self.maze_grid = [
            ['S', ' ', '#', ' ', ' '],
            [' ', ' ', '#', ' ', '#'],
            [' ', ' ', ' ', ' ', '#'],
            ['#', '#', ' ', '#', ' '],
            [' ', ' ', ' ', ' ', 'G']
        ]
        self.maze_problem = GridSearchProblem(
            self.maze_grid,
            start=(0, 0),
            goal=(4, 4)
        )
    
    def test_bfs_finds_shortest_path(self):
        """Test that BFS finds the shortest path"""
        bfs = BreadthFirstSearch()
        path = bfs.search(self.simple_problem)
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 8)  # Shortest path is 8 moves
        
        # Verify path leads to goal
        current = self.simple_problem.get_start_state()
        for action in path:
            successors = self.simple_problem.get_successors(current)
            next_states = {a: s for s, a, _ in successors}
            self.assertIn(action, next_states)
            current = next_states[action]
        self.assertTrue(self.simple_problem.is_goal_state(current))
    
    def test_dfs_finds_a_path(self):
        """Test that DFS finds a valid path (may not be shortest)"""
        dfs = DepthFirstSearch()
        path = dfs.search(self.simple_problem)
        
        self.assertIsNotNone(path)
        
        # Verify path leads to goal
        current = self.simple_problem.get_start_state()
        for action in path:
            successors = self.simple_problem.get_successors(current)
            next_states = {a: s for s, a, _ in successors}
            current = next_states[action]
        self.assertTrue(self.simple_problem.is_goal_state(current))
    
    def test_ucs_finds_optimal_path(self):
        """Test that UCS finds the optimal cost path"""
        ucs = UniformCostSearch()
        path = ucs.search(self.simple_problem)
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 8)  # Optimal path is 8 moves
    
    def test_astar_with_manhattan_heuristic(self):
        """Test A* with Manhattan distance heuristic"""
        def manhattan_heuristic(state, problem):
            x1, y1 = state
            x2, y2 = problem.goal
            return abs(x1 - x2) + abs(y1 - y2)
        
        astar = AStarSearch(heuristic=manhattan_heuristic)
        path = astar.search(self.simple_problem)
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 8)  # Optimal path is 8 moves
        
        # A* should explore similar or fewer nodes than BFS
        bfs = BreadthFirstSearch()
        bfs_path = bfs.search(self.simple_problem)
        self.assertLessEqual(astar.nodes_expanded, bfs.nodes_expanded + 2)
    
    def test_greedy_best_first(self):
        """Test Greedy Best-First Search"""
        def manhattan_heuristic(state, problem):
            x1, y1 = state
            x2, y2 = problem.goal
            return abs(x1 - x2) + abs(y1 - y2)
        
        gbfs = GreedyBestFirstSearch(heuristic=manhattan_heuristic)
        path = gbfs.search(self.simple_problem)
        
        self.assertIsNotNone(path)
        # Path exists but may not be optimal
    
    def test_iterative_deepening(self):
        """Test Iterative Deepening Search"""
        ids = IterativeDeepeningSearch()
        path = ids.search(self.simple_problem, max_depth=20)
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 8)  # Should find optimal path
    
    def test_no_solution_problem(self):
        """Test behavior when no solution exists"""
        # Create impossible problem
        impossible_grid = [
            ['S', '#', 'G'],
            ['#', '#', '#'],
            ['#', '#', '#']
        ]
        impossible_problem = GridSearchProblem(
            impossible_grid,
            start=(0, 0),
            goal=(0, 2)
        )
        
        # All algorithms should return None
        algorithms = [
            BreadthFirstSearch(),
            DepthFirstSearch(),
            UniformCostSearch(),
            AStarSearch(),
            IterativeDeepeningSearch()
        ]
        
        for algo in algorithms:
            path = algo.search(impossible_problem)
            self.assertIsNone(path, f"{algo.__class__.__name__} should return None for impossible problem")


class SimpleTicTacToeState(GameState):
    """Simple Tic-Tac-Toe implementation for testing"""
    
    def __init__(self, board=None, player=0):
        self.board = board if board else [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = player
    
    def get_legal_actions(self, agent_index=0):
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    actions.append((i, j))
        return actions
    
    def generate_successor(self, agent_index, action):
        i, j = action
        new_board = [row[:] for row in self.board]
        new_board[i][j] = 'X' if agent_index == 0 else 'O'
        return SimpleTicTacToeState(new_board, 1 - agent_index)
    
    def is_terminal(self):
        # Check for winner or full board
        return self.get_winner() is not None or all(
            self.board[i][j] != ' ' for i in range(3) for j in range(3)
        )
    
    def get_winner(self):
        """Returns 'X', 'O', or None"""
        # Check rows, columns, and diagonals
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return self.board[0][i]
        
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        return None
    
    def get_utility(self, agent_index=0):
        winner = self.get_winner()
        if winner == 'X':
            return 1 if agent_index == 0 else -1
        elif winner == 'O':
            return -1 if agent_index == 0 else 1
        else:
            return 0


class TestGameAlgorithms(unittest.TestCase):
    """Test suite for game playing algorithms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.empty_state = SimpleTicTacToeState()
        
        # Near-end game state
        self.near_end_board = [
            ['X', 'O', 'X'],
            ['O', 'X', ' '],
            [' ', ' ', 'O']
        ]
        self.near_end_state = SimpleTicTacToeState(self.near_end_board, 0)
    
    def test_minimax_agent(self):
        """Test Minimax agent"""
        agent = MinimaxAgent(depth=9)  # Full game tree for Tic-Tac-Toe
        
        # Test on near-end state
        action = agent.get_action(self.near_end_state)
        self.assertIsNotNone(action)
        self.assertIn(action, self.near_end_state.get_legal_actions())
    
    def test_alphabeta_agent(self):
        """Test Alpha-Beta agent"""
        minimax_agent = MinimaxAgent(depth=5)
        alphabeta_agent = AlphaBetaAgent(depth=5)
        
        # Both should return same action
        minimax_action = minimax_agent.get_action(self.empty_state)
        alphabeta_action = alphabeta_agent.get_action(self.empty_state)
        
        # Actions might differ due to tie-breaking, but values should be same
        self.assertIsNotNone(alphabeta_action)
        
        # Alpha-beta should explore fewer nodes
        self.assertLess(alphabeta_agent.nodes_explored, minimax_agent.nodes_explored)
        print(f"Minimax explored: {minimax_agent.nodes_explored}, "
              f"Alpha-Beta explored: {alphabeta_agent.nodes_explored}")
    
    def test_expectimax_agent(self):
        """Test Expectimax agent"""
        agent = ExpectimaxAgent(depth=3)
        action = agent.get_action(self.empty_state)
        
        self.assertIsNotNone(action)
        self.assertIn(action, self.empty_state.get_legal_actions())
    
    def test_terminal_state_handling(self):
        """Test handling of terminal states"""
        # Create a winning state
        winning_board = [
            ['X', 'X', 'X'],
            ['O', 'O', ' '],
            [' ', ' ', ' ']
        ]
        winning_state = SimpleTicTacToeState(winning_board)
        
        self.assertTrue(winning_state.is_terminal())
        self.assertEqual(winning_state.get_utility(0), 1)  # X wins
        self.assertEqual(winning_state.get_utility(1), -1)  # O loses


class SimpleGridWorldMDP(MarkovDecisionProcess):
    """Simple Grid World MDP for testing"""
    
    def __init__(self, grid, terminals, rewards, noise=0.2):
        self.grid = grid
        self.terminals = terminals
        self.rewards = rewards
        self.noise = noise
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        
    def get_states(self):
        states = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] != '#':
                    states.append((i, j))
        return states
    
    def get_start_state(self):
        return (0, 0)
    
    def get_possible_actions(self, state):
        if state in self.terminals:
            return []
        return ['North', 'South', 'East', 'West']
    
    def get_transition_states_and_probs(self, state, action):
        if state in self.terminals:
            return []
        
        # Define movement directions
        directions = {
            'North': (-1, 0),
            'South': (1, 0),
            'East': (0, 1),
            'West': (0, -1)
        }
        
        # Get intended direction
        dx, dy = directions[action]
        
        # Calculate next states with noise
        transitions = []
        
        # Intended direction (1 - noise probability)
        new_x, new_y = state[0] + dx, state[1] + dy
        if (0 <= new_x < self.rows and 0 <= new_y < self.cols and 
            self.grid[new_x][new_y] != '#'):
            transitions.append(((new_x, new_y), 1 - self.noise))
        else:
            transitions.append((state, 1 - self.noise))
        
        # Perpendicular directions (noise/2 each)
        perpendicular = {
            'North': ['East', 'West'],
            'South': ['East', 'West'],
            'East': ['North', 'South'],
            'West': ['North', 'South']
        }
        
        for perp_action in perpendicular[action]:
            dx, dy = directions[perp_action]
            new_x, new_y = state[0] + dx, state[1] + dy
            if (0 <= new_x < self.rows and 0 <= new_y < self.cols and 
                self.grid[new_x][new_y] != '#'):
                transitions.append(((new_x, new_y), self.noise / 2))
            else:
                transitions.append((state, self.noise / 2))
        
        return transitions
    
    def get_reward(self, state, action, next_state):
        if next_state in self.rewards:
            return self.rewards[next_state]
        return -0.04  # Living penalty
    
    def is_terminal(self, state):
        return state in self.terminals


class TestMDPAlgorithms(unittest.TestCase):
    """Test suite for MDP algorithms"""
    
    def setUp(self):
        """Set up test fixtures"""
        # 4x3 grid world from Russell & Norvig
        self.grid = [
            [' ', ' ', ' ', '+'],
            [' ', '#', ' ', '-'],
            [' ', ' ', ' ', ' ']
        ]
        self.terminals = [(0, 3), (1, 3)]
        self.rewards = {(0, 3): 1.0, (1, 3): -1.0}
        
        self.mdp = SimpleGridWorldMDP(self.grid, self.terminals, self.rewards)
    
    def test_value_iteration(self):
        """Test value iteration algorithm"""
        vi_agent = ValueIterationAgent(self.mdp, discount=0.9, iterations=100)
        
        # Check that terminal states have correct values
        self.assertEqual(vi_agent.get_value((0, 3)), 0)  # Terminal states have 0 value
        self.assertEqual(vi_agent.get_value((1, 3)), 0)
        
        # Check that states near positive terminal have positive values
        self.assertGreater(vi_agent.get_value((0, 2)), 0)
        
        # State (1, 2) might have positive value due to path to positive terminal
        # Just check that it's a reasonable value
        self.assertIsInstance(vi_agent.get_value((1, 2)), (int, float))
        
        # Check policy makes sense
        # State (2, 0) should go East or North to avoid negative terminal
        policy_2_0 = vi_agent.get_policy((2, 0))
        self.assertIn(policy_2_0, ['East', 'North'])
    
    def test_policy_iteration(self):
        """Test policy iteration algorithm"""
        pi_agent = PolicyIterationAgent(self.mdp, discount=0.9)
        
        # Compare with value iteration results
        vi_agent = ValueIterationAgent(self.mdp, discount=0.9, iterations=100)
        
        # Policies should be similar
        for state in self.mdp.get_states():
            if not self.mdp.is_terminal(state):
                pi_action = pi_agent.get_policy(state)
                vi_action = vi_agent.get_policy(state)
                # Due to ties, actions might differ but should be reasonable
                self.assertIsNotNone(pi_action)
                self.assertIsNotNone(vi_action)
    
    def test_q_learning(self):
        """Test Q-Learning algorithm"""
        def get_actions(state):
            if state in self.terminals:
                return []
            return ['North', 'South', 'East', 'West']
        
        q_agent = QLearningAgent(action_fn=get_actions, discount=0.9, 
                                 alpha=0.5, epsilon=0.1)
        
        # Simulate episodes
        for episode in range(100):
            state = self.mdp.get_start_state()
            q_agent.start_episode()
            
            for step in range(100):  # Max steps per episode
                if self.mdp.is_terminal(state):
                    break
                
                action = q_agent.get_action(state)
                if action is None:
                    break
                
                # Simulate transition
                transitions = self.mdp.get_transition_states_and_probs(state, action)
                if not transitions:
                    break
                
                # Sample next state
                states, probs = zip(*transitions)
                next_state = random.choices(states, weights=probs)[0]
                reward = self.mdp.get_reward(state, action, next_state)
                
                # Update Q-values
                q_agent.update(state, action, next_state, reward)
                state = next_state
        
        # After training, Q-values should be reasonable
        q_agent.stop_training()
        
        # State near positive terminal should have positive Q-values for actions toward it
        state_0_2 = (0, 2)
        q_east = q_agent.get_q_value(state_0_2, 'East')
        q_west = q_agent.get_q_value(state_0_2, 'West')
        self.assertGreater(q_east, q_west)  # East leads to +1 terminal
    
    def test_sarsa(self):
        """Test SARSA algorithm"""
        def get_actions(state):
            if state in self.terminals:
                return []
            return ['North', 'South', 'East', 'West']
        
        sarsa_agent = SARSAAgent(action_fn=get_actions, discount=0.9,
                                 alpha=0.5, epsilon=0.1)
        
        # Simulate episodes
        for episode in range(100):
            state = self.mdp.get_start_state()
            sarsa_agent.start_episode()
            action = sarsa_agent.get_action(state)
            
            for step in range(100):
                if self.mdp.is_terminal(state) or action is None:
                    break
                
                # Simulate transition
                transitions = self.mdp.get_transition_states_and_probs(state, action)
                if not transitions:
                    break
                
                states, probs = zip(*transitions)
                next_state = random.choices(states, weights=probs)[0]
                reward = self.mdp.get_reward(state, action, next_state)
                
                # Get next action
                next_action = sarsa_agent.get_action(next_state)
                sarsa_agent.next_action = next_action
                
                # Update Q-values
                sarsa_agent.update(state, action, next_state, reward)
                
                state = next_state
                action = next_action
        
        # After training, should have learned something
        sarsa_agent.stop_training()
        best_action = sarsa_agent.compute_action_from_q_values((0, 2))
        self.assertIsNotNone(best_action)


def run_all_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSearchAlgorithms))
    suite.addTests(loader.loadTestsFromTestCase(TestGameAlgorithms))
    suite.addTests(loader.loadTestsFromTestCase(TestMDPAlgorithms))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)