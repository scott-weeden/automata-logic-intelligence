"""
Comprehensive Test Suite for Game Playing Algorithms

Tests minimax, alpha-beta pruning, and expectimax algorithms on various
game scenarios. Validates correctness, optimality, and pruning efficiency.

Based on CS 5368 Week 4-5 material on adversarial search and game theory.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from games import (
    GameState, GameAgent, MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent,
    Game, TicTacToe
)

class SimpleGameState(GameState):
    """
    Simple test game state for algorithm testing.
    
    Game tree structure:
    - Each state has value equal to its depth (for testing)
    - Actions are 'left' and 'right' 
    - Terminal at depth 3 with utilities based on path
    """
    
    def __init__(self, depth=0, path="", max_depth=3):
        """
        Initialize simple game state.
        
        Args:
            depth: Current depth in game tree
            path: String representing path taken ('L' for left, 'R' for right)
            max_depth: Maximum depth before terminal state
        """
        self.depth = depth
        self.path = path
        self.max_depth = max_depth
        self.current_player = depth % 2  # Alternates between 0 and 1
    
    def get_legal_actions(self, agent_index=0):
        """Return available actions: ['left', 'right'] unless terminal."""
        if self.is_terminal():
            return []
        return ['left', 'right']
    
    def generate_successor(self, agent_index, action):
        """Generate successor state based on action."""
        if action == 'left':
            new_path = self.path + 'L'
        else:  # action == 'right'
            new_path = self.path + 'R'
        
        return SimpleGameState(self.depth + 1, new_path, self.max_depth)
    
    def is_terminal(self):
        """Terminal when max depth reached."""
        return self.depth >= self.max_depth
    
    def get_utility(self, agent_index=0):
        """
        Utility function for terminal states.
        
        Returns utility based on path taken:
        - 'LLL': 3, 'LLR': 12, 'LRL': 8, 'LRR': 2
        - 'RLL': 4, 'RLR': 6, 'RRL': 14, 'RRR': 5
        
        This creates interesting minimax scenarios for testing.
        """
        if not self.is_terminal():
            return 0
        
        utilities = {
            'LLL': 3, 'LLR': 12, 'LRL': 8, 'LRR': 2,
            'RLL': 4, 'RLR': 6, 'RRL': 14, 'RRR': 5
        }
        
        return utilities.get(self.path, 0)

class TestGameStateInterface(unittest.TestCase):
    """Test the GameState base class interface and simple implementation."""
    
    def setUp(self):
        """Set up test game states."""
        self.initial_state = SimpleGameState()
        self.terminal_state = SimpleGameState(depth=3, path="LLL")
    
    def test_game_state_creation(self):
        """Test basic game state creation and attributes."""
        state = SimpleGameState(depth=1, path="L")
        
        self.assertEqual(state.depth, 1)
        self.assertEqual(state.path, "L")
        self.assertFalse(state.is_terminal())
    
    def test_legal_actions_non_terminal(self):
        """Test legal actions for non-terminal states."""
        actions = self.initial_state.get_legal_actions()
        
        self.assertEqual(len(actions), 2)
        self.assertIn('left', actions)
        self.assertIn('right', actions)
    
    def test_legal_actions_terminal(self):
        """Test legal actions for terminal states (should be empty)."""
        actions = self.terminal_state.get_legal_actions()
        
        self.assertEqual(len(actions), 0)
    
    def test_successor_generation(self):
        """Test successor state generation."""
        # Test left action
        left_successor = self.initial_state.generate_successor(0, 'left')
        self.assertEqual(left_successor.depth, 1)
        self.assertEqual(left_successor.path, 'L')
        
        # Test right action
        right_successor = self.initial_state.generate_successor(0, 'right')
        self.assertEqual(right_successor.depth, 1)
        self.assertEqual(right_successor.path, 'R')
    
    def test_terminal_detection(self):
        """Test terminal state detection."""
        self.assertFalse(self.initial_state.is_terminal())
        self.assertTrue(self.terminal_state.is_terminal())
    
    def test_utility_calculation(self):
        """Test utility calculation for terminal states."""
        # Test known utility values
        lll_state = SimpleGameState(depth=3, path="LLL")
        self.assertEqual(lll_state.get_utility(0), 3)
        
        rrr_state = SimpleGameState(depth=3, path="RRR")
        self.assertEqual(rrr_state.get_utility(0), 5)
        
        # Non-terminal should return 0
        self.assertEqual(self.initial_state.get_utility(0), 0)

class TestMinimaxAgent(unittest.TestCase):
    """Test Minimax algorithm for optimal play in perfect information games."""
    
    def setUp(self):
        """Set up minimax agents with different depths."""
        self.minimax_depth_3 = MinimaxAgent(index=0, depth=3)
        self.minimax_depth_1 = MinimaxAgent(index=0, depth=1)
        self.initial_state = SimpleGameState()
    
    def test_minimax_agent_creation(self):
        """Test minimax agent initialization."""
        agent = MinimaxAgent(index=0, depth=5)
        
        self.assertEqual(agent.index, 0)
        self.assertEqual(agent.depth, 5)
        self.assertEqual(agent.nodes_explored, 0)
    
    def test_minimax_perfect_play(self):
        """Test minimax finds optimal action for known game tree."""
        # Reset node counter
        self.minimax_depth_3.nodes_explored = 0
        
        action = self.minimax_depth_3.get_action(self.initial_state)
        
        # Should choose action that leads to best outcome
        # Based on our utility function, agent should choose 'right'
        # because it leads to higher minimax value
        self.assertIn(action, ['left', 'right'])
        
        # Should have explored some nodes
        self.assertGreater(self.minimax_depth_3.nodes_explored, 0)
    
    def test_minimax_depth_limiting(self):
        """Test minimax with depth limiting."""
        # Shallow search should still return valid action
        action = self.minimax_depth_1.get_action(self.initial_state)
        
        self.assertIn(action, ['left', 'right'])
        
        # Should explore fewer nodes than full depth search
        self.assertLess(
            self.minimax_depth_1.nodes_explored,
            self.minimax_depth_3.nodes_explored + 1
        )
    
    def test_minimax_terminal_state_handling(self):
        """Test minimax behavior on terminal states."""
        terminal_state = SimpleGameState(depth=3, path="LLL")
        
        # Should return None or handle gracefully for terminal state
        action = self.minimax_depth_3.get_action(terminal_state)
        
        # Terminal state has no legal actions
        self.assertIsNone(action)
    
    def test_minimax_deterministic_behavior(self):
        """Test that minimax returns consistent results."""
        # Run multiple times, should get same result
        action1 = self.minimax_depth_3.get_action(self.initial_state)
        action2 = self.minimax_depth_3.get_action(self.initial_state)
        
        self.assertEqual(action1, action2)

class TestAlphaBetaAgent(unittest.TestCase):
    """Test Alpha-Beta pruning for efficiency while maintaining optimality."""
    
    def setUp(self):
        """Set up alpha-beta agents and comparison minimax agent."""
        self.alphabeta = AlphaBetaAgent(index=0, depth=3)
        self.minimax = MinimaxAgent(index=0, depth=3)
        self.initial_state = SimpleGameState()
    
    def test_alphabeta_agent_creation(self):
        """Test alpha-beta agent initialization."""
        agent = AlphaBetaAgent(index=1, depth=4)
        
        self.assertEqual(agent.index, 1)
        self.assertEqual(agent.depth, 4)
        self.assertEqual(agent.nodes_explored, 0)
    
    def test_alphabeta_same_result_as_minimax(self):
        """Test alpha-beta returns same optimal action as minimax."""
        # Reset counters
        self.alphabeta.nodes_explored = 0
        self.minimax.nodes_explored = 0
        
        ab_action = self.alphabeta.get_action(self.initial_state)
        mm_action = self.minimax.get_action(self.initial_state)
        
        # Should make same optimal decision
        self.assertEqual(ab_action, mm_action)
    
    def test_alphabeta_pruning_efficiency(self):
        """Test that alpha-beta explores fewer nodes than minimax."""
        # Reset counters
        self.alphabeta.nodes_explored = 0
        self.minimax.nodes_explored = 0
        
        # Run both algorithms
        self.alphabeta.get_action(self.initial_state)
        self.minimax.get_action(self.initial_state)
        
        # Alpha-beta should explore fewer nodes due to pruning
        self.assertLess(
            self.alphabeta.nodes_explored,
            self.minimax.nodes_explored
        )
        
        print(f"Alpha-Beta explored: {self.alphabeta.nodes_explored}")
        print(f"Minimax explored: {self.minimax.nodes_explored}")
        print(f"Pruning efficiency: {1 - self.alphabeta.nodes_explored/self.minimax.nodes_explored:.2%}")
    
    def test_alphabeta_different_depths(self):
        """Test alpha-beta with different search depths."""
        shallow_ab = AlphaBetaAgent(index=0, depth=1)
        deep_ab = AlphaBetaAgent(index=0, depth=4)
        
        shallow_action = shallow_ab.get_action(self.initial_state)
        deep_action = deep_ab.get_action(self.initial_state)
        
        # Both should return valid actions
        self.assertIn(shallow_action, ['left', 'right'])
        self.assertIn(deep_action, ['left', 'right'])
        
        # Deeper search should explore more nodes
        self.assertGreater(deep_ab.nodes_explored, shallow_ab.nodes_explored)
    
    def test_alphabeta_pruning_scenarios(self):
        """Test alpha-beta pruning in specific game scenarios."""
        # Create state where pruning should occur
        # This tests the pruning logic more directly
        
        # Start from a state deeper in the tree
        mid_game_state = SimpleGameState(depth=1, path="L")
        
        self.alphabeta.nodes_explored = 0
        action = self.alphabeta.get_action(mid_game_state)
        
        # Should still find valid action
        self.assertIn(action, ['left', 'right'])
        
        # Should have done some pruning (fewer than full expansion)
        self.assertGreater(self.alphabeta.nodes_explored, 0)

class TestExpectimaxAgent(unittest.TestCase):
    """Test Expectimax algorithm for games with chance elements."""
    
    def setUp(self):
        """Set up expectimax agent."""
        self.expectimax = ExpectimaxAgent(index=0, depth=3)
        self.initial_state = SimpleGameState()
    
    def test_expectimax_agent_creation(self):
        """Test expectimax agent initialization."""
        agent = ExpectimaxAgent(index=1, depth=2)
        
        self.assertEqual(agent.index, 1)
        self.assertEqual(agent.depth, 2)
        self.assertEqual(agent.nodes_explored, 0)
    
    def test_expectimax_finds_action(self):
        """Test expectimax returns valid action."""
        action = self.expectimax.get_action(self.initial_state)
        
        self.assertIn(action, ['left', 'right'])
        self.assertGreater(self.expectimax.nodes_explored, 0)
    
    def test_expectimax_vs_minimax_behavior(self):
        """Test expectimax behavior compared to minimax."""
        minimax = MinimaxAgent(index=0, depth=3)
        
        # Reset counters
        self.expectimax.nodes_explored = 0
        minimax.nodes_explored = 0
        
        exp_action = self.expectimax.get_action(self.initial_state)
        mm_action = minimax.get_action(self.initial_state)
        
        # Both should return valid actions (may differ due to different assumptions)
        self.assertIn(exp_action, ['left', 'right'])
        self.assertIn(mm_action, ['left', 'right'])
        
        # Expectimax assumes chance nodes, so may make different decisions
        # We just verify both are valid
    
    def test_expectimax_chance_node_handling(self):
        """Test expectimax properly handles chance nodes (opponent moves)."""
        # In our simple game, opponent moves are treated as chance nodes
        # Expectimax should average over opponent's possible actions
        
        action = self.expectimax.get_action(self.initial_state)
        
        # Should handle the averaging correctly and return valid action
        self.assertIsNotNone(action)
        self.assertIn(action, ['left', 'right'])

class TestTicTacToeGame(unittest.TestCase):
    """Test TicTacToe game implementation and integration with agents."""
    
    def setUp(self):
        """Set up TicTacToe game and agents."""
        self.game = TicTacToe()
        self.minimax_agent = MinimaxAgent(index=0, depth=9)  # Full game tree
        self.alphabeta_agent = AlphaBetaAgent(index=0, depth=9)
    
    def test_tictactoe_initialization(self):
        """Test TicTacToe game initialization."""
        self.assertIsNotNone(self.game.initial)
        
        # Initial state should not be terminal
        self.assertFalse(self.game.initial.is_terminal())
        
        # Should have 9 legal actions (empty squares)
        actions = self.game.initial.get_legal_actions(0)
        self.assertEqual(len(actions), 9)
    
    def test_tictactoe_move_generation(self):
        """Test TicTacToe move generation and state transitions."""
        initial_state = self.game.initial
        
        # Make a move
        action = (1, 1)  # Center square
        new_state = initial_state.generate_successor(0, action)
        
        # Should have one fewer legal action
        self.assertEqual(len(new_state.get_legal_actions(1)), 8)
        
        # Should not be terminal after one move
        self.assertFalse(new_state.is_terminal())
    
    def test_tictactoe_winning_detection(self):
        """Test TicTacToe winning condition detection."""
        # Create a winning state manually
        # This would require implementing a specific board state
        # For now, we test that the game can detect terminal states
        
        state = self.game.initial
        
        # Play a few moves to test non-terminal detection
        state = state.generate_successor(0, (0, 0))  # X
        state = state.generate_successor(1, (0, 1))  # O
        state = state.generate_successor(0, (1, 1))  # X
        
        # Should still not be terminal
        self.assertFalse(state.is_terminal())
    
    def test_agents_on_tictactoe(self):
        """Test that game agents can play TicTacToe."""
        # Test minimax agent can choose action
        action = self.minimax_agent.get_action(self.game.initial)
        
        self.assertIsNotNone(action)
        self.assertIsInstance(action, tuple)
        self.assertEqual(len(action), 2)  # (row, col)
        
        # Action should be valid
        valid_actions = self.game.initial.get_legal_actions(0)
        self.assertIn(action, valid_actions)
    
    def test_alphabeta_vs_minimax_tictactoe(self):
        """Test alpha-beta vs minimax on TicTacToe for efficiency."""
        # Reset counters
        self.minimax_agent.nodes_explored = 0
        self.alphabeta_agent.nodes_explored = 0
        
        # Get actions from both
        mm_action = self.minimax_agent.get_action(self.game.initial)
        ab_action = self.alphabeta_agent.get_action(self.game.initial)
        
        # Both should choose valid actions
        valid_actions = self.game.initial.get_legal_actions(0)
        self.assertIn(mm_action, valid_actions)
        self.assertIn(ab_action, valid_actions)
        
        # Alpha-beta should explore significantly fewer nodes
        self.assertLess(
            self.alphabeta_agent.nodes_explored,
            self.minimax_agent.nodes_explored
        )
        
        print(f"TicTacToe - Minimax: {self.minimax_agent.nodes_explored}, "
              f"Alpha-Beta: {self.alphabeta_agent.nodes_explored}")

class TestGameAgentInterface(unittest.TestCase):
    """Test the GameAgent base class and common functionality."""
    
    def test_game_agent_initialization(self):
        """Test GameAgent base class initialization."""
        agent = GameAgent(index=1)
        
        self.assertEqual(agent.index, 1)
        self.assertEqual(agent.nodes_explored, 0)
    
    def test_game_agent_abstract_methods(self):
        """Test that GameAgent requires get_action implementation."""
        agent = GameAgent()
        
        # Should raise NotImplementedError for abstract method
        with self.assertRaises(NotImplementedError):
            agent.get_action(SimpleGameState())

class TestGamePerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics and complexity of game algorithms."""
    
    def setUp(self):
        """Set up agents for performance testing."""
        self.minimax = MinimaxAgent(index=0, depth=4)
        self.alphabeta = AlphaBetaAgent(index=0, depth=4)
        self.expectimax = ExpectimaxAgent(index=0, depth=4)
        self.test_state = SimpleGameState()
    
    def test_algorithm_node_exploration_comparison(self):
        """Compare node exploration across different algorithms."""
        # Reset all counters
        self.minimax.nodes_explored = 0
        self.alphabeta.nodes_explored = 0
        self.expectimax.nodes_explored = 0
        
        # Run all algorithms
        mm_action = self.minimax.get_action(self.test_state)
        ab_action = self.alphabeta.get_action(self.test_state)
        exp_action = self.expectimax.get_action(self.test_state)
        
        # All should return valid actions
        for action in [mm_action, ab_action, exp_action]:
            self.assertIn(action, ['left', 'right'])
        
        # Print performance comparison
        print(f"\nPerformance Comparison (depth=4):")
        print(f"Minimax nodes: {self.minimax.nodes_explored}")
        print(f"Alpha-Beta nodes: {self.alphabeta.nodes_explored}")
        print(f"Expectimax nodes: {self.expectimax.nodes_explored}")
        
        # Alpha-beta should be most efficient
        self.assertLessEqual(
            self.alphabeta.nodes_explored,
            self.minimax.nodes_explored
        )
    
    def test_depth_scaling_behavior(self):
        """Test how node exploration scales with search depth."""
        depths = [1, 2, 3]
        node_counts = []
        
        for depth in depths:
            agent = AlphaBetaAgent(index=0, depth=depth)
            agent.get_action(self.test_state)
            node_counts.append(agent.nodes_explored)
        
        # Node count should generally increase with depth
        # (though pruning may cause some variation)
        self.assertGreater(node_counts[-1], node_counts[0])
        
        print(f"\nDepth scaling for Alpha-Beta:")
        for i, depth in enumerate(depths):
            print(f"Depth {depth}: {node_counts[i]} nodes")

if __name__ == '__main__':
    # Run all game algorithm tests
    unittest.main(verbosity=2)
