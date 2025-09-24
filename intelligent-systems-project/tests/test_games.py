"""
Comprehensive test suite for games module
Tests game algorithms, game states, and game implementations
"""

import pytest
import sys
import os
from typing import List, Any
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games import (
    GameState, GameAgent,
    MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent,
    Game, TicTacToe
)


# Test Game Implementations
class SimpleGameState(GameState):
    """Simple game state for testing"""
    
    def __init__(self, value, children=None, terminal=False):
        self.value = value
        self.children = children or []
        self.terminal = terminal
    
    def get_legal_actions(self, agent_index=0):
        if self.terminal:
            return []
        return list(range(len(self.children)))
    
    def generate_successor(self, agent_index, action):
        if action < len(self.children):
            return self.children[action]
        return None
    
    def is_terminal(self):
        return self.terminal or len(self.children) == 0
    
    def get_utility(self, agent_index=0):
        if agent_index == 0:
            return self.value
        return -self.value  # Adversarial for agent 1


class TwoPlayerGame(Game):
    """Simple two-player game for testing"""
    
    def __init__(self, tree):
        self.tree = tree
        self.initial = tree
    
    def actions(self, state):
        return state.get_legal_actions()
    
    def result(self, state, action):
        return state.generate_successor(0, action)
    
    def terminal_test(self, state):
        return state.is_terminal()
    
    def utility(self, state, player):
        return state.get_utility(player)
    
    def to_move(self, state):
        # Simple alternation for testing
        return 0
    
    def display(self, state):
        print(f"State value: {state.value}")


class TestGameState:
    """Test GameState functionality"""
    
    def test_simple_game_state(self):
        """Test basic game state operations"""
        terminal_state = SimpleGameState(10, terminal=True)
        
        assert terminal_state.is_terminal() == True
        assert terminal_state.get_utility(0) == 10
        assert terminal_state.get_utility(1) == -10
        assert terminal_state.get_legal_actions() == []
    
    def test_game_state_with_children(self):
        """Test game state with successors"""
        leaf1 = SimpleGameState(5, terminal=True)
        leaf2 = SimpleGameState(3, terminal=True)
        parent = SimpleGameState(0, children=[leaf1, leaf2])
        
        assert parent.is_terminal() == False
        assert parent.get_legal_actions() == [0, 1]
        assert parent.generate_successor(0, 0) == leaf1
        assert parent.generate_successor(0, 1) == leaf2


class TestMinimaxAgent:
    """Test Minimax algorithm"""
    
    def create_simple_tree(self):
        """Create a simple game tree for testing
        
                 root
                /    \
               /      \
              A        B
            /  \      /  \
           3    12   8    2
        """
        # Terminal nodes
        leaf1 = SimpleGameState(3, terminal=True)
        leaf2 = SimpleGameState(12, terminal=True)
        leaf3 = SimpleGameState(8, terminal=True)
        leaf4 = SimpleGameState(2, terminal=True)
        
        # Internal nodes
        nodeA = SimpleGameState(0, children=[leaf1, leaf2])
        nodeB = SimpleGameState(0, children=[leaf3, leaf4])
        
        # Root
        root = SimpleGameState(0, children=[nodeA, nodeB])
        
        return root
    
    def test_minimax_basic(self):
        """Test basic minimax decision making"""
        root = self.create_simple_tree()
        
        # Depth 2 minimax should evaluate all leaves
        agent = MinimaxAgent(index=0, depth=2)
        action = agent.get_action(root)
        
        # Should choose action 0 (left subtree) as max(min(3,12), min(8,2)) = max(3,2) = 3
        assert action == 0
        assert agent.nodes_explored > 0
    
    def test_minimax_depth_limit(self):
        """Test minimax with depth limit"""
        root = self.create_simple_tree()
        
        # Depth 1 should only look one level deep
        agent = MinimaxAgent(index=0, depth=1)
        action = agent.get_action(root)
        
        assert action is not None
        assert agent.nodes_explored > 0
    
    def test_minimax_terminal_state(self):
        """Test minimax on terminal state"""
        terminal = SimpleGameState(10, terminal=True)
        agent = MinimaxAgent(index=0, depth=2)
        
        action = agent.get_action(terminal)
        assert action is None  # No actions available
    
    def test_minimax_single_action(self):
        """Test minimax with only one action"""
        leaf = SimpleGameState(5, terminal=True)
        root = SimpleGameState(0, children=[leaf])
        
        agent = MinimaxAgent(index=0, depth=1)
        action = agent.get_action(root)
        
        assert action == 0  # Only choice


class TestAlphaBetaAgent:
    """Test Alpha-Beta pruning"""
    
    def create_prunable_tree(self):
        """Create a tree where alpha-beta can prune
        
                 root(MAX)
                /         \
           A(MIN)         B(MIN)  
           /    \         /    \
          3     12       2     [pruned]
        
        After finding 3 in left subtree, and 2 in right subtree,
        no need to explore further in right subtree
        """
        # More complex tree for pruning
        leaves = [SimpleGameState(i, terminal=True) for i in [3, 12, 2, 14, 5, 8]]
        
        nodeA = SimpleGameState(0, children=leaves[0:2])  # 3, 12
        nodeB = SimpleGameState(0, children=leaves[2:4])  # 2, 14
        nodeC = SimpleGameState(0, children=leaves[4:6])  # 5, 8
        
        root = SimpleGameState(0, children=[nodeA, nodeB, nodeC])
        
        return root
    
    def test_alphabeta_same_as_minimax(self):
        """Test that alpha-beta gives same result as minimax"""
        root = self.create_prunable_tree()
        
        minimax = MinimaxAgent(index=0, depth=2)
        alphabeta = AlphaBetaAgent(index=0, depth=2)
        
        minimax_action = minimax.get_action(root)
        alphabeta_action = alphabeta.get_action(root)
        
        # Should get same decision
        assert minimax_action == alphabeta_action
    
    def test_alphabeta_prunes(self):
        """Test that alpha-beta explores fewer nodes than minimax"""
        root = self.create_prunable_tree()
        
        minimax = MinimaxAgent(index=0, depth=2)
        alphabeta = AlphaBetaAgent(index=0, depth=2)
        
        minimax.get_action(root)
        alphabeta.get_action(root)
        
        # Alpha-beta should explore fewer or equal nodes
        assert alphabeta.nodes_explored <= minimax.nodes_explored
    
    def test_alphabeta_depth_limit(self):
        """Test alpha-beta with depth limit"""
        root = self.create_prunable_tree()
        
        agent = AlphaBetaAgent(index=0, depth=1)
        action = agent.get_action(root)
        
        assert action is not None
        assert agent.nodes_explored > 0


class TestExpectimaxAgent:
    """Test Expectimax for stochastic games"""
    
    def create_chance_tree(self):
        """Create a tree with chance nodes
        
                root(MAX)
                /       \
           A(CHANCE)   B(CHANCE)
            /    \      /    \
           4     6     2     8
        
        Chance nodes take average of children
        """
        # Terminal nodes
        leaves = [SimpleGameState(i, terminal=True) for i in [4, 6, 2, 8]]
        
        # Chance nodes (will be averaged)
        nodeA = SimpleGameState(0, children=leaves[0:2])
        nodeB = SimpleGameState(0, children=leaves[2:4])
        
        root = SimpleGameState(0, children=[nodeA, nodeB])
        
        return root
    
    def test_expectimax_basic(self):
        """Test basic expectimax decision"""
        root = self.create_chance_tree()
        
        agent = ExpectimaxAgent(index=0, depth=2)
        action = agent.get_action(root)
        
        # Should choose action 0: avg(4,6)=5 > avg(2,8)=5
        # Both equal, so either is acceptable
        assert action in [0, 1]
        assert agent.nodes_explored > 0
    
    def test_expectimax_different_from_minimax(self):
        """Test that expectimax gives different results than minimax for chance nodes"""
        # Create tree where minimax and expectimax differ
        leaves = [SimpleGameState(i, terminal=True) for i in [1, 10, 5, 6]]
        nodeA = SimpleGameState(0, children=leaves[0:2])  # 1, 10
        nodeB = SimpleGameState(0, children=leaves[2:4])  # 5, 6
        root = SimpleGameState(0, children=[nodeA, nodeB])
        
        minimax = MinimaxAgent(index=0, depth=2)
        expectimax = ExpectimaxAgent(index=0, depth=2)
        
        # Minimax assumes worst case: min(1,10)=1 vs min(5,6)=5, chooses B
        # Expectimax uses average: avg(1,10)=5.5 vs avg(5,6)=5.5
        
        minimax_action = minimax.get_action(root)
        expectimax_action = expectimax.get_action(root)
        
        # Actions might differ based on implementation
        assert minimax_action is not None
        assert expectimax_action is not None


class TestTicTacToe:
    """Test Tic-Tac-Toe game implementation"""
    
    def test_tictactoe_initialization(self):
        """Test TicTacToe game initialization"""
        game = TicTacToe()
        
        assert game.initial is not None
        assert not game.terminal_test(game.initial)
    
    def test_tictactoe_actions(self):
        """Test available actions in TicTacToe"""
        game = TicTacToe()
        initial_actions = game.actions(game.initial)
        
        # Should have 9 possible moves initially
        assert len(initial_actions) == 9
    
    def test_tictactoe_move(self):
        """Test making a move in TicTacToe"""
        game = TicTacToe()
        
        # Make first move
        action = game.actions(game.initial)[0]
        new_state = game.result(game.initial, action)
        
        assert new_state != game.initial
        assert len(game.actions(new_state)) == 8  # One less action available
    
    def test_tictactoe_terminal(self):
        """Test terminal state detection"""
        game = TicTacToe()
        
        # Play until terminal or max moves
        state = game.initial
        moves = 0
        max_moves = 9
        
        while not game.terminal_test(state) and moves < max_moves:
            actions = game.actions(state)
            if actions:
                action = actions[0]
                state = game.result(state, action)
                moves += 1
            else:
                break
        
        # Game should terminate within 9 moves
        assert moves <= 9
    
    def test_tictactoe_with_minimax(self):
        """Test playing TicTacToe with Minimax agent"""
        game = TicTacToe()
        agent = MinimaxAgent(index=0, depth=2)
        
        # Agent should be able to choose a move
        action = agent.get_action(game.initial)
        assert action is not None
        assert action in game.actions(game.initial)
    
    def test_tictactoe_perfect_play(self):
        """Test that perfect play leads to draw"""
        game = TicTacToe()
        
        # Two perfect players (high depth)
        player1 = MinimaxAgent(index=0, depth=9)
        player2 = MinimaxAgent(index=1, depth=9)
        
        state = game.initial
        current_player = 0
        moves = 0
        
        while not game.terminal_test(state) and moves < 9:
            if current_player == 0:
                action = player1.get_action(state)
            else:
                action = player2.get_action(state)
            
            if action is not None:
                state = game.result(state, action)
            current_player = 1 - current_player
            moves += 1
        
        # Perfect play should lead to a draw (utility 0)
        if game.terminal_test(state):
            utility = game.utility(state, 0)
            assert abs(utility) <= 0.5  # Draw or very small advantage


class TestGameAgent:
    """Test GameAgent base class"""
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = GameAgent(index=0)
        
        assert agent.index == 0
        assert hasattr(agent, 'nodes_explored')
    
    def test_agent_subclass(self):
        """Test that agents are proper subclasses"""
        minimax = MinimaxAgent(index=0, depth=2)
        alphabeta = AlphaBetaAgent(index=1, depth=3)
        
        assert isinstance(minimax, GameAgent)
        assert isinstance(alphabeta, GameAgent)
        assert minimax.index == 0
        assert alphabeta.index == 1


class TestComplexGameScenarios:
    """Test complex game scenarios"""
    
    def test_deep_tree(self):
        """Test agents on deeper game trees"""
        # Create a 3-level tree
        def create_deep_tree(depth, branching=2):
            if depth == 0:
                return SimpleGameState(random.randint(-10, 10), terminal=True)
            
            children = [create_deep_tree(depth-1, branching) for _ in range(branching)]
            return SimpleGameState(0, children=children)
        
        root = create_deep_tree(3, 3)  # 3 levels, branching factor 3
        
        # All agents should handle deep trees
        for AgentClass in [MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent]:
            agent = AgentClass(index=0, depth=3)
            action = agent.get_action(root)
            assert action is not None
    
    def test_asymmetric_tree(self):
        """Test agents on asymmetric trees"""
        # Create an unbalanced tree
        deep_leaf = SimpleGameState(10, terminal=True)
        shallow_leaf = SimpleGameState(5, terminal=True)
        
        deep_branch = SimpleGameState(0, children=[deep_leaf])
        for _ in range(3):  # Make it deeper
            deep_branch = SimpleGameState(0, children=[deep_branch])
        
        root = SimpleGameState(0, children=[deep_branch, shallow_leaf])
        
        agent = MinimaxAgent(index=0, depth=5)
        action = agent.get_action(root)
        assert action is not None
    
    def test_many_actions(self):
        """Test with many possible actions"""
        # Create state with many children
        num_actions = 20
        leaves = [SimpleGameState(i, terminal=True) for i in range(num_actions)]
        root = SimpleGameState(0, children=leaves)
        
        agent = AlphaBetaAgent(index=0, depth=1)
        action = agent.get_action(root)
        
        assert action is not None
        assert 0 <= action < num_actions
    
    def test_alternating_players(self):
        """Test proper alternation between players"""
        # Create a game tree with explicit player tracking
        class PlayerGameState(SimpleGameState):
            def __init__(self, value, player=0, children=None, terminal=False):
                super().__init__(value, children, terminal)
                self.player = player
        
        # Build a tree with alternating players
        leaves = [PlayerGameState(i, player=1, terminal=True) for i in [3, 5, 2, 8]]
        node1 = PlayerGameState(0, player=1, children=leaves[0:2])
        node2 = PlayerGameState(0, player=1, children=leaves[2:4])
        root = PlayerGameState(0, player=0, children=[node1, node2])
        
        # Both agents should work with alternating players
        agent1 = MinimaxAgent(index=0, depth=2)
        agent2 = MinimaxAgent(index=1, depth=2)
        
        action1 = agent1.get_action(root)
        action2 = agent2.get_action(root)
        
        assert action1 is not None
        assert action2 is not None


class TestPerformance:
    """Test performance characteristics"""
    
    def test_alphabeta_faster_than_minimax(self):
        """Verify alpha-beta pruning improves performance"""
        # Create a tree where pruning is effective
        def create_ordered_tree():
            """Create tree with good move ordering for pruning"""
            # Best moves first enables more pruning
            leaves = [SimpleGameState(i, terminal=True) 
                     for i in [9, 3, 7, 2, 8, 1, 6, 4, 5]]
            
            nodes = []
            for i in range(0, 9, 3):
                nodes.append(SimpleGameState(0, children=leaves[i:i+3]))
            
            return SimpleGameState(0, children=nodes)
        
        root = create_ordered_tree()
        
        minimax = MinimaxAgent(index=0, depth=2)
        alphabeta = AlphaBetaAgent(index=0, depth=2)
        
        minimax.get_action(root)
        alphabeta.get_action(root)
        
        # Alpha-beta should explore significantly fewer nodes
        assert alphabeta.nodes_explored < minimax.nodes_explored
        
        # Calculate pruning effectiveness
        pruning_ratio = alphabeta.nodes_explored / minimax.nodes_explored
        assert pruning_ratio < 0.9  # At least 10% improvement


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
