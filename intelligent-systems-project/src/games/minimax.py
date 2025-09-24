"""
Minimax Algorithm for Game Playing

Implements minimax decision making for two-player zero-sum games.
Based on CS 5368 Week 4-5 material on adversarial search.

Minimax assumes:
- Perfect information (both players see full game state)
- Deterministic games (no chance elements)
- Zero-sum (one player's gain = other's loss)
- Rational opponents (both play optimally)
"""

import math

def minimax_decision(game_state, game):
    """
    Return best action for current player using minimax algorithm.
    
    Args:
        game_state: Current state of the game
        game: Game object with methods for actions, result, terminal_test, utility
    
    Returns:
        Best action according to minimax principle
    """
    def max_value(state):
        """Return maximum utility value for MAX player."""
        if game.terminal_test(state):
            return game.utility(state, game.to_move(game_state))
        
        v = -math.inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action)))
        return v
    
    def min_value(state):
        """Return minimum utility value for MIN player."""
        if game.terminal_test(state):
            return game.utility(state, game.to_move(game_state))
        
        v = math.inf
        for action in game.actions(state):
            v = min(v, max_value(game.result(state, action)))
        return v
    
    # Find action that leads to best outcome for current player
    current_player = game.to_move(game_state)
    best_action = None
    best_value = -math.inf
    
    for action in game.actions(game_state):
        next_state = game.result(game_state, action)
        action_value = min_value(next_state)
        
        if action_value > best_value:
            best_value = action_value
            best_action = action
    
    return best_action

def minimax_cutoff_decision(game_state, game, depth_limit=4, eval_fn=None):
    """
    Minimax with depth cutoff for practical game playing.
    Uses evaluation function when depth limit reached.
    
    Args:
        game_state: Current game state
        game: Game object
        depth_limit: Maximum search depth
        eval_fn: Evaluation function for non-terminal states
    
    Returns:
        Best action within depth limit
    """
    if eval_fn is None:
        eval_fn = lambda state, player: 0  # Default neutral evaluation
    
    def max_value(state, depth):
        """Max value with depth cutoff."""
        if game.terminal_test(state):
            return game.utility(state, game.to_move(game_state))
        if depth >= depth_limit:
            return eval_fn(state, game.to_move(game_state))
        
        v = -math.inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action), depth + 1))
        return v
    
    def min_value(state, depth):
        """Min value with depth cutoff."""
        if game.terminal_test(state):
            return game.utility(state, game.to_move(game_state))
        if depth >= depth_limit:
            return eval_fn(state, game.to_move(game_state))
        
        v = math.inf
        for action in game.actions(state):
            v = min(v, max_value(game.result(state, action), depth + 1))
        return v
    
    # Find best action within depth limit
    best_action = None
    best_value = -math.inf
    
    for action in game.actions(game_state):
        next_state = game.result(game_state, action)
        action_value = min_value(next_state, 1)
        
        if action_value > best_value:
            best_value = action_value
            best_action = action
    
    return best_action

class MinimaxAgent:
    """
    Agent that uses minimax algorithm for decision making.
    Can be configured with depth limits and evaluation functions.
    """
    
    def __init__(self, depth_limit=None, eval_fn=None):
        """
        Initialize minimax agent.
        
        Args:
            depth_limit: Maximum search depth (None for full search)
            eval_fn: Evaluation function for non-terminal positions
        """
        self.depth_limit = depth_limit
        self.eval_fn = eval_fn
    
    def get_action(self, game_state, game):
        """Get best action using minimax algorithm."""
        if self.depth_limit is None:
            return minimax_decision(game_state, game)
        else:
            return minimax_cutoff_decision(
                game_state, game, self.depth_limit, self.eval_fn
            )
