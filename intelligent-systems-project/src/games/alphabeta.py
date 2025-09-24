"""
Alpha-Beta Pruning for Game Trees

Optimized minimax that eliminates branches that cannot affect final decision.
Based on CS 5368 Week 4-5 material on adversarial search optimization.

Key insight: If we know MAX can get at least α and MIN can get at most β,
then if α ≥ β, we can prune remaining branches.

Best case: O(b^(d/2)) instead of O(b^d) with perfect move ordering.
"""

import math

def alphabeta_decision(game_state, game):
    """
    Return best action using alpha-beta pruning.
    
    Args:
        game_state: Current game state
        game: Game object with required methods
    
    Returns:
        Best action according to minimax with alpha-beta pruning
    """
    def max_value(state, alpha, beta):
        """Return max value with alpha-beta pruning."""
        if game.terminal_test(state):
            return game.utility(state, game.to_move(game_state))
        
        v = -math.inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action), alpha, beta))
            if v >= beta:
                return v  # Beta cutoff
            alpha = max(alpha, v)
        return v
    
    def min_value(state, alpha, beta):
        """Return min value with alpha-beta pruning."""
        if game.terminal_test(state):
            return game.utility(state, game.to_move(game_state))
        
        v = math.inf
        for action in game.actions(state):
            v = min(v, max_value(game.result(state, action), alpha, beta))
            if v <= alpha:
                return v  # Alpha cutoff
            beta = min(beta, v)
        return v
    
    # Find best action with alpha-beta pruning
    best_action = None
    best_value = -math.inf
    alpha = -math.inf
    beta = math.inf
    
    for action in game.actions(game_state):
        next_state = game.result(game_state, action)
        action_value = min_value(next_state, alpha, beta)
        
        if action_value > best_value:
            best_value = action_value
            best_action = action
        
        alpha = max(alpha, action_value)
    
    return best_action

def alphabeta_cutoff_decision(game_state, game, depth_limit=4, eval_fn=None):
    """
    Alpha-beta pruning with depth cutoff and evaluation function.
    
    Args:
        game_state: Current game state
        game: Game object
        depth_limit: Maximum search depth
        eval_fn: Evaluation function for non-terminal states
    
    Returns:
        Best action within depth limit using alpha-beta pruning
    """
    if eval_fn is None:
        eval_fn = lambda state, player: 0
    
    def max_value(state, alpha, beta, depth):
        """Max value with alpha-beta pruning and depth cutoff."""
        if game.terminal_test(state):
            return game.utility(state, game.to_move(game_state))
        if depth >= depth_limit:
            return eval_fn(state, game.to_move(game_state))
        
        v = -math.inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action), alpha, beta, depth + 1))
            if v >= beta:
                return v  # Beta cutoff
            alpha = max(alpha, v)
        return v
    
    def min_value(state, alpha, beta, depth):
        """Min value with alpha-beta pruning and depth cutoff."""
        if game.terminal_test(state):
            return game.utility(state, game.to_move(game_state))
        if depth >= depth_limit:
            return eval_fn(state, game.to_move(game_state))
        
        v = math.inf
        for action in game.actions(state):
            v = min(v, max_value(game.result(state, action), alpha, beta, depth + 1))
            if v <= alpha:
                return v  # Alpha cutoff
            beta = min(beta, v)
        return v
    
    # Find best action with alpha-beta pruning and depth limit
    best_action = None
    best_value = -math.inf
    alpha = -math.inf
    beta = math.inf
    
    for action in game.actions(game_state):
        next_state = game.result(game_state, action)
        action_value = min_value(next_state, alpha, beta, 1)
        
        if action_value > best_value:
            best_value = action_value
            best_action = action
        
        alpha = max(alpha, action_value)
    
    return best_action

class AlphaBetaAgent:
    """
    Game-playing agent using alpha-beta pruning.
    More efficient than pure minimax for deep searches.
    """
    
    def __init__(self, depth_limit=6, eval_fn=None, move_ordering=None):
        """
        Initialize alpha-beta agent.
        
        Args:
            depth_limit: Maximum search depth
            eval_fn: Position evaluation function
            move_ordering: Function to order moves for better pruning
        """
        self.depth_limit = depth_limit
        self.eval_fn = eval_fn
        self.move_ordering = move_ordering
        self.nodes_explored = 0
    
    def get_action(self, game_state, game):
        """Get best action using alpha-beta pruning."""
        self.nodes_explored = 0
        
        if self.depth_limit is None:
            return alphabeta_decision(game_state, game)
        else:
            return alphabeta_cutoff_decision(
                game_state, game, self.depth_limit, self.eval_fn
            )
    
    def order_moves(self, game_state, actions, game):
        """
        Order moves to improve alpha-beta pruning efficiency.
        Good moves first leads to more cutoffs.
        """
        if self.move_ordering is None:
            return actions
        
        return sorted(actions, key=lambda a: self.move_ordering(game_state, a, game), reverse=True)
