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

class GameState:
    """Base class for game states."""
    
    def get_legal_actions(self, agent_index=0):
        """Return list of legal actions for agent."""
        raise NotImplementedError
    
    def generate_successor(self, agent_index, action):
        """Return successor state after agent takes action."""
        raise NotImplementedError
    
    def is_terminal(self):
        """Return True if game is over."""
        raise NotImplementedError
    
    def get_utility(self, agent_index=0):
        """Return utility for agent in terminal state."""
        raise NotImplementedError

class GameAgent:
    """Base class for game agents."""
    
    def __init__(self, index=0):
        self.index = index
        self.nodes_explored = 0
    
    def get_action(self, game_state):
        """Return action for current game state."""
        raise NotImplementedError

class MinimaxAgent(GameAgent):
    """Agent that uses minimax algorithm for decision making."""
    
    def __init__(self, index=0, depth=2):
        super().__init__(index)
        self.depth = depth
    
    def get_action(self, game_state):
        """Get best action using minimax algorithm."""
        self.nodes_explored = 0
        
        def minimax(state, depth, agent_index):
            self.nodes_explored += 1
            
            if state.is_terminal() or depth == 0:
                return state.get_utility(self.index)
            
            if agent_index == self.index:  # Maximizing player
                max_eval = -math.inf
                for action in state.get_legal_actions(agent_index):
                    successor = state.generate_successor(agent_index, action)
                    eval_score = minimax(successor, depth - 1, 1 - agent_index)
                    max_eval = max(max_eval, eval_score)
                return max_eval
            else:  # Minimizing player
                min_eval = math.inf
                for action in state.get_legal_actions(agent_index):
                    successor = state.generate_successor(agent_index, action)
                    eval_score = minimax(successor, depth - 1, 1 - agent_index)
                    min_eval = min(min_eval, eval_score)
                return min_eval
        
        # Find best action
        best_action = None
        best_value = -math.inf
        
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            action_value = minimax(successor, self.depth - 1, 1 - self.index)
            
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        return best_action

class AlphaBetaAgent(GameAgent):
    """Game-playing agent using alpha-beta pruning."""
    
    def __init__(self, index=0, depth=2):
        super().__init__(index)
        self.depth = depth
    
    def get_action(self, game_state):
        """Get best action using alpha-beta pruning."""
        self.nodes_explored = 0
        
        def alphabeta(state, depth, alpha, beta, agent_index):
            self.nodes_explored += 1
            
            if state.is_terminal() or depth == 0:
                return state.get_utility(self.index)
            
            if agent_index == self.index:  # Maximizing player
                max_eval = -math.inf
                for action in state.get_legal_actions(agent_index):
                    successor = state.generate_successor(agent_index, action)
                    eval_score = alphabeta(successor, depth - 1, alpha, beta, 1 - agent_index)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff
                return max_eval
            else:  # Minimizing player
                min_eval = math.inf
                for action in state.get_legal_actions(agent_index):
                    successor = state.generate_successor(agent_index, action)
                    eval_score = alphabeta(successor, depth - 1, alpha, beta, 1 - agent_index)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff
                return min_eval
        
        # Find best action
        best_action = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            action_value = alphabeta(successor, self.depth - 1, alpha, beta, 1 - self.index)
            
            if action_value > best_value:
                best_value = action_value
                best_action = action
            
            alpha = max(alpha, action_value)
        
        return best_action

class ExpectimaxAgent(GameAgent):
    """Agent using expectimax for games with chance elements."""
    
    def __init__(self, index=0, depth=2):
        super().__init__(index)
        self.depth = depth
    
    def get_action(self, game_state):
        """Get best action using expectimax algorithm."""
        self.nodes_explored = 0
        
        def expectimax(state, depth, agent_index):
            self.nodes_explored += 1
            
            if state.is_terminal() or depth == 0:
                return state.get_utility(self.index)
            
            if agent_index == self.index:  # Maximizing player
                max_eval = -math.inf
                for action in state.get_legal_actions(agent_index):
                    successor = state.generate_successor(agent_index, action)
                    eval_score = expectimax(successor, depth - 1, 1 - agent_index)
                    max_eval = max(max_eval, eval_score)
                return max_eval
            else:  # Chance node (expectation)
                actions = state.get_legal_actions(agent_index)
                if not actions:
                    return state.get_utility(self.index)
                
                total_value = 0
                for action in actions:
                    successor = state.generate_successor(agent_index, action)
                    total_value += expectimax(successor, depth - 1, 1 - agent_index)
                return total_value / len(actions)
        
        # Find best action
        best_action = None
        best_value = -math.inf
        
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            action_value = expectimax(successor, self.depth - 1, 1 - self.index)
            
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        return best_action
