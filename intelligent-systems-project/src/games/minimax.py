"""Adversarial search agents built on the core game abstractions."""

from __future__ import annotations

import math
from .game_state import GameAgent, GameState


class MinimaxAgent(GameAgent):
    """Classic minimax search assuming a perfect opponent."""

    def __init__(self, index: int = 0, depth: int = 2) -> None:
        super().__init__(index)
        self.depth = depth

    def get_action(self, game_state: GameState):
        self.reset_statistics()

        def minimax(state: GameState, depth: int, agent_index: int) -> float:
            self.nodes_explored += 1
            if state.is_terminal() or depth == 0:
                return state.get_utility(self.index)

            legal_actions = state.get_legal_actions(agent_index)
            if agent_index == self.index:
                value = -math.inf
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, minimax(successor, depth - 1, 1 - agent_index))
                return value
            value = math.inf
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                value = min(value, minimax(successor, depth - 1, 1 - agent_index))
            return value

        best_action = None
        best_value = -math.inf
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            value = minimax(successor, self.depth - 1, 1 - self.index)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action


class AlphaBetaAgent(GameAgent):
    """Minimax variant with alpha-beta pruning."""

    def __init__(self, index: int = 0, depth: int = 2) -> None:
        super().__init__(index)
        self.depth = depth

    def get_action(self, game_state: GameState):
        self.reset_statistics()

        def alphabeta(state: GameState, depth: int, agent_index: int, alpha: float, beta: float) -> float:
            self.nodes_explored += 1
            if state.is_terminal() or depth == 0:
                return state.get_utility(self.index)

            legal_actions = state.get_legal_actions(agent_index)
            if agent_index == self.index:
                value = -math.inf
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, alphabeta(successor, depth - 1, 1 - agent_index, alpha, beta))
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
                return value

            value = math.inf
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                value = min(value, alphabeta(successor, depth - 1, 1 - agent_index, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

        best_action = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            value = alphabeta(successor, self.depth - 1, 1 - self.index, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)
        return best_action


class ExpectimaxAgent(GameAgent):
    """Expectimax agent modelling non-deterministic opponents."""

    def __init__(self, index: int = 0, depth: int = 2) -> None:
        super().__init__(index)
        self.depth = depth

    def get_action(self, game_state: GameState):
        self.reset_statistics()

        def expectimax(state: GameState, depth: int, agent_index: int) -> float:
            self.nodes_explored += 1
            if state.is_terminal() or depth == 0:
                return state.get_utility(self.index)

            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return state.get_utility(self.index)

            if agent_index == self.index:
                value = -math.inf
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, expectimax(successor, depth - 1, 1 - agent_index))
                return value

            total = 0.0
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                total += expectimax(successor, depth - 1, 1 - agent_index)
            return total / len(legal_actions)

        best_action = None
        best_value = -math.inf
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            value = expectimax(successor, self.depth - 1, 1 - self.index)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
