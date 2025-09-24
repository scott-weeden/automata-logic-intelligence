"""Game-playing agent template.

The template demonstrates how to structure minimax-style agents that work with
:mod:`src.games`.  Plug in your evaluation heuristics and search depth to build
custom behaviour for assignments or experiments.
"""

from __future__ import annotations

from typing import Optional

from src.games import GameAgent, GameState


class CustomGameAgent(GameAgent):
    """Skeleton agent that students can extend with their own logic."""

    def __init__(self, index: int = 0, depth: int = 2) -> None:
        super().__init__(index=index)
        self.depth = depth

    def evaluate(self, state: GameState) -> float:  # pragma: no cover - template hook
        """Supply a heuristic evaluation function for non-terminal states."""
        raise NotImplementedError("Provide a domain-specific evaluation function")

    def get_action(self, game_state: GameState):  # pragma: no cover - template hook
        """Implement the decision procedure (e.g., minimax or alpha-beta)."""
        raise NotImplementedError("Decision logic not implemented")
