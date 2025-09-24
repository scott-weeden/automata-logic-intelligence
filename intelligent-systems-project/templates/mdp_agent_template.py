"""MDP agent template.

This file sketches out the structure of a planning agent that relies on the
value- and policy-iteration utilities in :mod:`src.mdp`.  Extend it with domain
specific behaviour, convergence checks, or logging as required by coursework.
"""

from __future__ import annotations

from typing import Dict, Optional

from src.mdp import MarkovDecisionProcess, ValueIterationAgent


class CustomMDPAgent(ValueIterationAgent):
    """Start from value iteration and override hooks as needed."""

    def __init__(
        self,
        mdp: MarkovDecisionProcess,
        discount: float = 0.95,
        iterations: int = 100,
    ) -> None:
        super().__init__(mdp, discount=discount, iterations=iterations)

    def post_process(self, values: Dict) -> None:  # pragma: no cover - template
        """Hook for analysing value updates after convergence."""
        # TODO: add custom analysis or logging here.
        pass
