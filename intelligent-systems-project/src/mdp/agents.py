"""Legacy compatibility layer for MDP agents.

The primary implementations now live in :mod:`src.mdp.value_iteration` and are
re-exported here for backwards compatibility with earlier course materials.
"""

from .mdp import MarkovDecisionProcess  # noqa: F401 - re-exported name
from .value_iteration import PolicyIterationAgent, ValueIterationAgent  # noqa: F401

__all__ = [
    "MarkovDecisionProcess",
    "ValueIterationAgent",
    "PolicyIterationAgent",
]
