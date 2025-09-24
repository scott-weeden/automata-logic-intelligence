"""Common heuristic functions for informed search algorithms."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple

Position = Tuple[float, float]


def _coerce_position(position: Iterable[float]) -> Position:
    try:
        x, y = position
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"position must be a length-2 iterable, got {position!r}") from exc
    return (float(x), float(y))


def manhattan_distance(pos1: Iterable[float], pos2: Iterable[float]) -> float:
    """Return the Manhattan (L1) distance between two points."""
    x1, y1 = _coerce_position(pos1)
    x2, y2 = _coerce_position(pos2)
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean_distance(pos1: Iterable[float], pos2: Iterable[float]) -> float:
    """Return the Euclidean (L2) distance between two points."""
    x1, y1 = _coerce_position(pos1)
    x2, y2 = _coerce_position(pos2)
    return math.hypot(x1 - x2, y1 - y2)


def chebyshev_distance(pos1: Iterable[float], pos2: Iterable[float]) -> float:
    """Return the Chebyshev (L∞) distance between two points."""
    x1, y1 = _coerce_position(pos1)
    x2, y2 = _coerce_position(pos2)
    return max(abs(x1 - x2), abs(y1 - y2))


class GridHeuristic:
    """Callable object selecting an appropriate grid heuristic.

    Args:
        goal: Target position on the grid.
        movement_type: ``'4-way'`` (Manhattan), ``'8-way'`` (Chebyshev), or
            ``'continuous'`` (Euclidean).
    """

    def __init__(self, goal: Iterable[float], movement_type: str = "4-way") -> None:
        self.goal = _coerce_position(goal)
        valid = {"4-way", "8-way", "continuous"}
        if movement_type not in valid:
            raise ValueError(f"movement_type must be one of {sorted(valid)}")
        self.movement_type = movement_type

    def __call__(self, state: Iterable[float], problem=None) -> float:
        position = _coerce_position(state)
        if self.movement_type == "4-way":
            return manhattan_distance(position, self.goal)
        if self.movement_type == "8-way":
            return chebyshev_distance(position, self.goal)
        return euclidean_distance(position, self.goal)


def null_heuristic(state, problem=None) -> float:
    """Heuristic that always returns zero."""
    return 0.0
