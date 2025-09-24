"""Search problem interfaces and a simple grid-world implementation.

Phase 1 establishes the abstractions that every search algorithm in the project
relies on.  The :class:`SearchProblem` base class follows the conventional API
introduced in lectures: a problem supplies a start state, a goal test, and the
successor function returning ``(state, action, cost)`` triples.  Algorithms
consume these methods without knowing the underlying domain.

A lightweight :class:`GridSearchProblem` is included for tests and demos.  It
models 4-connected movement on a 2D grid where cells marked with ``1`` or ``#``
are treated as obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Any

Successor = Tuple[Any, str, float]


class SearchProblem:
    """Abstract search problem definition.

    Subclasses describe problem-specific behaviour while search algorithms only
    depend on this interface.  Concrete implementations must override the first
    three methods; ``get_cost_of_actions`` has a sensible default that assumes a
    unit cost per action but can be specialised when required.
    """

    def get_start_state(self) -> Any:
        """Return the initial state for the search."""
        raise NotImplementedError

    def is_goal_state(self, state: Any) -> bool:
        """Return ``True`` when *state* satisfies the goal condition."""
        raise NotImplementedError

    def get_successors(self, state: Any) -> Iterable[Successor]:
        """Yield successor states as ``(state, action, step_cost)`` triples."""
        raise NotImplementedError

    def get_cost_of_actions(self, actions: Sequence[str]) -> float:
        """Return the cumulative cost for executing *actions*.

        The base implementation assumes every action has unit cost.  Concrete
        problems with varying costs should override this method.  The function
        validates the input closely because search code often uses it when
        verifying a proposed solution.
        """

        if actions is None:
            raise ValueError("actions cannot be None")

        cost = 0.0
        for action in actions:
            if action is None:
                raise ValueError("action sequence contains None")
            cost += 1.0
        return cost


@dataclass(frozen=True)
class GridSearchProblem(SearchProblem):
    """Simple path-finding problem on a 4-connected grid.

    Args:
        grid: 2D structure containing passability markers.  Elements equal to
            ``1`` or ``'#'`` are treated as blocked cells; any other value is
            considered traversable.
        start: ``(row, col)`` coordinates of the start state.
        goal: ``(row, col)`` coordinates of the goal state.

    The implementation keeps the grid immutable for hashing convenience and
    validates that ``start`` and ``goal`` reside inside the grid.
    """

    grid: Sequence[Sequence[Any]]
    start: Tuple[int, int]
    goal: Tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", tuple(tuple(row) for row in self.grid))
        rows = len(self.grid)
        if rows == 0:
            raise ValueError("grid must contain at least one row")
        cols = len(self.grid[0])
        if any(len(row) != cols for row in self.grid):
            raise ValueError("grid rows must all have the same length")

        for position_name, position in ("start", self.start), ("goal", self.goal):
            if not self._in_bounds(position):
                raise ValueError(f"{position_name} position {position} out of bounds")
            if self._is_blocked(position):
                raise ValueError(f"{position_name} position {position} is blocked")

    # Movement deltas paired with canonical action names.
    _MOVES: Tuple[Tuple[str, Tuple[int, int]], ...] = (
        ('UP', (-1, 0)),
        ('DOWN', (1, 0)),
        ('LEFT', (0, -1)),
        ('RIGHT', (0, 1)),
    )

    def _in_bounds(self, position: Tuple[int, int]) -> bool:
        row, col = position
        return 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0])

    def _is_blocked(self, position: Tuple[int, int]) -> bool:
        row, col = position
        value = self.grid[row][col]
        return value in {1, '#', True}

    def get_start_state(self) -> Tuple[int, int]:
        return self.start

    def is_goal_state(self, state: Tuple[int, int]) -> bool:
        return state == self.goal

    def get_successors(self, state: Tuple[int, int]) -> List[Successor]:
        if not self._in_bounds(state):
            raise ValueError(f"state {state} outside of grid")

        successors: List[Successor] = []
        row, col = state
        for action, (dr, dc) in self._MOVES:
            next_state = (row + dr, col + dc)
            if not self._in_bounds(next_state) or self._is_blocked(next_state):
                continue
            successors.append((next_state, action, 1.0))
        return successors

    def get_cost_of_actions(self, actions: Sequence[str]) -> float:
        # Grid paths incur unit cost per move but we still perform validation to
        # catch illegal action sequences early in debugging.
        return super().get_cost_of_actions(actions)
