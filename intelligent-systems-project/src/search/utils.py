"""Utility classes used by informed search algorithms."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence


@dataclass(frozen=True)
class Node:
    """Representation of a node in the search tree."""

    state: Any
    parent: Optional["Node"] = None
    action: Optional[Any] = None
    path_cost: float = 0.0
    depth: int = field(init=False)

    def __post_init__(self) -> None:
        depth = 0 if self.parent is None else self.parent.depth + 1
        object.__setattr__(self, "depth", depth)

    def expand(self, problem) -> List["Node"]:
        """Return child nodes generated from this node."""
        children: List[Node] = []
        for successor, action, cost in problem.get_successors(self.state):
            child = Node(successor, self, action, self.path_cost + cost)
            children.append(child)
        return children

    def solution(self) -> List[Any]:
        """Return the list of actions that reaches this node from the root."""
        actions: List[Any] = []
        node: Optional[Node] = self
        while node and node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions

    def path(self) -> List[Any]:
        """Return the list of states from the root to this node."""
        states: List[Any] = []
        node: Optional[Node] = self
        while node is not None:
            states.append(node.state)
            node = node.parent
        states.reverse()
        return states

    def __lt__(self, other: "Node") -> bool:
        return self.path_cost < other.path_cost

    def __hash__(self) -> int:  # pragma: no cover - trivial wrapper
        return hash(self.state)


class PriorityQueue:
    """A heap-backed priority queue with membership helpers.

    The implementation mirrors the interface used in Berkeley's Pac-Man
    projects: ``append`` inserts elements, ``pop`` removes the element with the
    smallest (or largest) priority, and the container provides membership tests
    and item retrieval by key.
    """

    def __init__(self, order: str = "min", f: Callable[[Any], float] = lambda x: x):
        if order not in {"min", "max"}:
            raise ValueError("order must be either 'min' or 'max'")
        self.order = order
        self.f = f
        self._heap: List[tuple[float, int, Any]] = []
        self._counter = itertools.count()

    def append(self, item: Any) -> None:
        priority = self.f(item)
        if self.order == "max":
            priority = -priority
        entry = (priority, next(self._counter), item)
        heapq.heappush(self._heap, entry)

    def extend(self, items: Iterable[Any]) -> None:
        for item in items:
            self.append(item)

    def pop(self) -> Any:
        if not self._heap:
            raise IndexError("pop from empty PriorityQueue")
        _, _, item = heapq.heappop(self._heap)
        return item

    def __len__(self) -> int:
        return len(self._heap)

    def _matches(self, item: Any, key: Any) -> bool:
        if item == key:
            return True
        if hasattr(item, "state") and item.state == key:
            return True
        if isinstance(item, tuple) and len(item) >= 2 and item[1] == key:
            return True
        return False

    def __contains__(self, key: Any) -> bool:
        return any(self._matches(item, key) for _, _, item in self._heap)

    def __getitem__(self, key: Any) -> Any:
        for _, _, item in self._heap:
            if self._matches(item, key):
                return item
        raise KeyError(f"{key!r} not found in priority queue")

    def __delitem__(self, key: Any) -> None:
        for index, (_, _, item) in enumerate(self._heap):
            if self._matches(item, key):
                del self._heap[index]
                heapq.heapify(self._heap)
                return
        raise KeyError(f"{key!r} not found in priority queue")

    def __iter__(self) -> Iterator[Any]:  # pragma: no cover - helper for debugging
        return (item for _, _, item in sorted(self._heap))
