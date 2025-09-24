"""Search algorithms implemented over the shared problem abstractions."""

from __future__ import annotations

from collections import deque
import heapq
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Any

from .utils import Node

HeuristicFn = Callable[[Any, Any], float]


class SearchAgent:
    """Base class for all search agents."""

    def __init__(self) -> None:
        self.nodes_expanded = 0

    def reset_statistics(self) -> None:
        self.nodes_expanded = 0

    def search(self, problem):  # pragma: no cover - abstract interface
        raise NotImplementedError


class BreadthFirstSearch(SearchAgent):
    """Breadth-first tree search that guarantees shortest paths in step cost."""

    def search(self, problem) -> Optional[List[Any]]:
        self.reset_statistics()
        start_state = problem.get_start_state()
        if problem.is_goal_state(start_state):
            return []

        frontier = deque([Node(start_state)])
        frontier_states: Set[Any] = {start_state}
        explored: Set[Any] = set()

        while frontier:
            node = frontier.popleft()
            frontier_states.discard(node.state)

            if node.state in explored:
                continue

            if problem.is_goal_state(node.state):
                return node.solution()

            explored.add(node.state)
            self.nodes_expanded += 1

            for successor, action, cost in reversed(list(problem.get_successors(node.state))):
                if successor in explored or successor in frontier_states:
                    continue
                child = Node(successor, node, action, node.path_cost + cost)
                frontier.append(child)
                frontier_states.add(successor)
        return None


class DepthFirstSearch(SearchAgent):
    """Depth-first tree search.  Does not guarantee optimality."""

    def search(self, problem) -> Optional[List[Any]]:
        self.reset_statistics()
        start_node = Node(problem.get_start_state())
        visited: Set[Any] = set()

        def dfs(node: Node) -> Optional[List[Any]]:
            if problem.is_goal_state(node.state):
                return node.solution()

            visited.add(node.state)
            self.nodes_expanded += 1

            for successor, action, cost in problem.get_successors(node.state):
                if successor in visited:
                    continue
                child = Node(successor, node, action, node.path_cost + cost)
                result = dfs(child)
                if result is not None:
                    return result
            visited.remove(node.state)
            return None

        return dfs(start_node)


class UniformCostSearch(SearchAgent):
    """Uniform cost search that minimises path cost for positive step costs."""

    def search(self, problem) -> Optional[List[Any]]:
        self.reset_statistics()
        start_node = Node(problem.get_start_state())
        frontier: List[Tuple[float, int, Node]] = []
        counter = 0
        heapq.heappush(frontier, (0.0, counter, start_node))
        best_cost: Dict[Any, float] = {start_node.state: 0.0}

        while frontier:
            cost, _, node = heapq.heappop(frontier)
            if cost > best_cost.get(node.state, float("inf")):
                continue

            if problem.is_goal_state(node.state):
                return node.solution()

            self.nodes_expanded += 1

            for successor, action, step_cost in problem.get_successors(node.state):
                new_cost = node.path_cost + step_cost
                if new_cost < best_cost.get(successor, float("inf")):
                    best_cost[successor] = new_cost
                    counter += 1
                    child = Node(successor, node, action, new_cost)
                    heapq.heappush(frontier, (new_cost, counter, child))
        return None


class AStarSearch(SearchAgent):
    """A* search parameterised by an admissible/consistent heuristic."""

    def __init__(self, heuristic: Optional[HeuristicFn] = None) -> None:
        super().__init__()
        self.heuristic: HeuristicFn = heuristic or (lambda state, problem: 0.0)

    def search(self, problem) -> Optional[List[Any]]:
        self.reset_statistics()
        start_node = Node(problem.get_start_state())
        start_cost = self.heuristic(start_node.state, problem)
        frontier: List[Tuple[float, float, int, Node]] = []
        counter = 0
        heapq.heappush(frontier, (start_cost, 0.0, counter, start_node))
        best_cost: Dict[Any, float] = {start_node.state: 0.0}

        while frontier:
            f_cost, g_cost, _, node = heapq.heappop(frontier)
            if g_cost > best_cost.get(node.state, float("inf")):
                continue

            if problem.is_goal_state(node.state):
                return node.solution()

            self.nodes_expanded += 1

            for successor, action, step_cost in problem.get_successors(node.state):
                new_cost = node.path_cost + step_cost
                if new_cost < best_cost.get(successor, float("inf")):
                    best_cost[successor] = new_cost
                    heuristic_cost = self.heuristic(successor, problem)
                    counter += 1
                    child = Node(successor, node, action, new_cost)
                    heapq.heappush(frontier, (new_cost + heuristic_cost, new_cost, counter, child))
        return None


class GreedyBestFirstSearch(SearchAgent):
    """Greedy best-first search guided purely by the heuristic."""

    def __init__(self, heuristic: Optional[HeuristicFn] = None) -> None:
        super().__init__()
        self.heuristic: HeuristicFn = heuristic or (lambda state, problem: 0.0)

    def search(self, problem) -> Optional[List[Any]]:
        self.reset_statistics()
        start_node = Node(problem.get_start_state())
        frontier: List[Tuple[float, int, Node]] = []
        counter = 0
        start_priority = self.heuristic(start_node.state, problem)
        heapq.heappush(frontier, (start_priority, counter, start_node))
        seen: Set[Any] = set()

        while frontier:
            priority, _, node = heapq.heappop(frontier)
            if node.state in seen:
                continue

            if problem.is_goal_state(node.state):
                return node.solution()

            seen.add(node.state)
            self.nodes_expanded += 1

            for successor, action, step_cost in problem.get_successors(node.state):
                if successor in seen:
                    continue
                counter += 1
                child = Node(successor, node, action, node.path_cost + step_cost)
                heapq.heappush(frontier, (self.heuristic(successor, problem), counter, child))
        return None


class IterativeDeepeningSearch(SearchAgent):
    """Iterative deepening depth-first search."""

    def search(self, problem, max_depth: int = 50) -> Optional[List[Any]]:
        self.reset_statistics()
        start_node = Node(problem.get_start_state())

        for depth_limit in range(max_depth + 1):
            visited: Set[Any] = set()
            result = self._depth_limited_search(problem, start_node, depth_limit, visited)
            if result is not None:
                return result
        return None

    def _depth_limited_search(
        self,
        problem,
        node: Node,
        limit: int,
        visited: Set[Any],
    ) -> Optional[List[Any]]:
        if problem.is_goal_state(node.state):
            return node.solution()
        if limit == 0:
            return None

        visited.add(node.state)
        self.nodes_expanded += 1

        for successor, action, cost in problem.get_successors(node.state):
            if successor in visited:
                continue
            child = Node(successor, node, action, node.path_cost + cost)
            result = self._depth_limited_search(problem, child, limit - 1, visited)
            if result is not None:
                visited.remove(node.state)
                return result
        visited.remove(node.state)
        return None
