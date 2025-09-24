"""Pathfinding demonstration showcasing core search algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from search import (
    AStarSearch,
    BreadthFirstSearch,
    DepthFirstSearch,
    GridHeuristic,
    GridSearchProblem,
    GreedyBestFirstSearch,
    UniformCostSearch,
)

Coordinate = Tuple[int, int]
Action = str

ACTION_TO_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


@dataclass
class DemoResult:
    algorithm: str
    path: List[Action]
    nodes_expanded: int


def _build_problem() -> GridSearchProblem:
    grid = [
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    return GridSearchProblem(grid, start=(0, 0), goal=(7, 7))


def _trace_path(start: Coordinate, actions: Iterable[Action]) -> List[Coordinate]:
    row, col = start
    path = [start]
    for action in actions:
        dr, dc = ACTION_TO_DELTA[action]
        row += dr
        col += dc
        path.append((row, col))
    return path


def solve_demo() -> List[DemoResult]:
    problem = _build_problem()
    heuristic = GridHeuristic(problem.goal, "4-way")

    agents = [
        ("Breadth-First Search", BreadthFirstSearch()),
        ("Depth-First Search", DepthFirstSearch()),
        ("Uniform Cost Search", UniformCostSearch()),
        ("Greedy Best-First", GreedyBestFirstSearch(heuristic=heuristic)),
        ("A* Search", AStarSearch(heuristic=heuristic)),
    ]

    results: List[DemoResult] = []
    for label, agent in agents:
        actions = agent.search(problem) or []
        results.append(DemoResult(label, actions, agent.nodes_expanded))
    return results


def run_pathfinding_demo(show_output: bool = True) -> List[DemoResult]:
    results = solve_demo()
    if show_output:
        problem = _build_problem()
        print("Start -> Goal:", problem.get_start_state(), "->", problem.goal)
        for result in results:
            path_coordinates = _trace_path(problem.get_start_state(), result.path)
            print(f"{result.algorithm:20} | steps: {len(result.path):2} | nodes expanded: {result.nodes_expanded:3}")
            print("  path:", path_coordinates)
    return results


if __name__ == "__main__":  # pragma: no cover - manual demo
    run_pathfinding_demo(show_output=True)
