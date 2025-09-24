"""Robot navigation demo solved via value and policy iteration."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mdp import GridMDP, PolicyIterationAgent, ValueIterationAgent

Action = Optional[Tuple[int, int]]

ACTION_TO_SYMBOL = {
    (0, 1): '→',
    (0, -1): '←',
    (1, 0): '↓',
    (-1, 0): '↑',
    None: '·',
}


def _build_mdp() -> GridMDP:
    grid = [
        [-0.04, -0.04, -0.04, 1],
        [-0.04, None, -0.04, -1],
        [-0.04, -0.04, -0.04, -0.04],
    ]
    terminals = {(0, 3), (1, 3)}
    return GridMDP(grid=grid, terminals=terminals, init=(2, 0), gamma=0.9)


def _policy_to_grid(mdp: GridMDP, policy: Dict[Tuple[int, int], Action]) -> List[List[str]]:
    result = []
    for r, row in enumerate(mdp.grid):
        display_row: List[str] = []
        for c, cell in enumerate(row):
            if cell is None:
                display_row.append('#')
            else:
                action = policy.get((r, c))
                display_row.append(ACTION_TO_SYMBOL.get(action, '?'))
        result.append(display_row)
    return result


def analyse_robot_navigation(show_output: bool = True):
    mdp = _build_mdp()
    vi_agent = ValueIterationAgent(mdp, discount=0.9, iterations=100)
    pi_agent = PolicyIterationAgent(mdp, discount=0.9)

    vi_policy_grid = _policy_to_grid(mdp, vi_agent.policy)
    pi_policy_grid = _policy_to_grid(mdp, pi_agent.policy)
    vi_values = mdp.to_grid(vi_agent.values)

    if show_output:
        print("Value Iteration Policy:")
        for row in vi_policy_grid:
            print(' '.join(row))
        print("\nPolicy Iteration Policy:")
        for row in pi_policy_grid:
            print(' '.join(row))
        print("\nValue Function:")
        for row in vi_values:
            formatted = ["#" if value is None else f"{value:5.2f}" for value in row]
            print(' '.join(formatted))

    return {
        "value_iteration": {
            "values": vi_values,
            "policy": vi_policy_grid,
        },
        "policy_iteration": {
            "policy": pi_policy_grid,
        },
    }


if __name__ == "__main__":  # pragma: no cover - manual demo
    analyse_robot_navigation(show_output=True)
