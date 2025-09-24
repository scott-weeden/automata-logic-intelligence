"""Dynamic-programming algorithms for solving finite MDPs."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Optional, Sequence, Tuple, Any

from .mdp import MarkovDecisionProcess

State = Any
Action = Any


def value_iteration(
    mdp: MarkovDecisionProcess,
    discount: float = 0.9,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> Dict[State, float]:
    """Run value iteration and return the resulting state-value function."""

    values: Dict[State, float] = {state: 0.0 for state in mdp.get_states()}

    for _ in range(iterations):
        delta = 0.0
        new_values = values.copy()
        for state in mdp.get_states():
            if mdp.is_terminal(state):
                new_values[state] = 0.0
                continue

            actions = [action for action in mdp.get_possible_actions(state) if action is not None]
            if not actions:
                new_values[state] = 0.0
                continue

            q_values = [
                _q_value(mdp, state, action, values, discount)
                for action in actions
            ]
            best = max(q_values)
            new_values[state] = best
            delta = max(delta, abs(best - values[state]))
        values = new_values
        if delta < tolerance:
            break
    return values


def policy_evaluation(
    mdp: MarkovDecisionProcess,
    policy: Mapping[State, Optional[Action]],
    discount: float = 0.9,
    tolerance: float = 1e-6,
) -> Dict[State, float]:
    """Compute the value function for a fixed policy using iterative evaluation."""

    values: Dict[State, float] = {state: 0.0 for state in mdp.get_states()}

    while True:
        delta = 0.0
        new_values = values.copy()
        for state in mdp.get_states():
            if mdp.is_terminal(state):
                new_values[state] = 0.0
                continue

            action = policy.get(state)
            if action is None:
                new_values[state] = 0.0
                continue

            new_val = _q_value(mdp, state, action, values, discount)
            new_values[state] = new_val
            delta = max(delta, abs(new_val - values[state]))
        values = new_values
        if delta < tolerance:
            break
    return values


def extract_policy(
    mdp: MarkovDecisionProcess,
    values: Mapping[State, float],
    discount: float = 0.9,
) -> Dict[State, Optional[Action]]:
    """Derive a greedy policy from a state-value function."""

    policy: Dict[State, Optional[Action]] = {}
    for state in mdp.get_states():
        if mdp.is_terminal(state):
            policy[state] = None
            continue

        actions = [action for action in mdp.get_possible_actions(state) if action is not None]
        if not actions:
            policy[state] = None
            continue

        best_action = None
        best_value = float("-inf")
        for action in actions:
            q_val = _q_value(mdp, state, action, values, discount)
            if q_val > best_value:
                best_value = q_val
                best_action = action
        policy[state] = best_action
    return policy


def _q_value(
    mdp: MarkovDecisionProcess,
    state: State,
    action: Action,
    values: Mapping[State, float],
    discount: float,
) -> float:
    total = 0.0
    for next_state, prob in mdp.get_transition_states_and_probs(state, action):
        reward = mdp.get_reward(state, action, next_state)
        total += prob * (reward + discount * values.get(next_state, 0.0))
    return total


class ValueIterationAgent:
    """Solve an MDP with value iteration and expose the resulting policy."""

    def __init__(
        self,
        mdp: MarkovDecisionProcess,
        discount: float = 0.9,
        iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.tolerance = tolerance
        self.values = value_iteration(mdp, discount, iterations, tolerance)
        self.policy = extract_policy(mdp, self.values, discount)

    def get_value(self, state: State) -> float:
        return self.values.get(state, 0.0)

    def get_policy(self, state: State) -> Optional[Action]:
        return self.policy.get(state)


class PolicyIterationAgent:
    """Policy iteration agent that alternates evaluation and improvement."""

    def __init__(
        self,
        mdp: MarkovDecisionProcess,
        discount: float = 0.9,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
    ) -> None:
        self.mdp = mdp
        self.discount = discount
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.policy: Dict[State, Optional[Action]] = {}
        for state in mdp.get_states():
            if mdp.is_terminal(state):
                self.policy[state] = None
            else:
                actions = [action for action in mdp.get_possible_actions(state) if action is not None]
                self.policy[state] = actions[0] if actions else None

        self.values = {state: 0.0 for state in mdp.get_states()}
        self._run_policy_iteration()

    def _run_policy_iteration(self) -> None:
        for _ in range(self.max_iterations):
            self.values = policy_evaluation(self.mdp, self.policy, self.discount, self.tolerance)
            stable = True
            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    continue
                actions = [action for action in self.mdp.get_possible_actions(state) if action is not None]
                if not actions:
                    if self.policy.get(state) is not None:
                        self.policy[state] = None
                        stable = False
                    continue
                best_action = max(
                    actions,
                    key=lambda action: _q_value(self.mdp, state, action, self.values, self.discount),
                )
                if self.policy.get(state) != best_action:
                    self.policy[state] = best_action
                    stable = False
            if stable:
                break

    def get_value(self, state: State) -> float:
        return self.values.get(state, 0.0)

    def get_policy(self, state: State) -> Optional[Action]:
        return self.policy.get(state)
