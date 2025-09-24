"""Tabular reinforcement learning agents (Q-Learning and SARSA)."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

State = Any
Action = Any
ActionFn = Callable[[State], Sequence[Action]]


class QLearningAgent:
    """Epsilon-greedy Q-learning agent operating on a discrete action space."""

    def __init__(
        self,
        action_fn: ActionFn,
        discount: float = 0.9,
        alpha: float = 0.5,
        epsilon: float = 0.1,
    ) -> None:
        self.action_fn = action_fn
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.training = True

        self.q_values: Dict[tuple[State, Action], float] = defaultdict(float)
        self.episode_rewards: float = 0.0
        self.episodes_completed: int = 0

    # ------------------------------------------------------------------
    # Utility helpers

    def _available_actions(self, state: State) -> List[Action]:
        actions = self.action_fn(state)
        return list(actions) if actions is not None else []

    def get_q_value(self, state: State, action: Action) -> float:
        return self.q_values[(state, action)]

    def get_max_q_value(self, state: State) -> float:
        actions = self._available_actions(state)
        if not actions:
            return 0.0
        return max(self.get_q_value(state, action) for action in actions)

    def compute_action_from_q_values(self, state: State) -> Optional[Action]:
        actions = self._available_actions(state)
        if not actions:
            return None
        best_value = float("-inf")
        best_actions: List[Action] = []
        for action in actions:
            value = self.get_q_value(state, action)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        return random.choice(best_actions)

    def get_action(self, state: State) -> Optional[Action]:
        actions = self._available_actions(state)
        if not actions:
            return None
        if not self.training:
            return self.compute_action_from_q_values(state)
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self.compute_action_from_q_values(state)

    # ------------------------------------------------------------------
    # Learning interface

    def start_episode(self) -> None:
        self.episode_rewards = 0.0

    def stop_training(self) -> None:
        self.training = False

    def update(self, state: State, action: Action, next_state: State, reward: float) -> None:
        actions = self._available_actions(next_state)
        max_next_q = max((self.get_q_value(next_state, a) for a in actions), default=0.0)
        current_q = self.get_q_value(state, action)
        target = reward + self.discount * max_next_q
        self.q_values[(state, action)] = current_q + self.alpha * (target - current_q)
        self.episode_rewards += reward


class SARSAAgent(QLearningAgent):
    """On-policy SARSA agent sharing the Q-table implementation."""

    def __init__(
        self,
        action_fn: ActionFn,
        discount: float = 0.9,
        alpha: float = 0.5,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(action_fn, discount, alpha, epsilon)
        self.next_action: Optional[Action] = None

    def update(self, state: State, action: Action, next_state: State, reward: float) -> None:
        if self.next_action is None:
            next_q = 0.0
        else:
            next_q = self.get_q_value(next_state, self.next_action)
        current_q = self.get_q_value(state, action)
        target = reward + self.discount * next_q
        self.q_values[(state, action)] = current_q + self.alpha * (target - current_q)
        self.episode_rewards += reward
