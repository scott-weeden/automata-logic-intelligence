"""Core Markov Decision Process abstractions and a grid-world example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

Action = Any
State = Any
Transition = List[Tuple[State, float]]


class MarkovDecisionProcess:
    """Abstract interface expected by dynamic-programming based solvers."""

    def get_states(self) -> Sequence[State]:  # pragma: no cover - documentation only
        raise NotImplementedError

    def get_start_state(self) -> State:  # pragma: no cover - documentation only
        raise NotImplementedError

    def get_possible_actions(self, state: State) -> Sequence[Action]:  # pragma: no cover - documentation only
        raise NotImplementedError

    def get_transition_states_and_probs(self, state: State, action: Action) -> Sequence[Tuple[State, float]]:  # pragma: no cover
        raise NotImplementedError

    def get_reward(self, state: State, action: Action, next_state: State) -> float:  # pragma: no cover
        raise NotImplementedError

    def is_terminal(self, state: State) -> bool:  # pragma: no cover - documentation only
        raise NotImplementedError


class MDP(MarkovDecisionProcess):
    """Finite MDP with optionally stochastic transitions and state-based rewards."""

    def __init__(
        self,
        init: State,
        actlist: Sequence[Action] | Mapping[State, Sequence[Action]] | Callable[[State], Sequence[Action]] | None,
        terminals: Iterable[State] | None = None,
        transitions: Optional[Mapping[Any, Any]] = None,
        reward: Optional[Callable[..., float] | Mapping[Any, float]] = None,
        gamma: float = 0.9,
    ) -> None:
        self.init = init
        self.gamma = gamma
        self.terminals = set(terminals or [])
        self.reward = reward or {}
        self._states: List[State] = []
        self._default_actions: Optional[List[Action]] = None
        self._actions: Dict[State, List[Action]] = {}
        self._transitions: Dict[Tuple[State, Action], Transition] = {}
        self._action_source = actlist

        self._add_state(init)
        for terminal in self.terminals:
            self._add_state(terminal)

        # Record explicit actions when provided.
        if isinstance(actlist, Mapping):
            for state, actions in actlist.items():
                self._add_state(state)
                self._actions[state] = list(actions)
        elif isinstance(actlist, Sequence) and actlist is not None:
            self._default_actions = list(actlist)
        else:
            self._default_actions = None

        # Populate transitions.
        if transitions:
            self._ingest_transitions(transitions)

        # Ensure reward-defined states are tracked.
        if isinstance(self.reward, Mapping):
            for key in self.reward.keys():
                if isinstance(key, tuple) and len(key) == 3:
                    self._add_state(key[0])
                    self._add_state(key[2])
                else:
                    self._add_state(key)

        self.states: Tuple[State, ...] = tuple(self._states)

    # ------------------------------------------------------------------
    # Helpers

    def _add_state(self, state: State) -> None:
        if state not in self._states:
            self._states.append(state)

    def _normalise_transition_list(self, pairs: Iterable[Tuple[Any, Any]]) -> Transition:
        normalised: Dict[State, float] = {}
        for entry in pairs:
            if len(entry) != 2:
                raise ValueError("transition entries must have length two")
            first, second = entry
            # Accept either (next_state, prob) or (prob, next_state).
            if isinstance(first, (int, float)) and not isinstance(second, (int, float)):
                prob = float(first)
                next_state = second
            else:
                next_state = first
                prob = float(second)
            normalised[next_state] = normalised.get(next_state, 0.0) + prob
            self._add_state(next_state)
        return [(state, normalised[state]) for state in normalised]

    def _register_transition(self, state: State, action: Action, pairs: Iterable[Tuple[Any, Any]]) -> None:
        self._add_state(state)
        canonical = self._normalise_transition_list(pairs)
        self._transitions[(state, action)] = canonical
        if state not in self._actions:
            self._actions[state] = []
        if action not in self._actions[state]:
            self._actions[state].append(action)

    def _ingest_transitions(self, transitions: Mapping[Any, Any]) -> None:
        for key, value in transitions.items():
            if isinstance(key, tuple) and len(key) == 2 and not isinstance(value, Mapping):
                state, action = key
                self._register_transition(state, action, value)
            else:
                state = key
                actions_map = value
                if not isinstance(actions_map, Mapping):
                    raise ValueError("nested transition map must be a mapping from actions to outcomes")
                for action, outcomes in actions_map.items():
                    self._register_transition(state, action, outcomes)

    # ------------------------------------------------------------------
    # MarkovDecisionProcess API

    def get_states(self) -> Sequence[State]:
        return list(self.states)

    def get_start_state(self) -> State:
        return self.init

    def actions(self, state: State) -> List[Action]:
        if state in self.terminals:
            return []
        if state in self._actions:
            return list(self._actions[state])
        if isinstance(self._action_source, Callable):
            return list(self._action_source(state))
        if self._default_actions is not None:
            return list(self._default_actions)
        return []

    def get_possible_actions(self, state: State) -> List[Action]:
        return self.actions(state)

    def T(self, state: State, action: Action) -> Transition:
        return self._transitions.get((state, action), [])

    def get_transition_states_and_probs(self, state: State, action: Action) -> Transition:
        return self.T(state, action)

    def R(self, state: State) -> float:
        if callable(self.reward):
            try:
                return float(self.reward(state))
            except TypeError:
                return float(self.reward(state, None, None))
        if isinstance(self.reward, Mapping):
            return float(self.reward.get(state, 0.0))
        return 0.0

    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        if callable(self.reward):
            try:
                return float(self.reward(state, action, next_state))
            except TypeError:
                return float(self.reward(next_state))
        if isinstance(self.reward, Mapping):
            triplet = (state, action, next_state)
            if triplet in self.reward:
                return float(self.reward[triplet])
            if next_state in self.reward:
                return float(self.reward[next_state])
            if state in self.reward:
                return float(self.reward[state])
        return 0.0

    def is_terminal(self, state: State) -> bool:
        return state in self.terminals


@dataclass
class GridMDP(MDP):
    """Classic stochastic grid-world environment."""

    grid: Sequence[Sequence[Optional[float]]]
    terminals: Iterable[Tuple[int, int]]
    init: Tuple[int, int] = (0, 0)
    gamma: float = 0.9

    def __post_init__(self) -> None:
        valid_states: List[Tuple[int, int]] = []
        reward: Dict[Tuple[int, int], float] = {}
        for r, row in enumerate(self.grid):
            for c, value in enumerate(row):
                if value is not None:
                    state = (r, c)
                    valid_states.append(state)
                    reward[state] = float(value)

        action_list: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        transitions: Dict[Tuple[int, int], Dict[Tuple[int, int], Transition]] = {}

        state_set = set(valid_states)
        terminal_set = set(self.terminals)

        for state in valid_states:
            if state in terminal_set:
                continue
            transitions[state] = {}
            for action in action_list:
                outcomes = self._stochastic_transitions(state, action, state_set)
                transitions[state][action] = outcomes

        super().__init__(
            init=self.init,
            actlist=action_list,
            terminals=terminal_set,
            transitions=transitions,
            reward=reward,
            gamma=self.gamma,
        )
        self._action_list = tuple(action_list)

    # ------------------------------------------------------------------
    # Helpers specific to the grid domain

    def _stochastic_transitions(
        self,
        state: Tuple[int, int],
        action: Tuple[int, int],
        valid_states: set[Tuple[int, int]],
    ) -> Transition:
        outcome_weights: Dict[Tuple[int, int], float] = {}

        def add_outcome(next_state: Tuple[int, int], probability: float) -> None:
            outcome_weights[next_state] = outcome_weights.get(next_state, 0.0) + probability

        add_outcome(self._move(state, action, valid_states), 0.8)
        for perp in self._perpendicular_actions(action):
            add_outcome(self._move(state, perp, valid_states), 0.1)

        return [(s, p) for s, p in outcome_weights.items()]

    def _move(
        self,
        state: Tuple[int, int],
        action: Tuple[int, int],
        valid_states: set[Tuple[int, int]],
    ) -> Tuple[int, int]:
        r, c = state
        dr, dc = action
        candidate = (r + dr, c + dc)
        return candidate if candidate in valid_states else state

    def _perpendicular_actions(self, action: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        dr, dc = action
        if dr == 0:
            return (1, 0), (-1, 0)
        return (0, 1), (0, -1)

    # ------------------------------------------------------------------
    # Overrides for convenience

    def actions(self, state: Tuple[int, int]) -> List[Optional[Tuple[int, int]]]:
        if self.is_terminal(state):
            return [None]
        return list(self._action_list)

    def R(self, state: Tuple[int, int]) -> float:  # type: ignore[override]
        return super().R(state)

    def to_grid(self, mapping: Mapping[Tuple[int, int], Any]) -> List[List[Optional[Any]]]:
        rows = len(self.grid)
        cols = len(self.grid[0]) if rows else 0
        result: List[List[Optional[Any]]] = [[None for _ in range(cols)] for _ in range(rows)]
        for (r, c), value in mapping.items():
            if 0 <= r < rows and 0 <= c < cols and self.grid[r][c] is not None:
                result[r][c] = value
        return result
