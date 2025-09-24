"""Toy reinforcement-learning trader using tabular Q-learning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from learning import QLearningAgent

State = Tuple[bool, str]  # (holding, price_trend)
Action = str

ACTIONS: Tuple[Action, ...] = ("buy", "sell", "hold")


@dataclass
class TradingEpisodeSummary:
    episode: int
    total_reward: float
    steps: int


class TradingEnvironment:
    def __init__(self, prices: List[float]) -> None:
        self.prices = prices
        self.index = 0
        self.holding = False
        self.entry_price = 0.0

    def reset(self) -> State:
        self.index = 0
        self.holding = False
        self.entry_price = 0.0
        return self._observe_state()

    def _observe_state(self) -> State:
        if self.index >= len(self.prices) - 1:
            trend = "flat"
        else:
            trend = "up" if self.prices[self.index + 1] >= self.prices[self.index] else "down"
        return (self.holding, trend)

    def step(self, action: Action) -> Tuple[State, float, bool]:
        reward = 0.0
        if action == "buy" and not self.holding:
            self.holding = True
            self.entry_price = self.prices[self.index]
            reward -= 0.05  # transaction fee
        elif action == "sell" and self.holding:
            reward += self.prices[self.index] - self.entry_price
            self.holding = False
            self.entry_price = 0.0
        elif action == "hold":
            reward -= 0.01  # discourage inactivity

        self.index += 1
        done = self.index >= len(self.prices)
        next_state = self._observe_state()
        return next_state, reward, done


def _generate_price_series(length: int = 60, seed: int = 7) -> List[float]:
    rng = random.Random(seed)
    prices = [100.0]
    for _ in range(length - 1):
        change = rng.uniform(-1.5, 1.5)
        prices.append(max(50.0, prices[-1] + change))
    return prices


def train_trading_agent(episodes: int = 150, show_output: bool = True) -> List[TradingEpisodeSummary]:
    prices = _generate_price_series()
    env = TradingEnvironment(prices)
    agent = QLearningAgent(action_fn=lambda state: ACTIONS, discount=0.95, alpha=0.3, epsilon=0.2)

    summaries: List[TradingEpisodeSummary] = []
    epsilon_decay = 0.99

    for episode in range(1, episodes + 1):
        state = env.reset()
        agent.start_episode()
        total_reward = 0.0
        steps = 0

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, next_state, reward)
            total_reward += reward
            steps += 1
            state = next_state
            if done:
                break
        agent.episodes_completed += 1
        agent.epsilon *= epsilon_decay

        summaries.append(TradingEpisodeSummary(episode, total_reward, steps))

    if show_output:
        print("Episode | Reward  | Steps | Epsilon")
        for summary in summaries[-10:]:
            print(f"{summary.episode:7d} | {summary.total_reward:7.2f} | {summary.steps:5d} | {agent.epsilon:7.4f}")
        recent_avg = sum(s.total_reward for s in summaries[-10:]) / 10
        print(f"Average reward (last 10 episodes): {recent_avg:.2f}")

    return summaries


if __name__ == "__main__":  # pragma: no cover - manual demo
    train_trading_agent(show_output=True)
