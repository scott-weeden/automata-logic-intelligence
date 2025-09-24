"""
Learning Module

Implements reinforcement learning algorithms.
Based on CS 5368 Week 7-8 material on learning from interaction.
"""

from .qlearning import QLearningAgent, q_learning_episode, train_q_learning

__all__ = [
    'QLearningAgent', 'q_learning_episode', 'train_q_learning'
]
