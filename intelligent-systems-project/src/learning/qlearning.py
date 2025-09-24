"""
Q-Learning Algorithm

Model-free reinforcement learning that learns optimal Q-values.
Based on CS 5368 Week 7-8 material on reinforcement learning.

Q-Learning update rule:
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

Converges to optimal Q* values under appropriate conditions.
"""

import random
import math
from collections import defaultdict

class QLearningAgent:
    """
    Q-Learning agent that learns optimal action-value function.
    Uses epsilon-greedy exploration strategy.
    """
    
    def __init__(self, actions, alpha=0.1, epsilon=0.1, gamma=0.9):
        """
        Initialize Q-Learning agent.
        
        Args:
            actions: List of possible actions
            alpha: Learning rate [0,1]
            epsilon: Exploration rate [0,1] 
            gamma: Discount factor [0,1]
        """
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_values = defaultdict(float)  # Q(s,a) values
        self.last_state = None
        self.last_action = None
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        return self.q_values[(state, action)]
    
    def get_max_q_value(self, state):
        """Get maximum Q-value over all actions in state."""
        if not self.actions:
            return 0.0
        
        max_q = -math.inf
        for action in self.actions:
            q_val = self.get_q_value(state, action)
            max_q = max(max_q, q_val)
        
        return max_q if max_q != -math.inf else 0.0
    
    def get_best_action(self, state):
        """Get action with highest Q-value in state."""
        if not self.actions:
            return None
        
        best_actions = []
        max_q = self.get_max_q_value(state)
        
        for action in self.actions:
            if self.get_q_value(state, action) == max_q:
                best_actions.append(action)
        
        return random.choice(best_actions)
    
    def get_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        Explore with probability epsilon, exploit otherwise.
        """
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(self.actions)
        else:
            # Exploit: choose best known action
            return self.get_best_action(state)
    
    def update(self, state, action, next_state, reward):
        """
        Update Q-value using Q-learning rule.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            reward: Reward received
        """
        current_q = self.get_q_value(state, action)
        max_next_q = self.get_max_q_value(next_state)
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_values[(state, action)] = new_q
    
    def observe_transition(self, state, action, next_state, reward):
        """
        Observe a transition and update Q-values.
        Used for batch learning from experience.
        """
        self.update(state, action, next_state, reward)
    
    def start_episode(self, state):
        """Start new episode in given state."""
        self.last_state = state
        self.last_action = self.get_action(state)
        return self.last_action
    
    def step(self, reward, next_state):
        """
        Take step in environment and update Q-values.
        
        Args:
            reward: Reward from last action
            next_state: New state after last action
        
        Returns:
            Next action to take
        """
        if self.last_state is not None and self.last_action is not None:
            self.update(self.last_state, self.last_action, next_state, reward)
        
        self.last_state = next_state
        self.last_action = self.get_action(next_state)
        return self.last_action
    
    def end_episode(self, reward):
        """End episode and perform final update."""
        if self.last_state is not None and self.last_action is not None:
            # Terminal state has Q-value of 0
            self.update(self.last_state, self.last_action, None, reward)
        
        self.last_state = None
        self.last_action = None

def q_learning_episode(agent, environment, max_steps=1000):
    """
    Run single Q-learning episode.
    
    Args:
        agent: QLearningAgent
        environment: Environment with step() and reset() methods
        max_steps: Maximum steps per episode
    
    Returns:
        Total reward accumulated in episode
    """
    state = environment.reset()
    action = agent.start_episode(state)
    total_reward = 0
    
    for step in range(max_steps):
        next_state, reward, done = environment.step(action)
        total_reward += reward
        
        if done:
            agent.end_episode(reward)
            break
        else:
            action = agent.step(reward, next_state)
    
    return total_reward

def train_q_learning(agent, environment, num_episodes=1000, verbose=False):
    """
    Train Q-learning agent over multiple episodes.
    
    Args:
        agent: QLearningAgent to train
        environment: Environment to learn in
        num_episodes: Number of training episodes
        verbose: Print progress information
    
    Returns:
        List of rewards per episode
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        reward = q_learning_episode(agent, environment)
        episode_rewards.append(reward)
        
        if verbose and episode % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return episode_rewards
