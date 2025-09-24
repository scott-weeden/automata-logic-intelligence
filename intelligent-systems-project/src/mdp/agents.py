"""
MDP Agent Implementations

Agents that solve MDPs using various algorithms.
Based on CS 5368 Week 6-7 material on sequential decision making.
"""

import math
import random
from collections import defaultdict
from .value_iteration import value_iteration, extract_policy

class MarkovDecisionProcess:
    """Base class for MDP problems."""
    
    def get_states(self):
        """Return list of all states."""
        raise NotImplementedError
    
    def get_start_state(self):
        """Return start state."""
        raise NotImplementedError
    
    def get_possible_actions(self, state):
        """Return list of possible actions from state."""
        raise NotImplementedError
    
    def get_transition_states_and_probs(self, state, action):
        """Return list of (next_state, probability) pairs."""
        raise NotImplementedError
    
    def get_reward(self, state, action, next_state):
        """Return reward for transition."""
        raise NotImplementedError
    
    def is_terminal(self, state):
        """Return True if state is terminal."""
        raise NotImplementedError

class ValueIterationAgent:
    """Agent that solves MDPs using value iteration."""
    
    def __init__(self, mdp, discount=0.9, iterations=100):
        """Initialize value iteration agent."""
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = defaultdict(float)
        
        # Run value iteration
        self._run_value_iteration()
    
    def _run_value_iteration(self):
        """Run value iteration algorithm."""
        for iteration in range(self.iterations):
            new_values = defaultdict(float)
            
            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    new_values[state] = 0
                else:
                    max_value = -math.inf
                    
                    for action in self.mdp.get_possible_actions(state):
                        q_value = self._compute_q_value(state, action)
                        max_value = max(max_value, q_value)
                    
                    new_values[state] = max_value if max_value != -math.inf else 0
            
            self.values = new_values
    
    def _compute_q_value(self, state, action):
        """Compute Q-value for state-action pair."""
        q_value = 0
        
        for next_state, prob in self.mdp.get_transition_states_and_probs(state, action):
            reward = self.mdp.get_reward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        
        return q_value
    
    def get_value(self, state):
        """Return value of state."""
        return self.values[state]
    
    def get_policy(self, state):
        """Return best action for state."""
        if self.mdp.is_terminal(state):
            return None
        
        best_action = None
        best_value = -math.inf
        
        for action in self.mdp.get_possible_actions(state):
            q_value = self._compute_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action

class PolicyIterationAgent:
    """Agent that solves MDPs using policy iteration."""
    
    def __init__(self, mdp, discount=0.9):
        """Initialize policy iteration agent."""
        self.mdp = mdp
        self.discount = discount
        self.values = defaultdict(float)
        self.policy = {}
        
        # Initialize random policy
        for state in self.mdp.get_states():
            if not self.mdp.is_terminal(state):
                actions = self.mdp.get_possible_actions(state)
                self.policy[state] = actions[0] if actions else None
        
        # Run policy iteration
        self._run_policy_iteration()
    
    def _run_policy_iteration(self):
        """Run policy iteration algorithm."""
        for iteration in range(100):  # Max iterations
            # Policy evaluation
            self._policy_evaluation()
            
            # Policy improvement
            policy_changed = self._policy_improvement()
            
            if not policy_changed:
                break
    
    def _policy_evaluation(self):
        """Evaluate current policy."""
        for _ in range(100):  # Max evaluation iterations
            new_values = defaultdict(float)
            
            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    new_values[state] = 0
                else:
                    action = self.policy.get(state)
                    if action is not None:
                        new_values[state] = self._compute_q_value(state, action)
            
            self.values = new_values
    
    def _policy_improvement(self):
        """Improve policy based on current values."""
        policy_changed = False
        
        for state in self.mdp.get_states():
            if not self.mdp.is_terminal(state):
                old_action = self.policy.get(state)
                
                best_action = None
                best_value = -math.inf
                
                for action in self.mdp.get_possible_actions(state):
                    q_value = self._compute_q_value(state, action)
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                
                self.policy[state] = best_action
                
                if old_action != best_action:
                    policy_changed = True
        
        return policy_changed
    
    def _compute_q_value(self, state, action):
        """Compute Q-value for state-action pair."""
        q_value = 0
        
        for next_state, prob in self.mdp.get_transition_states_and_probs(state, action):
            reward = self.mdp.get_reward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        
        return q_value
    
    def get_value(self, state):
        """Return value of state."""
        return self.values[state]
    
    def get_policy(self, state):
        """Return best action for state."""
        return self.policy.get(state)

class QLearningAgent:
    """Q-Learning agent for model-free reinforcement learning."""
    
    def __init__(self, action_fn, discount=0.9, alpha=0.5, epsilon=0.1):
        """Initialize Q-Learning agent."""
        self.action_fn = action_fn
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_values = defaultdict(float)
        self.training = True
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        return self.q_values[(state, action)]
    
    def get_max_q_value(self, state):
        """Get maximum Q-value over all actions in state."""
        actions = self.action_fn(state)
        if not actions:
            return 0.0
        
        return max(self.get_q_value(state, action) for action in actions)
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy."""
        actions = self.action_fn(state)
        if not actions:
            return None
        
        if self.training and random.random() < self.epsilon:
            return random.choice(actions)
        else:
            return self.compute_action_from_q_values(state)
    
    def compute_action_from_q_values(self, state):
        """Compute best action from Q-values."""
        actions = self.action_fn(state)
        if not actions:
            return None
        
        best_actions = []
        best_value = -math.inf
        
        for action in actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)
        
        return random.choice(best_actions)
    
    def update(self, state, action, next_state, reward):
        """Update Q-values using Q-learning rule."""
        current_q = self.get_q_value(state, action)
        max_next_q = self.get_max_q_value(next_state)
        
        new_q = current_q + self.alpha * (reward + self.discount * max_next_q - current_q)
        self.q_values[(state, action)] = new_q
    
    def start_episode(self):
        """Start new episode."""
        pass
    
    def stop_training(self):
        """Stop training (turn off exploration)."""
        self.training = False

class SARSAAgent(QLearningAgent):
    """SARSA agent for on-policy reinforcement learning."""
    
    def __init__(self, action_fn, discount=0.9, alpha=0.5, epsilon=0.1):
        """Initialize SARSA agent."""
        super().__init__(action_fn, discount, alpha, epsilon)
        self.next_action = None
    
    def update(self, state, action, next_state, reward):
        """Update Q-values using SARSA rule."""
        current_q = self.get_q_value(state, action)
        
        # Use next action's Q-value (on-policy)
        if self.next_action is not None:
            next_q = self.get_q_value(next_state, self.next_action)
        else:
            next_q = 0
        
        new_q = current_q + self.alpha * (reward + self.discount * next_q - current_q)
        self.q_values[(state, action)] = new_q
