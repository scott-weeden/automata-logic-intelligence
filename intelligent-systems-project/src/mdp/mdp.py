"""
Markov Decision Process (MDP) Implementation

Defines MDP framework and example problems.
Based on CS 5368 Week 6-7 material on sequential decision making.

MDP Components:
- States S: Set of possible states
- Actions A(s): Actions available in state s  
- Transition Model P(s'|s,a): Probability of reaching s' from s via a
- Reward Function R(s): Immediate reward for being in state s
- Discount Factor γ: Preference for immediate vs future rewards
"""

import random
from collections import defaultdict

class MDP:
    """
    Abstract base class for Markov Decision Processes.
    Implements the mathematical framework for sequential decision problems.
    """
    
    def __init__(self, init, actlist, terminals, transitions=None, reward=None, gamma=0.9):
        """
        Initialize MDP.
        
        Args:
            init: Initial state
            actlist: List of actions or dict mapping states to action lists
            terminals: Set of terminal states
            transitions: Transition probabilities P(s'|s,a)
            reward: Reward function R(s) or dict mapping states to rewards
            gamma: Discount factor [0,1]
        """
        self.init = init
        self.actlist = actlist
        self.terminals = set(terminals) if terminals else set()
        self.transitions = transitions or {}
        self.reward = reward or {}
        self.gamma = gamma
        self.states = set()
        
        # Collect all states from transitions
        if transitions:
            for (s, a) in transitions:
                self.states.add(s)
                for (prob, s_prime) in transitions[(s, a)]:
                    self.states.add(s_prime)
    
    def R(self, state):
        """Return reward for being in state."""
        if callable(self.reward):
            return self.reward(state)
        return self.reward.get(state, 0)
    
    def T(self, state, action):
        """
        Return list of (probability, next_state) pairs for transition model.
        P(s'|s,a) for all possible next states s'.
        """
        return self.transitions.get((state, action), [])
    
    def actions(self, state):
        """Return list of actions possible in given state."""
        if state in self.terminals:
            return []
        elif callable(self.actlist):
            return self.actlist(state)
        elif isinstance(self.actlist, dict):
            return self.actlist.get(state, [])
        else:
            return self.actlist

class GridMDP(MDP):
    """
    Grid-based MDP for navigation problems.
    Classic example: robot navigation with stochastic actions.
    """
    
    def __init__(self, grid, terminals, init=(0, 0), gamma=0.9):
        """
        Initialize grid MDP.
        
        Args:
            grid: 2D array where None=obstacle, numbers=rewards
            terminals: Set of terminal state positions
            init: Initial position (row, col)
            gamma: Discount factor
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        
        # Create states and rewards
        states = []
        reward = {}
        for i in range(self.rows):
            for j in range(self.cols):
                if grid[i][j] is not None:
                    states.append((i, j))
                    reward[(i, j)] = grid[i][j]
        
        # Actions: up, down, left, right
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        # Build transition model with stochastic actions
        transitions = {}
        for s in states:
            if s not in terminals:
                for a in actions:
                    transitions[(s, a)] = self._get_transitions(s, a, states)
        
        super().__init__(init, actions, terminals, transitions, reward, gamma)
        self.states = set(states)
    
    def _get_transitions(self, state, action, valid_states):
        """
        Get transition probabilities for stochastic grid world.
        80% chance of intended action, 10% each for perpendicular actions.
        """
        transitions = []
        
        # Intended action (80% probability)
        next_state = self._move(state, action, valid_states)
        transitions.append((0.8, next_state))
        
        # Perpendicular actions (10% each)
        perp_actions = self._perpendicular_actions(action)
        for perp_action in perp_actions:
            next_state = self._move(state, perp_action, valid_states)
            transitions.append((0.1, next_state))
        
        return transitions
    
    def _move(self, state, action, valid_states):
        """Apply action to state, staying in place if move is invalid."""
        row, col = state
        dr, dc = action
        new_state = (row + dr, col + dc)
        
        if new_state in valid_states:
            return new_state
        else:
            return state  # Stay in place if move is invalid
    
    def _perpendicular_actions(self, action):
        """Return two actions perpendicular to given action."""
        dr, dc = action
        if dr == 0:  # Horizontal action
            return [(1, 0), (-1, 0)]  # Vertical actions
        else:  # Vertical action  
            return [(0, 1), (0, -1)]  # Horizontal actions
    
    def to_grid(self, mapping):
        """Convert state->value mapping to 2D grid for visualization."""
        result = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) in mapping:
                    result[i][j] = mapping[(i, j)]
        return result
