"""
Value Iteration Algorithm for MDPs

Implements value iteration to find optimal policies.
Based on CS 5368 Week 6-7 Bellman equations and dynamic programming.

Value iteration repeatedly applies Bellman update:
U(s) = R(s) + γ max_a Σ P(s'|s,a) U(s')

Converges to optimal utilities U* and optimal policy π*.
"""

import math
from collections import defaultdict

def value_iteration(mdp, epsilon=0.001):
    """
    Solve MDP using value iteration algorithm.
    
    Args:
        mdp: MDP object with states, actions, transitions, rewards
        epsilon: Convergence threshold for utility changes
    
    Returns:
        Dictionary mapping states to optimal utilities
    """
    # Initialize utilities arbitrarily (often to 0)
    U = defaultdict(float)
    U_prime = defaultdict(float)
    
    iteration = 0
    while True:
        # Copy current utilities
        U_prime = U.copy()
        delta = 0
        
        # Update utility for each state
        for s in mdp.states:
            if s in mdp.terminals:
                U[s] = mdp.R(s)
            else:
                # Bellman update: U(s) = R(s) + γ max_a Σ P(s'|s,a) U(s')
                max_utility = -math.inf
                
                for a in mdp.actions(s):
                    utility_sum = 0
                    for prob, s_prime in mdp.T(s, a):
                        utility_sum += prob * U_prime[s_prime]
                    
                    action_utility = mdp.R(s) + mdp.gamma * utility_sum
                    max_utility = max(max_utility, action_utility)
                
                U[s] = max_utility
            
            # Track maximum change in utilities
            delta = max(delta, abs(U[s] - U_prime[s]))
        
        iteration += 1
        
        # Check convergence
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break
    
    print(f"Value iteration converged after {iteration} iterations")
    return dict(U)

def extract_policy(mdp, utilities):
    """
    Extract optimal policy from utilities using one-step lookahead.
    
    Args:
        mdp: MDP object
        utilities: State utilities from value iteration
    
    Returns:
        Dictionary mapping states to optimal actions
    """
    policy = {}
    
    for s in mdp.states:
        if s in mdp.terminals:
            policy[s] = None
        else:
            best_action = None
            best_utility = -math.inf
            
            for a in mdp.actions(s):
                utility_sum = 0
                for prob, s_prime in mdp.T(s, a):
                    utility_sum += prob * utilities[s_prime]
                
                action_utility = mdp.R(s) + mdp.gamma * utility_sum
                
                if action_utility > best_utility:
                    best_utility = action_utility
                    best_action = a
            
            policy[s] = best_action
    
    return policy

def policy_evaluation(mdp, policy, utilities=None, max_iterations=100):
    """
    Evaluate given policy to compute state utilities.
    
    Args:
        mdp: MDP object
        policy: Policy mapping states to actions
        utilities: Initial utilities (default: all zeros)
        max_iterations: Maximum number of iterations
    
    Returns:
        Dictionary of state utilities under given policy
    """
    if utilities is None:
        utilities = defaultdict(float)
    
    for iteration in range(max_iterations):
        U_prime = utilities.copy()
        
        for s in mdp.states:
            if s in mdp.terminals:
                utilities[s] = mdp.R(s)
            else:
                action = policy[s]
                if action is not None:
                    utility_sum = 0
                    for prob, s_prime in mdp.T(s, action):
                        utility_sum += prob * U_prime[s_prime]
                    
                    utilities[s] = mdp.R(s) + mdp.gamma * utility_sum
    
    return dict(utilities)

def value_iteration_with_policy(mdp, epsilon=0.001):
    """
    Run value iteration and return both utilities and optimal policy.
    
    Args:
        mdp: MDP object
        epsilon: Convergence threshold
    
    Returns:
        Tuple of (utilities, policy) dictionaries
    """
    utilities = value_iteration(mdp, epsilon)
    policy = extract_policy(mdp, utilities)
    return utilities, policy
