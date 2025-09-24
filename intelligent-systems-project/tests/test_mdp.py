"""
Comprehensive Test Suite for Markov Decision Process Algorithms

Tests MDP formulation, value iteration, policy iteration, and various
MDP problem types. Validates convergence, optimality, and correctness.

Based on CS 5368 Week 6-7 material on sequential decision making.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mdp import (
    MDP, GridMDP, MarkovDecisionProcess,
    ValueIterationAgent, PolicyIterationAgent,
    value_iteration, extract_policy, policy_evaluation
)
import math

class SimpleMDP(MarkovDecisionProcess):
    """
    Simple 3-state MDP for testing basic functionality.
    
    States: 'A', 'B', 'C'
    Actions: 'left', 'right'
    Transitions: Deterministic with known rewards
    
    This creates a simple chain: A <-> B <-> C
    where C is terminal with reward +10, A has reward -1
    """
    
    def __init__(self):
        """Initialize simple 3-state MDP."""
        self.states = ['A', 'B', 'C']
        self.start = 'A'
        self.terminals = {'C'}
        
        # Rewards: A=-1 (living penalty), B=0, C=+10 (goal)
        self.rewards = {'A': -1, 'B': 0, 'C': 10}
        
        # Transitions: A<->B<->C chain
        self.transition_probs = {
            ('A', 'right'): [('B', 1.0)],
            ('A', 'left'): [('A', 1.0)],  # Stay in A
            ('B', 'right'): [('C', 1.0)],
            ('B', 'left'): [('A', 1.0)],
            ('C', 'right'): [('C', 1.0)],  # Terminal state
            ('C', 'left'): [('C', 1.0)]
        }
    
    def get_states(self):
        """Return all states."""
        return self.states
    
    def get_start_state(self):
        """Return start state."""
        return self.start
    
    def get_possible_actions(self, state):
        """Return possible actions from state."""
        if state in self.terminals:
            return []
        return ['left', 'right']
    
    def get_transition_states_and_probs(self, state, action):
        """Return list of (next_state, probability) pairs."""
        return self.transition_probs.get((state, action), [])
    
    def get_reward(self, state, action, next_state):
        """Return reward for transition."""
        return self.rewards.get(next_state, 0)
    
    def is_terminal(self, state):
        """Return True if state is terminal."""
        return state in self.terminals

class TestMDPInterface(unittest.TestCase):
    """Test MDP interface and basic problem formulation."""
    
    def setUp(self):
        """Set up test MDP instances."""
        self.simple_mdp = SimpleMDP()
        
        # Create a small grid MDP
        self.grid = [
            [0, -1, 10],  # 0=neutral, -1=penalty, 10=reward
            [-1, -1, -1]
        ]
        self.grid_mdp = GridMDP(
            grid=self.grid,
            terminals={(0, 2)},  # Goal at top-right
            init=(0, 0),
            gamma=0.9
        )
    
    def test_mdp_states_interface(self):
        """Test MDP states interface."""
        states = self.simple_mdp.get_states()
        
        self.assertEqual(len(states), 3)
        self.assertIn('A', states)
        self.assertIn('B', states)
        self.assertIn('C', states)
    
    def test_mdp_actions_interface(self):
        """Test MDP actions interface."""
        # Non-terminal state should have actions
        actions_a = self.simple_mdp.get_possible_actions('A')
        self.assertEqual(len(actions_a), 2)
        self.assertIn('left', actions_a)
        self.assertIn('right', actions_a)
        
        # Terminal state should have no actions
        actions_c = self.simple_mdp.get_possible_actions('C')
        self.assertEqual(len(actions_c), 0)
    
    def test_mdp_transitions_interface(self):
        """Test MDP transition model interface."""
        # Test deterministic transition
        transitions = self.simple_mdp.get_transition_states_and_probs('A', 'right')
        
        self.assertEqual(len(transitions), 1)
        next_state, prob = transitions[0]
        self.assertEqual(next_state, 'B')
        self.assertEqual(prob, 1.0)
    
    def test_mdp_rewards_interface(self):
        """Test MDP reward function interface."""
        # Test reward calculation
        reward = self.simple_mdp.get_reward('A', 'right', 'B')
        self.assertEqual(reward, 0)  # Reward for reaching B
        
        reward = self.simple_mdp.get_reward('B', 'right', 'C')
        self.assertEqual(reward, 10)  # Reward for reaching goal C
    
    def test_mdp_terminal_detection(self):
        """Test terminal state detection."""
        self.assertFalse(self.simple_mdp.is_terminal('A'))
        self.assertFalse(self.simple_mdp.is_terminal('B'))
        self.assertTrue(self.simple_mdp.is_terminal('C'))

class TestGridMDP(unittest.TestCase):
    """Test GridMDP implementation for spatial navigation problems."""
    
    def setUp(self):
        """Set up grid MDP test cases."""
        # Simple 3x3 grid with goal and penalty
        self.grid = [
            [0, 0, 10],   # Goal at (0,2)
            [0, -5, 0],   # Penalty at (1,1)
            [-1, 0, 0]    # Small penalty at (2,0)
        ]
        
        self.grid_mdp = GridMDP(
            grid=self.grid,
            terminals={(0, 2)},
            init=(2, 0),
            gamma=0.9
        )
    
    def test_grid_mdp_initialization(self):
        """Test GridMDP initialization and setup."""
        self.assertEqual(self.grid_mdp.rows, 3)
        self.assertEqual(self.grid_mdp.cols, 3)
        self.assertEqual(self.grid_mdp.init, (2, 0))
        self.assertIn((0, 2), self.grid_mdp.terminals)
    
    def test_grid_mdp_states_generation(self):
        """Test that GridMDP generates correct state space."""
        states = self.grid_mdp.get_states()
        
        # Should have 9 states (3x3 grid)
        self.assertEqual(len(states), 9)
        
        # Check specific states exist
        self.assertIn((0, 0), states)
        self.assertIn((1, 1), states)
        self.assertIn((2, 2), states)
    
    def test_grid_mdp_actions(self):
        """Test GridMDP action generation."""
        # Non-terminal state should have 4 directional actions
        actions = self.grid_mdp.get_possible_actions((1, 1))
        
        self.assertEqual(len(actions), 4)
        # Actions are direction tuples: (dr, dc)
        expected_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for action in expected_actions:
            self.assertIn(action, actions)
        
        # Terminal state should have no actions
        terminal_actions = self.grid_mdp.get_possible_actions((0, 2))
        self.assertEqual(len(terminal_actions), 0)
    
    def test_grid_mdp_stochastic_transitions(self):
        """Test GridMDP stochastic transition model."""
        # Test transition from (1,1) with 'right' action
        transitions = self.grid_mdp.get_transition_states_and_probs((1, 1), (0, 1))
        
        # Should have 3 possible outcomes due to noise
        self.assertEqual(len(transitions), 3)
        
        # Check probabilities sum to 1
        total_prob = sum(prob for state, prob in transitions)
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
        # Main action should have highest probability (0.8)
        probs = [prob for state, prob in transitions]
        self.assertIn(0.8, probs)  # Intended direction
        self.assertEqual(probs.count(0.1), 2)  # Two perpendicular directions
    
    def test_grid_mdp_boundary_handling(self):
        """Test GridMDP handles grid boundaries correctly."""
        # Test transition from corner that would go out of bounds
        transitions = self.grid_mdp.get_transition_states_and_probs((0, 0), (-1, 0))  # Try to go up
        
        # Should stay in place when hitting boundary
        for next_state, prob in transitions:
            # All resulting states should be valid grid positions
            row, col = next_state
            self.assertGreaterEqual(row, 0)
            self.assertLess(row, 3)
            self.assertGreaterEqual(col, 0)
            self.assertLess(col, 3)
    
    def test_grid_mdp_rewards(self):
        """Test GridMDP reward calculation."""
        # Test reward for reaching goal
        reward = self.grid_mdp.get_reward((0, 1), (0, 1), (0, 2))
        self.assertEqual(reward, 10)
        
        # Test penalty for reaching bad state
        reward = self.grid_mdp.get_reward((1, 0), (0, 1), (1, 1))
        self.assertEqual(reward, -5)
    
    def test_grid_mdp_visualization(self):
        """Test GridMDP to_grid visualization method."""
        # Create a mapping of states to values
        values = {(0, 0): 1.0, (1, 1): -2.0, (2, 2): 3.0}
        
        grid_viz = self.grid_mdp.to_grid(values)
        
        # Should be 3x3 grid
        self.assertEqual(len(grid_viz), 3)
        self.assertEqual(len(grid_viz[0]), 3)
        
        # Check specific values
        self.assertEqual(grid_viz[0][0], 1.0)
        self.assertEqual(grid_viz[1][1], -2.0)
        self.assertEqual(grid_viz[2][2], 3.0)
        self.assertIsNone(grid_viz[0][1])  # Unmapped state

class TestValueIteration(unittest.TestCase):
    """Test Value Iteration algorithm for solving MDPs."""
    
    def setUp(self):
        """Set up MDPs for value iteration testing."""
        self.simple_mdp = SimpleMDP()
        
        # Create deterministic grid for predictable results
        self.grid = [
            [0, 0, 10],
            [0, -1, 0]
        ]
        self.grid_mdp = GridMDP(
            grid=self.grid,
            terminals={(0, 2)},
            init=(0, 0),
            gamma=0.9
        )
    
    def test_value_iteration_convergence(self):
        """Test that value iteration converges to stable values."""
        # Run value iteration
        values = value_iteration(self.simple_mdp, epsilon=0.001)
        
        # Should return values for all states
        self.assertIn('A', values)
        self.assertIn('B', values)
        self.assertIn('C', values)
        
        # Terminal state should have value equal to its reward
        self.assertEqual(values['C'], 10)
        
        # Values should reflect optimal policy
        # B should have higher value than A (closer to goal)
        self.assertGreater(values['B'], values['A'])
    
    def test_value_iteration_agent(self):
        """Test ValueIterationAgent wrapper class."""
        agent = ValueIterationAgent(self.simple_mdp, discount=0.9, iterations=100)
        
        # Test value retrieval
        value_a = agent.get_value('A')
        value_b = agent.get_value('B')
        value_c = agent.get_value('C')
        
        # Terminal state value
        self.assertEqual(value_c, 0)  # Terminal states have 0 value in agent
        
        # Ordering should reflect distance to goal
        self.assertGreater(value_b, value_a)
    
    def test_value_iteration_policy_extraction(self):
        """Test policy extraction from value iteration."""
        values = value_iteration(self.simple_mdp, epsilon=0.001)
        policy = extract_policy(self.simple_mdp, values)
        
        # Should have policy for non-terminal states
        self.assertIn('A', policy)
        self.assertIn('B', policy)
        
        # Policy should be optimal: A->right (toward B), B->right (toward C)
        self.assertEqual(policy['A'], 'right')
        self.assertEqual(policy['B'], 'right')
        
        # Terminal state should have no policy
        self.assertIsNone(policy['C'])
    
    def test_value_iteration_discount_factor_effect(self):
        """Test effect of different discount factors on values."""
        # High discount (patient agent)
        values_high = value_iteration(
            MDP(init='A', actlist=['left', 'right'], terminals={'C'}, 
                transitions=self.simple_mdp.transition_probs,
                reward=self.simple_mdp.rewards, gamma=0.95)
        )
        
        # Low discount (impatient agent)
        values_low = value_iteration(
            MDP(init='A', actlist=['left', 'right'], terminals={'C'},
                transitions=self.simple_mdp.transition_probs, 
                reward=self.simple_mdp.rewards, gamma=0.5)
        )
        
        # High discount should lead to higher values (more future-oriented)
        self.assertGreater(values_high['A'], values_low['A'])
    
    def test_value_iteration_grid_world(self):
        """Test value iteration on grid world problem."""
        agent = ValueIterationAgent(self.grid_mdp, discount=0.9, iterations=50)
        
        # Goal state should have highest value in neighborhood
        goal_value = agent.get_value((0, 2))
        neighbor_value = agent.get_value((0, 1))
        
        # Values should be reasonable (non-infinite, non-NaN)
        self.assertFalse(math.isnan(goal_value))
        self.assertFalse(math.isinf(goal_value))
        self.assertFalse(math.isnan(neighbor_value))
        self.assertFalse(math.isinf(neighbor_value))
    
    def test_value_iteration_policy_optimality(self):
        """Test that value iteration produces optimal policy."""
        agent = ValueIterationAgent(self.simple_mdp, discount=0.9, iterations=100)
        
        # Get policy for each state
        policy_a = agent.get_policy('A')
        policy_b = agent.get_policy('B')
        
        # Optimal policy should move toward goal
        self.assertEqual(policy_a, 'right')  # A should go to B
        self.assertEqual(policy_b, 'right')  # B should go to C

class TestPolicyIteration(unittest.TestCase):
    """Test Policy Iteration algorithm for solving MDPs."""
    
    def setUp(self):
        """Set up MDPs for policy iteration testing."""
        self.simple_mdp = SimpleMDP()
        
        # Small grid for testing
        self.grid = [[0, 10], [-1, 0]]
        self.grid_mdp = GridMDP(
            grid=self.grid,
            terminals={(0, 1)},
            init=(1, 0),
            gamma=0.9
        )
    
    def test_policy_iteration_agent_creation(self):
        """Test PolicyIterationAgent initialization."""
        agent = PolicyIterationAgent(self.simple_mdp, discount=0.9)
        
        # Should have computed values and policy
        self.assertIsNotNone(agent.values)
        self.assertIsNotNone(agent.policy)
        
        # Should have policy for non-terminal states
        self.assertIn('A', agent.policy)
        self.assertIn('B', agent.policy)
    
    def test_policy_iteration_convergence(self):
        """Test that policy iteration converges to optimal policy."""
        agent = PolicyIterationAgent(self.simple_mdp, discount=0.9)
        
        # Get final policy
        policy_a = agent.get_policy('A')
        policy_b = agent.get_policy('B')
        
        # Should converge to optimal policy
        self.assertEqual(policy_a, 'right')
        self.assertEqual(policy_b, 'right')
    
    def test_policy_iteration_vs_value_iteration(self):
        """Test policy iteration gives same result as value iteration."""
        pi_agent = PolicyIterationAgent(self.simple_mdp, discount=0.9)
        vi_agent = ValueIterationAgent(self.simple_mdp, discount=0.9, iterations=100)
        
        # Policies should be the same
        for state in ['A', 'B']:
            pi_policy = pi_agent.get_policy(state)
            vi_policy = vi_agent.get_policy(state)
            self.assertEqual(pi_policy, vi_policy)
    
    def test_policy_evaluation_function(self):
        """Test standalone policy evaluation function."""
        # Create a fixed policy: always go right
        policy = {'A': 'right', 'B': 'right', 'C': None}
        
        # Evaluate this policy
        values = policy_evaluation(self.simple_mdp, policy, max_iterations=50)
        
        # Should compute reasonable values
        self.assertIn('A', values)
        self.assertIn('B', values)
        self.assertIn('C', values)
        
        # Values should reflect policy: B closer to goal than A
        self.assertGreater(values['B'], values['A'])
    
    def test_policy_iteration_grid_world(self):
        """Test policy iteration on grid world."""
        agent = PolicyIterationAgent(self.grid_mdp, discount=0.9)
        
        # Should produce reasonable policy
        start_policy = agent.get_policy((1, 0))
        
        # Policy should be valid action
        valid_actions = self.grid_mdp.get_possible_actions((1, 0))
        self.assertIn(start_policy, valid_actions)

class TestMDPUtilityFunctions(unittest.TestCase):
    """Test utility functions for MDP analysis."""
    
    def setUp(self):
        """Set up test MDP."""
        self.simple_mdp = SimpleMDP()
    
    def test_policy_evaluation_convergence(self):
        """Test policy evaluation converges to correct values."""
        # Fixed policy: always go right
        policy = {'A': 'right', 'B': 'right', 'C': None}
        
        # Evaluate with different iteration counts
        values_10 = policy_evaluation(self.simple_mdp, policy, max_iterations=10)
        values_50 = policy_evaluation(self.simple_mdp, policy, max_iterations=50)
        
        # More iterations should give more accurate values
        # (values should converge, so difference should be small)
        diff_a = abs(values_50['A'] - values_10['A'])
        diff_b = abs(values_50['B'] - values_10['B'])
        
        # Differences should be small (convergence)
        self.assertLess(diff_a, 1.0)
        self.assertLess(diff_b, 1.0)
    
    def test_extract_policy_optimality(self):
        """Test policy extraction produces optimal actions."""
        # Run value iteration to get optimal values
        values = value_iteration(self.simple_mdp, epsilon=0.001)
        
        # Extract policy
        policy = extract_policy(self.simple_mdp, values)
        
        # Policy should be greedy with respect to values
        # This means each action should maximize expected value
        
        for state in ['A', 'B']:
            chosen_action = policy[state]
            
            # Calculate Q-value for chosen action
            transitions = self.simple_mdp.get_transition_states_and_probs(state, chosen_action)
            q_chosen = sum(prob * (self.simple_mdp.get_reward(state, chosen_action, next_state) + 
                                 0.9 * values[next_state])
                          for next_state, prob in transitions)
            
            # Check that no other action has higher Q-value
            for action in self.simple_mdp.get_possible_actions(state):
                if action != chosen_action:
                    transitions_alt = self.simple_mdp.get_transition_states_and_probs(state, action)
                    q_alt = sum(prob * (self.simple_mdp.get_reward(state, action, next_state) + 
                                      0.9 * values[next_state])
                               for next_state, prob in transitions_alt)
                    
                    # Chosen action should be at least as good
                    self.assertGreaterEqual(q_chosen, q_alt - 0.001)  # Small tolerance

class TestMDPEdgeCases(unittest.TestCase):
    """Test MDP algorithms on edge cases and special scenarios."""
    
    def test_single_state_mdp(self):
        """Test MDP with only one state."""
        class SingleStateMDP(MarkovDecisionProcess):
            def get_states(self):
                return ['only']
            
            def get_start_state(self):
                return 'only'
            
            def get_possible_actions(self, state):
                return []  # No actions available
            
            def get_transition_states_and_probs(self, state, action):
                return []
            
            def get_reward(self, state, action, next_state):
                return 5  # Fixed reward
            
            def is_terminal(self, state):
                return True  # Always terminal
        
        single_mdp = SingleStateMDP()
        
        # Value iteration should handle this gracefully
        values = value_iteration(single_mdp, epsilon=0.001)
        self.assertIn('only', values)
        
        # Policy extraction should handle no actions
        policy = extract_policy(single_mdp, values)
        self.assertIsNone(policy['only'])
    
    def test_disconnected_mdp(self):
        """Test MDP with disconnected state components."""
        class DisconnectedMDP(MarkovDecisionProcess):
            def get_states(self):
                return ['A', 'B', 'C', 'D']  # A-B disconnected from C-D
            
            def get_start_state(self):
                return 'A'
            
            def get_possible_actions(self, state):
                if state in ['A', 'B']:
                    return ['move']
                elif state in ['C', 'D']:
                    return ['move']
                return []
            
            def get_transition_states_and_probs(self, state, action):
                transitions = {
                    ('A', 'move'): [('B', 1.0)],
                    ('B', 'move'): [('A', 1.0)],
                    ('C', 'move'): [('D', 1.0)],
                    ('D', 'move'): [('C', 1.0)]
                }
                return transitions.get((state, action), [])
            
            def get_reward(self, state, action, next_state):
                return 0
            
            def is_terminal(self, state):
                return False
        
        disconnected_mdp = DisconnectedMDP()
        
        # Should still compute values (may be 0 due to no terminal rewards)
        values = value_iteration(disconnected_mdp, epsilon=0.001)
        
        self.assertEqual(len(values), 4)
        for state in ['A', 'B', 'C', 'D']:
            self.assertIn(state, values)
    
    def test_zero_discount_mdp(self):
        """Test MDP with zero discount factor (myopic agent)."""
        # Create MDP with gamma=0 (only immediate rewards matter)
        zero_discount_mdp = MDP(
            init='A',
            actlist=['left', 'right'],
            terminals={'C'},
            transitions=self.simple_mdp.transition_probs,
            reward=self.simple_mdp.rewards,
            gamma=0.0
        )
        
        values = value_iteration(zero_discount_mdp, epsilon=0.001)
        
        # With gamma=0, only immediate rewards matter
        # Values should reflect immediate reward structure
        self.assertIn('A', values)
        self.assertIn('B', values)
        self.assertIn('C', values)

if __name__ == '__main__':
    # Run all MDP algorithm tests
    unittest.main(verbosity=2)
