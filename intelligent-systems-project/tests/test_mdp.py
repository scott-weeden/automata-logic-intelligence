"""
Comprehensive test suite for MDP module
Tests MDP implementations and solving algorithms
"""

import pytest
import sys
import os
from typing import List, Tuple, Dict, Set, Any
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mdp import (
    MarkovDecisionProcess, MDP, GridMDP,
    ValueIterationAgent, PolicyIterationAgent
)


# Test MDP Implementations
class SimpleMDP(MarkovDecisionProcess):
    """Simple MDP for testing"""
    
    def __init__(self):
        self.states = ['S1', 'S2', 'S3', 'Terminal']
        self.start = 'S1'
        self.terminals = {'Terminal'}
        self.actions_dict = {
            'S1': ['a', 'b'],
            'S2': ['a', 'b'],
            'S3': ['a'],
            'Terminal': []
        }
        self.transitions = {
            ('S1', 'a'): [('S2', 0.8), ('S3', 0.2)],
            ('S1', 'b'): [('S3', 1.0)],
            ('S2', 'a'): [('Terminal', 1.0)],
            ('S2', 'b'): [('S1', 0.5), ('S3', 0.5)],
            ('S3', 'a'): [('Terminal', 1.0)]
        }
        self.rewards = {
            ('S1', 'a', 'S2'): 5,
            ('S1', 'a', 'S3'): -1,
            ('S1', 'b', 'S3'): -1,
            ('S2', 'a', 'Terminal'): 10,
            ('S2', 'b', 'S1'): 0,
            ('S2', 'b', 'S3'): -1,
            ('S3', 'a', 'Terminal'): 1
        }
    
    def get_states(self):
        return self.states
    
    def get_start_state(self):
        return self.start
    
    def get_possible_actions(self, state):
        return self.actions_dict.get(state, [])
    
    def get_transition_states_and_probs(self, state, action):
        return self.transitions.get((state, action), [])
    
    def get_reward(self, state, action, next_state):
        return self.rewards.get((state, action, next_state), 0)
    
    def is_terminal(self, state):
        return state in self.terminals


class TestMarkovDecisionProcess:
    """Test MDP base class functionality"""
    
    def test_simple_mdp(self):
        """Test basic MDP operations"""
        mdp = SimpleMDP()
        
        assert mdp.get_start_state() == 'S1'
        assert mdp.get_states() == ['S1', 'S2', 'S3', 'Terminal']
        assert mdp.is_terminal('Terminal') == True
        assert mdp.is_terminal('S1') == False
    
    def test_mdp_actions(self):
        """Test action availability"""
        mdp = SimpleMDP()
        
        assert mdp.get_possible_actions('S1') == ['a', 'b']
        assert mdp.get_possible_actions('S3') == ['a']
        assert mdp.get_possible_actions('Terminal') == []
    
    def test_mdp_transitions(self):
        """Test transition probabilities"""
        mdp = SimpleMDP()
        
        transitions = mdp.get_transition_states_and_probs('S1', 'a')
        assert len(transitions) == 2
        
        # Check probabilities sum to 1
        total_prob = sum(prob for _, prob in transitions)
        assert abs(total_prob - 1.0) < 0.001
        
        # Check specific transitions
        trans_dict = dict(transitions)
        assert abs(trans_dict['S2'] - 0.8) < 0.001
        assert abs(trans_dict['S3'] - 0.2) < 0.001
    
    def test_mdp_rewards(self):
        """Test reward function"""
        mdp = SimpleMDP()
        
        assert mdp.get_reward('S1', 'a', 'S2') == 5
        assert mdp.get_reward('S2', 'a', 'Terminal') == 10
        assert mdp.get_reward('S1', 'b', 'S3') == -1
        assert mdp.get_reward('S1', 'x', 'S2') == 0  # Default reward


class TestMDPClass:
    """Test general MDP implementation"""
    
    def test_mdp_initialization(self):
        """Test MDP class initialization"""
        transitions = {
            'S1': {'a': [('S2', 1.0)], 'b': [('S3', 1.0)]},
            'S2': {'a': [('Terminal', 1.0)]},
            'S3': {'a': [('Terminal', 1.0)]}
        }
        
        def reward_fn(state):
            if state == 'Terminal':
                return 10
            return -1
        
        mdp = MDP(
            init='S1',
            actlist=['a', 'b'],
            terminals={'Terminal'},
            transitions=transitions,
            reward=reward_fn,
            gamma=0.9
        )
        
        assert mdp.init == 'S1'
        assert mdp.gamma == 0.9
        assert 'Terminal' in mdp.terminals
    
    def test_mdp_reward_function(self):
        """Test MDP reward function"""
        def reward_fn(state):
            rewards = {'S1': 0, 'S2': 5, 'Terminal': 10}
            return rewards.get(state, -1)
        
        mdp = MDP(
            init='S1',
            actlist=['a'],
            terminals={'Terminal'},
            reward=reward_fn
        )
        
        assert mdp.R('S1') == 0
        assert mdp.R('S2') == 5
        assert mdp.R('Terminal') == 10
        assert mdp.R('Unknown') == -1
    
    def test_mdp_transition_model(self):
        """Test MDP transition model"""
        transitions = {
            'S1': {'a': [('S2', 0.7), ('S3', 0.3)]},
            'S2': {'a': [('Terminal', 1.0)]}
        }
        
        mdp = MDP(
            init='S1',
            actlist=['a'],
            terminals={'Terminal'},
            transitions=transitions
        )
        
        # Test transition probabilities
        trans = mdp.T('S1', 'a')
        assert len(trans) == 2
        assert ('S2', 0.7) in trans
        assert ('S3', 0.3) in trans
    
    def test_mdp_actions(self):
        """Test action availability in MDP"""
        # Test with function for actions
        def action_fn(state):
            if state == 'S1':
                return ['a', 'b', 'c']
            return ['a']
        
        mdp = MDP(
            init='S1',
            actlist=action_fn,
            terminals={'Terminal'}
        )
        
        assert mdp.actions('S1') == ['a', 'b', 'c']
        assert mdp.actions('S2') == ['a']


class TestGridMDP:
    """Test GridMDP implementation"""
    
    def test_grid_mdp_initialization(self):
        """Test GridMDP initialization"""
        grid = [
            [0, 0, 0, 1],
            [0, None, 0, -1],
            [0, 0, 0, 0]
        ]
        
        mdp = GridMDP(grid, terminals={(0, 3), (1, 3)}, init=(0, 0), gamma=0.9)
        
        assert mdp.init == (0, 0)
        assert (0, 3) in mdp.terminals
        assert (1, 3) in mdp.terminals
        assert mdp.gamma == 0.9
    
    def test_grid_mdp_rewards(self):
        """Test reward structure in GridMDP"""
        grid = [
            [-0.04, -0.04, -0.04, 1],
            [-0.04, None, -0.04, -1],
            [-0.04, -0.04, -0.04, -0.04]
        ]
        
        mdp = GridMDP(grid, terminals={(0, 3), (1, 3)})
        
        # Check rewards for different positions
        assert mdp.R((0, 0)) == -0.04
        assert mdp.R((0, 3)) == 1
        assert mdp.R((1, 3)) == -1
    
    def test_grid_mdp_actions(self):
        """Test available actions in GridMDP"""
        grid = [[0, 0], [0, 0]]
        mdp = GridMDP(grid, terminals={(1, 1)})
        
        # Non-terminal state should have 4 actions
        assert len(mdp.actions((0, 0))) == 4
        assert set(mdp.actions((0, 0))) == {(0, 1), (1, 0), (0, -1), (-1, 0)}
        
        # Terminal state should have no actions
        assert mdp.actions((1, 1)) == [None]
    
    def test_grid_mdp_stochastic_transitions(self):
        """Test stochastic transitions in GridMDP"""
        grid = [[0, 0, 0], [0, 0, 0]]
        mdp = GridMDP(grid, terminals={(1, 2)})
        
        # Action: move right from (0, 0)
        transitions = mdp.T((0, 0), (0, 1))  # Right
        
        # Should have 3 outcomes: intended + 2 perpendicular
        assert len(transitions) == 3
        
        # Check probabilities (80% intended, 10% each perpendicular)
        trans_dict = dict(transitions)
        
        # Intended direction
        assert abs(trans_dict.get((0, 1), 0) - 0.8) < 0.001
        # Perpendicular directions
        assert abs(trans_dict.get((1, 0), 0) - 0.1) < 0.001
        # The other perpendicular would be (-1, 0) but hits wall, stays in place
        assert abs(trans_dict.get((0, 0), 0) - 0.1) < 0.001
    
    def test_grid_mdp_obstacles(self):
        """Test GridMDP with obstacles"""
        grid = [
            [0, 0, 0],
            [0, None, 0],  # None represents obstacle
            [0, 0, 0]
        ]
        
        mdp = GridMDP(grid, terminals={(2, 2)})
        
        # Cannot move into obstacle
        transitions = mdp.T((0, 1), (1, 0))  # Try to move down into obstacle
        trans_dict = dict(transitions)
        
        # Should bounce back or move sideways
        assert (1, 1) not in trans_dict  # Cannot enter obstacle
    
    def test_grid_mdp_to_grid(self):
        """Test grid conversion with value mapping"""
        grid = [[0, 0], [0, 1]]
        mdp = GridMDP(grid, terminals={(1, 1)})
        
        # Create value mapping
        values = {
            (0, 0): 0.5,
            (0, 1): 0.7,
            (1, 0): 0.8,
            (1, 1): 1.0
        }
        
        grid_values = mdp.to_grid(values)
        
        assert len(grid_values) == 2
        assert len(grid_values[0]) == 2
        assert grid_values[0][0] == 0.5
        assert grid_values[1][1] == 1.0


class TestValueIterationAgent:
    """Test Value Iteration algorithm"""
    
    def test_value_iteration_basic(self):
        """Test basic value iteration"""
        mdp = SimpleMDP()
        agent = ValueIterationAgent(mdp, discount=0.9, iterations=100)
        
        # Check that values were computed
        assert len(agent.values) > 0
        
        # Terminal state should have value 0
        assert agent.get_value('Terminal') == 0
        
        # Non-terminal states should have values
        assert agent.get_value('S1') != 0
    
    def test_value_iteration_convergence(self):
        """Test that value iteration converges"""
        mdp = SimpleMDP()
        
        # Run with different iteration counts
        agent_10 = ValueIterationAgent(mdp, discount=0.9, iterations=10)
        agent_100 = ValueIterationAgent(mdp, discount=0.9, iterations=100)
        agent_1000 = ValueIterationAgent(mdp, discount=0.9, iterations=1000)
        
        # Values should stabilize
        v_100 = agent_100.get_value('S1')
        v_1000 = agent_1000.get_value('S1')
        
        # Should converge (difference should be small)
        assert abs(v_100 - v_1000) < 0.01
    
    def test_value_iteration_policy(self):
        """Test policy extraction from value iteration"""
        mdp = SimpleMDP()
        agent = ValueIterationAgent(mdp, discount=0.9, iterations=100)
        
        # Should have a policy for non-terminal states
        assert agent.get_policy('S1') in mdp.get_possible_actions('S1')
        assert agent.get_policy('S2') in mdp.get_possible_actions('S2')
        assert agent.get_policy('Terminal') is None
    
    def test_value_iteration_grid(self):
        """Test value iteration on GridMDP"""
        grid = [
            [-0.04, -0.04, -0.04, 1],
            [-0.04, None, -0.04, -1],
            [-0.04, -0.04, -0.04, -0.04]
        ]
        
        mdp = GridMDP(grid, terminals={(0, 3), (1, 3)})
        agent = ValueIterationAgent(mdp, discount=0.9, iterations=100)
        
        # Check values make sense
        # States closer to positive terminal should have higher values
        assert agent.get_value((0, 2)) > agent.get_value((0, 0))
        
        # States near negative terminal should have lower values
        assert agent.get_value((1, 2)) < agent.get_value((0, 2))
    
    def test_value_iteration_optimal_policy(self):
        """Test that value iteration finds optimal policy"""
        # Simple grid where optimal path is clear
        grid = [
            [0, 0, 0, 10],
            [0, 0, 0, 0],
            [-10, 0, 0, 0]
        ]
        
        mdp = GridMDP(grid, terminals={(0, 3), (2, 0)})
        agent = ValueIterationAgent(mdp, discount=0.9, iterations=100)
        
        # From (0, 0), should move right toward positive reward
        policy_00 = agent.get_policy((0, 0))
        assert policy_00[1] > 0  # Move right
        
        # From (2, 1), should move away from negative reward
        policy_21 = agent.get_policy((2, 1))
        assert policy_21[1] >= 0  # Not left


class TestPolicyIterationAgent:
    """Test Policy Iteration algorithm"""
    
    def test_policy_iteration_basic(self):
        """Test basic policy iteration"""
        mdp = SimpleMDP()
        agent = PolicyIterationAgent(mdp, discount=0.9)
        
        # Should have computed values and policy
        assert len(agent.values) > 0
        assert len(agent.policy) > 0
        
        # Check policy is valid
        for state in mdp.get_states():
            if not mdp.is_terminal(state):
                assert agent.get_policy(state) in mdp.get_possible_actions(state)
    
    def test_policy_iteration_convergence(self):
        """Test that policy iteration converges to optimal"""
        mdp = SimpleMDP()
        
        vi_agent = ValueIterationAgent(mdp, discount=0.9, iterations=1000)
        pi_agent = PolicyIterationAgent(mdp, discount=0.9)
        
        # Both should converge to similar values
        for state in mdp.get_states():
            vi_value = vi_agent.get_value(state)
            pi_value = pi_agent.get_value(state)
            assert abs(vi_value - pi_value) < 0.1
    
    def test_policy_iteration_grid(self):
        """Test policy iteration on GridMDP"""
        grid = [
            [-0.04, -0.04, -0.04, 1],
            [-0.04, None, -0.04, -1],
            [-0.04, -0.04, -0.04, -0.04]
        ]
        
        mdp = GridMDP(grid, terminals={(0, 3), (1, 3)})
        agent = PolicyIterationAgent(mdp, discount=0.9)
        
        # Should have policy for all non-terminal states
        for i in range(3):
            for j in range(4):
                if (i, j) not in mdp.terminals and grid[i][j] is not None:
                    policy = agent.get_policy((i, j))
                    assert policy is not None
                    assert policy in mdp.actions((i, j))
    
    def test_policy_iteration_improvement(self):
        """Test that policy improves during iteration"""
        # Create MDP where initial random policy is likely suboptimal
        grid = [
            [0, 0, 0, 100],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-100, 0, 0, 0]
        ]
        
        mdp = GridMDP(grid, terminals={(0, 3), (3, 0)})
        agent = PolicyIterationAgent(mdp, discount=0.9)
        
        # Final policy should navigate toward positive reward
        policy_00 = agent.get_policy((0, 0))
        # Should move right or down (toward positive terminal)
        assert policy_00 in [(0, 1), (1, 0)]


class TestMDPComparison:
    """Test comparing different MDP algorithms"""
    
    def create_test_mdp(self):
        """Create a standard test MDP"""
        grid = [
            [-0.04, -0.04, -0.04, 1],
            [-0.04, None, -0.04, -1],
            [-0.04, -0.04, -0.04, -0.04]
        ]
        return GridMDP(grid, terminals={(0, 3), (1, 3)}, gamma=0.9)
    
    def test_vi_vs_pi_values(self):
        """Compare values from VI and PI"""
        mdp = self.create_test_mdp()
        
        vi_agent = ValueIterationAgent(mdp, discount=0.9, iterations=100)
        pi_agent = PolicyIterationAgent(mdp, discount=0.9)
        
        # Values should be very similar
        for state in mdp.states:
            if state not in mdp.terminals:
                vi_val = vi_agent.get_value(state)
                pi_val = pi_agent.get_value(state)
                assert abs(vi_val - pi_val) < 0.01, f"State {state}: VI={vi_val}, PI={pi_val}"
    
    def test_vi_vs_pi_policies(self):
        """Compare policies from VI and PI"""
        mdp = self.create_test_mdp()
        
        vi_agent = ValueIterationAgent(mdp, discount=0.9, iterations=100)
        pi_agent = PolicyIterationAgent(mdp, discount=0.9)
        
        # Policies should be identical for optimal solution
        disagreements = 0
        for state in mdp.states:
            if state not in mdp.terminals:
                vi_policy = vi_agent.get_policy(state)
                pi_policy = pi_agent.get_policy(state)
                if vi_policy != pi_policy:
                    disagreements += 1
        
        # Should have very few or no disagreements
        assert disagreements <= 2  # Allow small differences due to ties


class TestDiscountFactor:
    """Test effect of discount factor"""
    
    def test_discount_factor_effect(self):
        """Test how discount factor affects policy"""
        grid = [
            [0, 0, 0, 10],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        terminals = {(0, 3)}
        
        # High discount: future rewards matter
        mdp_high = GridMDP(grid, terminals, gamma=0.95)
        agent_high = ValueIterationAgent(mdp_high, discount=0.95, iterations=100)
        
        # Low discount: immediate rewards matter more
        mdp_low = GridMDP(grid, terminals, gamma=0.1)
        agent_low = ValueIterationAgent(mdp_low, discount=0.1, iterations=100)
        
        # With high discount, distant states should have higher values
        # With low discount, distant states should have lower values
        assert agent_high.get_value((3, 0)) > agent_low.get_value((3, 0))
    
    def test_myopic_vs_farsighted(self):
        """Test myopic vs farsighted behavior"""
        # Grid with immediate small reward vs distant large reward
        grid = [
            [0, 1, 0, 0, 100],
            [0, 0, 0, 0, 0]
        ]
        terminals = {(0, 1), (0, 4)}
        
        # Very low discount (myopic)
        mdp_myopic = GridMDP(grid, terminals, gamma=0.1, init=(0, 0))
        agent_myopic = ValueIterationAgent(mdp_myopic, discount=0.1, iterations=100)
        
        # High discount (farsighted)
        mdp_farsighted = GridMDP(grid, terminals, gamma=0.99, init=(0, 0))
        agent_farsighted = ValueIterationAgent(mdp_farsighted, discount=0.99, iterations=100)
        
        # Myopic agent might prefer immediate small reward
        # Farsighted agent should prefer distant large reward
        policy_myopic = agent_myopic.get_policy((0, 0))
        policy_farsighted = agent_farsighted.get_policy((0, 0))
        
        # Policies might differ based on discount
        assert policy_myopic is not None
        assert policy_farsighted is not None


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_mdp(self):
        """Test MDP with minimal states"""
        mdp = MDP(
            init='S1',
            actlist=[],
            terminals={'S1'},
            gamma=0.9
        )
        
        agent = ValueIterationAgent(mdp, discount=0.9, iterations=10)
        assert agent.get_value('S1') == 0  # Terminal state
        assert agent.get_policy('S1') is None
    
    def test_single_state_mdp(self):
        """Test MDP with single non-terminal state"""
        transitions = {
            'S1': {'a': [('S1', 1.0)]}  # Self-loop
        }
        
        mdp = MDP(
            init='S1',
            actlist=['a'],
            terminals=set(),
            transitions=transitions,
            reward=lambda s: 1,  # Constant reward
            gamma=0.9
        )
        
        agent = ValueIterationAgent(mdp, discount=0.9, iterations=100)
        
        # Value should be r/(1-γ) = 1/(1-0.9) = 10
        assert abs(agent.get_value('S1') - 10) < 1
    
    def test_deterministic_vs_stochastic(self):
        """Test deterministic vs stochastic transitions"""
        # Deterministic MDP
        det_transitions = {
            'S1': {'a': [('S2', 1.0)], 'b': [('S3', 1.0)]},
            'S2': {'a': [('Terminal', 1.0)]},
            'S3': {'a': [('Terminal', 1.0)]}
        }
        
        # Stochastic MDP
        stoch_transitions = {
            'S1': {'a': [('S2', 0.5), ('S3', 0.5)], 'b': [('S3', 1.0)]},
            'S2': {'a': [('Terminal', 1.0)]},
            'S3': {'a': [('Terminal', 1.0)]}
        }
        
        reward_fn = lambda s: 10 if s == 'Terminal' else -1
        
        det_mdp = MDP('S1', ['a', 'b'], {'Terminal'}, det_transitions, reward_fn, 0.9)
        stoch_mdp = MDP('S1', ['a', 'b'], {'Terminal'}, stoch_transitions, reward_fn, 0.9)
        
        det_agent = ValueIterationAgent(det_mdp, 0.9, 100)
        stoch_agent = ValueIterationAgent(stoch_mdp, 0.9, 100)
        
        # Values should differ due to stochasticity
        assert det_agent.get_value('S1') != stoch_agent.get_value('S1')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
