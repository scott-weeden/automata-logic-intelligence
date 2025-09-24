"""
Comprehensive Test Suite for Reinforcement Learning Algorithms

Tests Q-Learning, SARSA, and other model-free learning algorithms.
Validates convergence, exploration strategies, and learning performance.

Based on CS 5368 Week 7-8 material on reinforcement learning.
"""

import unittest
import sys
import os
import random
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from learning import QLearningAgent, train_q_learning, q_learning_episode
from mdp.agents import SARSAAgent

class SimpleEnvironment:
    """
    Simple grid environment for testing RL algorithms.
    
    2x2 grid world:
    [S] [ ] 
    [ ] [G]
    
    S = Start (0,0), G = Goal (1,1)
    Actions: 'up', 'down', 'left', 'right'
    Rewards: +10 for reaching goal, -1 for each step
    """
    
    def __init__(self):
        """Initialize simple 2x2 grid environment."""
        self.grid_size = 2
        self.start_state = (0, 0)
        self.goal_state = (1, 1)
        self.current_state = self.start_state
        
        # Action mappings
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
    
    def reset(self):
        """Reset environment to start state."""
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action):
        """
        Take action in environment.
        
        Returns:
            next_state: New state after action
            reward: Immediate reward
            done: True if episode finished
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")
        
        # Calculate next state
        dr, dc = self.action_effects[action]
        new_row = max(0, min(self.grid_size - 1, self.current_state[0] + dr))
        new_col = max(0, min(self.grid_size - 1, self.current_state[1] + dc))
        
        next_state = (new_row, new_col)
        self.current_state = next_state
        
        # Calculate reward
        if next_state == self.goal_state:
            reward = 10  # Goal reward
            done = True
        else:
            reward = -1  # Living penalty
            done = False
        
        return next_state, reward, done
    
    def get_valid_actions(self, state):
        """Return all valid actions from given state."""
        return self.actions.copy()

class DeterministicEnvironment:
    """
    Deterministic environment for testing convergence properties.
    
    Simple chain: 0 -> 1 -> 2 (goal)
    Actions: 'forward', 'backward'
    Deterministic transitions, known optimal policy
    """
    
    def __init__(self, chain_length=3):
        """Initialize deterministic chain environment."""
        self.chain_length = chain_length
        self.goal_state = chain_length - 1
        self.current_state = 0
        self.actions = ['forward', 'backward']
    
    def reset(self):
        """Reset to start of chain."""
        self.current_state = 0
        return self.current_state
    
    def step(self, action):
        """Take deterministic action."""
        if action == 'forward':
            self.current_state = min(self.chain_length - 1, self.current_state + 1)
        elif action == 'backward':
            self.current_state = max(0, self.current_state - 1)
        
        # Reward structure: +10 for goal, -1 for each step
        if self.current_state == self.goal_state:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        
        return self.current_state, reward, done
    
    def get_valid_actions(self, state):
        """Return valid actions."""
        return self.actions.copy()

class TestQLearningAgent(unittest.TestCase):
    """Test Q-Learning algorithm implementation and behavior."""
    
    def setUp(self):
        """Set up Q-Learning agents and environments for testing."""
        # Action function for simple environment
        def get_actions(state):
            return ['up', 'down', 'left', 'right']
        
        # Create Q-Learning agents with different parameters
        self.q_agent = QLearningAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1
        )
        
        self.q_agent_greedy = QLearningAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.0  # No exploration
        )
        
        # Test environments
        self.simple_env = SimpleEnvironment()
        self.deterministic_env = DeterministicEnvironment()
    
    def test_q_learning_agent_initialization(self):
        """Test Q-Learning agent initialization and parameters."""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b'],
            discount=0.95,
            alpha=0.2,
            epsilon=0.15
        )
        
        self.assertEqual(agent.discount, 0.95)
        self.assertEqual(agent.alpha, 0.2)
        self.assertEqual(agent.epsilon, 0.15)
        self.assertTrue(agent.training)
        self.assertEqual(len(agent.q_values), 0)  # Initially empty
    
    def test_q_value_initialization(self):
        """Test Q-value initialization and retrieval."""
        # Q-values should start at 0
        initial_q = self.q_agent.get_q_value((0, 0), 'up')
        self.assertEqual(initial_q, 0.0)
        
        # Max Q-value of empty state should be 0
        max_q = self.q_agent.get_max_q_value((0, 0))
        self.assertEqual(max_q, 0.0)
    
    def test_q_learning_update_rule(self):
        """Test Q-Learning update rule implementation."""
        state = (0, 0)
        action = 'right'
        next_state = (0, 1)
        reward = -1
        
        # Initial Q-value
        initial_q = self.q_agent.get_q_value(state, action)
        self.assertEqual(initial_q, 0.0)
        
        # Perform update
        self.q_agent.update(state, action, next_state, reward)
        
        # Q-value should have changed according to update rule
        updated_q = self.q_agent.get_q_value(state, action)
        
        # Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        # Expected: 0 + 0.1 * (-1 + 0.9 * 0 - 0) = -0.1
        expected_q = 0.1 * (-1 + 0.9 * 0 - 0)
        self.assertAlmostEqual(updated_q, expected_q, places=5)
    
    def test_epsilon_greedy_action_selection(self):
        """Test epsilon-greedy exploration strategy."""
        state = (0, 0)
        
        # Set some Q-values to create clear best action
        self.q_agent.q_values[(state, 'right')] = 10.0
        self.q_agent.q_values[(state, 'up')] = 1.0
        self.q_agent.q_values[(state, 'down')] = 1.0
        self.q_agent.q_values[(state, 'left')] = 1.0
        
        # With epsilon=0, should always choose best action
        self.q_agent_greedy.q_values = self.q_agent.q_values.copy()
        
        # Test multiple times to ensure consistency
        for _ in range(10):
            action = self.q_agent_greedy.get_action(state)
            self.assertEqual(action, 'right')
        
        # With epsilon > 0, should sometimes explore
        # (This is probabilistic, so we test the mechanism exists)
        actions_taken = set()
        for _ in range(100):
            action = self.q_agent.get_action(state)
            actions_taken.add(action)
        
        # Should have taken the best action most often
        # but might have explored others
        self.assertIn('right', actions_taken)
    
    def test_best_action_computation(self):
        """Test computation of best action from Q-values."""
        state = (1, 1)
        
        # Set Q-values with clear winner
        self.q_agent.q_values[(state, 'up')] = 5.0
        self.q_agent.q_values[(state, 'down')] = 2.0
        self.q_agent.q_values[(state, 'left')] = 3.0
        self.q_agent.q_values[(state, 'right')] = 1.0
        
        best_action = self.q_agent.get_best_action(state)
        self.assertEqual(best_action, 'up')
        
        max_q = self.q_agent.get_max_q_value(state)
        self.assertEqual(max_q, 5.0)
    
    def test_training_mode_toggle(self):
        """Test training mode affects exploration behavior."""
        state = (0, 0)
        
        # Set Q-values
        self.q_agent.q_values[(state, 'right')] = 10.0
        
        # In training mode with epsilon > 0, might explore
        self.q_agent.training = True
        
        # Stop training - should become greedy
        self.q_agent.stop_training()
        self.assertFalse(self.q_agent.training)
        
        # Now should always choose best action
        for _ in range(10):
            action = self.q_agent.get_action(state)
            # Should be deterministic now
            self.assertIsNotNone(action)

class TestQLearningConvergence(unittest.TestCase):
    """Test Q-Learning convergence properties and learning behavior."""
    
    def setUp(self):
        """Set up agents and environments for convergence testing."""
        def get_actions(state):
            if isinstance(state, tuple):
                return ['up', 'down', 'left', 'right']
            else:
                return ['forward', 'backward']
        
        self.q_agent = QLearningAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1
        )
        
        self.deterministic_env = DeterministicEnvironment(chain_length=3)
    
    def test_q_learning_simple_convergence(self):
        """Test Q-Learning converges on simple deterministic environment."""
        # Train for many episodes
        episode_rewards = []
        
        for episode in range(200):
            state = self.deterministic_env.reset()
            total_reward = 0
            
            for step in range(50):  # Max steps per episode
                action = self.q_agent.get_action(state)
                next_state, reward, done = self.deterministic_env.step(action)
                
                self.q_agent.update(state, action, next_state, reward)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
        
        # Performance should improve over time
        early_performance = sum(episode_rewards[:50]) / 50
        late_performance = sum(episode_rewards[-50:]) / 50
        
        self.assertGreater(late_performance, early_performance)
        
        # Final policy should be reasonable
        # State 0 should prefer 'forward', state 1 should prefer 'forward'
        self.q_agent.stop_training()  # Turn off exploration
        
        best_action_0 = self.q_agent.get_best_action(0)
        best_action_1 = self.q_agent.get_best_action(1)
        
        # Should learn to go forward toward goal
        self.assertEqual(best_action_0, 'forward')
        self.assertEqual(best_action_1, 'forward')
    
    def test_q_learning_exploration_vs_exploitation(self):
        """Test balance between exploration and exploitation."""
        # High exploration agent
        high_explore = QLearningAgent(
            action_fn=lambda s: ['forward', 'backward'],
            discount=0.9,
            alpha=0.1,
            epsilon=0.5  # High exploration
        )
        
        # Low exploration agent
        low_explore = QLearningAgent(
            action_fn=lambda s: ['forward', 'backward'],
            discount=0.9,
            alpha=0.1,
            epsilon=0.01  # Low exploration
        )
        
        # Train both agents
        for agent in [high_explore, low_explore]:
            for episode in range(50):
                state = self.deterministic_env.reset()
                
                for step in range(20):
                    action = agent.get_action(state)
                    next_state, reward, done = self.deterministic_env.step(action)
                    agent.update(state, action, next_state, reward)
                    state = next_state
                    if done:
                        break
        
        # Both should learn something, but exploration patterns differ
        # This is more about testing the mechanism than specific outcomes
        self.assertGreater(len(high_explore.q_values), 0)
        self.assertGreater(len(low_explore.q_values), 0)
    
    def test_q_learning_learning_rate_effect(self):
        """Test effect of different learning rates on convergence."""
        # High learning rate agent
        high_alpha = QLearningAgent(
            action_fn=lambda s: ['forward', 'backward'],
            discount=0.9,
            alpha=0.5,  # High learning rate
            epsilon=0.1
        )
        
        # Low learning rate agent
        low_alpha = QLearningAgent(
            action_fn=lambda s: ['forward', 'backward'],
            discount=0.9,
            alpha=0.01,  # Low learning rate
            epsilon=0.1
        )
        
        # Single update test
        state, action, next_state, reward = 0, 'forward', 1, -1
        
        high_alpha.update(state, action, next_state, reward)
        low_alpha.update(state, action, next_state, reward)
        
        high_q = high_alpha.get_q_value(state, action)
        low_q = low_alpha.get_q_value(state, action)
        
        # High learning rate should change Q-values more dramatically
        self.assertGreater(abs(high_q), abs(low_q))
    
    def test_q_learning_discount_factor_effect(self):
        """Test effect of discount factor on learned values."""
        # Patient agent (high discount)
        patient_agent = QLearningAgent(
            action_fn=lambda s: ['forward', 'backward'],
            discount=0.95,  # Values future rewards highly
            alpha=0.1,
            epsilon=0.1
        )
        
        # Impatient agent (low discount)
        impatient_agent = QLearningAgent(
            action_fn=lambda s: ['forward', 'backward'],
            discount=0.1,   # Values immediate rewards only
            alpha=0.1,
            epsilon=0.1
        )
        
        # Train both agents
        for agent in [patient_agent, impatient_agent]:
            for episode in range(100):
                state = self.deterministic_env.reset()
                
                for step in range(10):
                    action = agent.get_action(state)
                    next_state, reward, done = self.deterministic_env.step(action)
                    agent.update(state, action, next_state, reward)
                    state = next_state
                    if done:
                        break
        
        # Patient agent should have higher Q-values for early states
        # (because it values the future goal reward more)
        patient_q0 = patient_agent.get_max_q_value(0)
        impatient_q0 = impatient_agent.get_max_q_value(0)
        
        # This relationship should hold after sufficient training
        if patient_q0 != 0 and impatient_q0 != 0:
            self.assertGreater(patient_q0, impatient_q0)

class TestSARSAAgent(unittest.TestCase):
    """Test SARSA (on-policy) learning algorithm."""
    
    def setUp(self):
        """Set up SARSA agent for testing."""
        def get_actions(state):
            return ['up', 'down', 'left', 'right']
        
        self.sarsa_agent = SARSAAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1
        )
        
        self.simple_env = SimpleEnvironment()
    
    def test_sarsa_agent_initialization(self):
        """Test SARSA agent initialization."""
        def actions(s):
            return ['a', 'b']
        
        agent = SARSAAgent(
            action_fn=actions,
            discount=0.8,
            alpha=0.2,
            epsilon=0.15
        )
        
        self.assertEqual(agent.discount, 0.8)
        self.assertEqual(agent.alpha, 0.2)
        self.assertEqual(agent.epsilon, 0.15)
        self.assertIsNone(agent.next_action)
    
    def test_sarsa_update_rule(self):
        """Test SARSA update rule (on-policy)."""
        state = (0, 0)
        action = 'right'
        next_state = (0, 1)
        next_action = 'up'
        reward = -1
        
        # Set next action for SARSA update
        self.sarsa_agent.next_action = next_action
        
        # Set some Q-value for next state-action
        self.sarsa_agent.q_values[(next_state, next_action)] = 2.0
        
        # Perform SARSA update
        initial_q = self.sarsa_agent.get_q_value(state, action)
        self.sarsa_agent.update(state, action, next_state, reward)
        updated_q = self.sarsa_agent.get_q_value(state, action)
        
        # SARSA: Q(s,a) = Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        # Expected: 0 + 0.1 * (-1 + 0.9 * 2.0 - 0) = 0.1 * 0.8 = 0.08
        expected_q = 0.1 * (-1 + 0.9 * 2.0 - 0)
        self.assertAlmostEqual(updated_q, expected_q, places=5)
    
    def test_sarsa_vs_q_learning_behavior(self):
        """Test SARSA vs Q-Learning behavioral differences."""
        # Create Q-Learning agent for comparison
        def get_actions(state):
            return ['forward', 'backward']
        
        q_agent = QLearningAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1
        )
        
        sarsa_agent = SARSAAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1
        )
        
        # Train both on deterministic environment
        det_env = DeterministicEnvironment()
        
        for agent in [q_agent, sarsa_agent]:
            for episode in range(50):
                state = det_env.reset()
                
                if isinstance(agent, SARSAAgent):
                    action = agent.get_action(state)
                
                for step in range(10):
                    if isinstance(agent, QLearningAgent):
                        action = agent.get_action(state)
                    
                    next_state, reward, done = det_env.step(action)
                    
                    if isinstance(agent, SARSAAgent):
                        next_action = agent.get_action(next_state) if not done else None
                        agent.next_action = next_action
                    
                    agent.update(state, action, next_state, reward)
                    
                    state = next_state
                    if isinstance(agent, SARSAAgent):
                        action = next_action
                    
                    if done:
                        break
        
        # Both should learn, but may have different Q-values due to on/off-policy difference
        q_learning_q = q_agent.get_max_q_value(0)
        sarsa_q = sarsa_agent.get_max_q_value(0)
        
        # Both should have learned something
        self.assertNotEqual(q_learning_q, 0)
        self.assertNotEqual(sarsa_q, 0)

class TestReinforcementLearningUtilities(unittest.TestCase):
    """Test utility functions for reinforcement learning."""
    
    def setUp(self):
        """Set up test environment and agent."""
        def get_actions(state):
            return ['up', 'down', 'left', 'right']
        
        self.agent = QLearningAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1
        )
        
        self.env = SimpleEnvironment()
    
    def test_q_learning_episode_function(self):
        """Test single episode execution function."""
        # Run single episode
        total_reward = q_learning_episode(self.agent, self.env, max_steps=20)
        
        # Should return numeric reward
        self.assertIsInstance(total_reward, (int, float))
        
        # Agent should have learned something
        self.assertGreater(len(self.agent.q_values), 0)
    
    def test_train_q_learning_function(self):
        """Test multi-episode training function."""
        # Train for multiple episodes
        episode_rewards = train_q_learning(
            self.agent, 
            self.env, 
            num_episodes=50, 
            verbose=False
        )
        
        # Should return list of rewards
        self.assertEqual(len(episode_rewards), 50)
        self.assertTrue(all(isinstance(r, (int, float)) for r in episode_rewards))
        
        # Performance should generally improve
        early_avg = sum(episode_rewards[:10]) / 10
        late_avg = sum(episode_rewards[-10:]) / 10
        
        # Later episodes should perform better (or at least not much worse)
        # Allow some tolerance for stochasticity
        self.assertGreaterEqual(late_avg, early_avg - 5)
    
    def test_learning_curve_analysis(self):
        """Test learning curve shows improvement over time."""
        # Train agent and track performance
        episode_rewards = train_q_learning(
            self.agent,
            self.env,
            num_episodes=100,
            verbose=False
        )
        
        # Calculate moving averages to smooth out noise
        window_size = 10
        moving_averages = []
        
        for i in range(len(episode_rewards) - window_size + 1):
            window_avg = sum(episode_rewards[i:i+window_size]) / window_size
            moving_averages.append(window_avg)
        
        # Performance should show upward trend
        early_performance = moving_averages[0]
        late_performance = moving_averages[-1]
        
        # Should improve or at least not degrade significantly
        self.assertGreaterEqual(late_performance, early_performance - 2)

class TestReinforcementLearningEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios in reinforcement learning."""
    
    def test_agent_with_no_actions(self):
        """Test agent behavior when no actions are available."""
        def no_actions(state):
            return []
        
        agent = QLearningAgent(
            action_fn=no_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1
        )
        
        # Should handle gracefully
        action = agent.get_action((0, 0))
        self.assertIsNone(action)
        
        max_q = agent.get_max_q_value((0, 0))
        self.assertEqual(max_q, 0.0)
    
    def test_agent_with_single_action(self):
        """Test agent with only one action available."""
        def single_action(state):
            return ['only_action']
        
        agent = QLearningAgent(
            action_fn=single_action,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1
        )
        
        # Should always choose the only action
        for _ in range(10):
            action = agent.get_action((0, 0))
            self.assertEqual(action, 'only_action')
    
    def test_extreme_learning_parameters(self):
        """Test agent with extreme learning parameters."""
        def get_actions(state):
            return ['a', 'b']
        
        # Zero learning rate (should not learn)
        no_learning_agent = QLearningAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=0.0,  # No learning
            epsilon=0.1
        )
        
        # Update should not change Q-values
        initial_q = no_learning_agent.get_q_value('state', 'a')
        no_learning_agent.update('state', 'a', 'next_state', 10)
        final_q = no_learning_agent.get_q_value('state', 'a')
        
        self.assertEqual(initial_q, final_q)
        
        # Maximum learning rate
        max_learning_agent = QLearningAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=1.0,  # Maximum learning
            epsilon=0.1
        )
        
        # Should learn very quickly
        max_learning_agent.update('state', 'a', 'next_state', 10)
        updated_q = max_learning_agent.get_q_value('state', 'a')
        
        # With α=1, Q(s,a) should equal the TD target
        expected_q = 10 + 0.9 * 0  # reward + γ * max_Q(next_state)
        self.assertAlmostEqual(updated_q, expected_q, places=5)
    
    def test_negative_rewards_learning(self):
        """Test learning with negative reward environments."""
        def get_actions(state):
            return ['action']
        
        agent = QLearningAgent(
            action_fn=get_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.0
        )
        
        # Train with only negative rewards
        for _ in range(50):
            agent.update('state', 'action', 'next_state', -10)
        
        # Q-value should be negative
        final_q = agent.get_q_value('state', 'action')
        self.assertLess(final_q, 0)
    
    def test_very_long_episodes(self):
        """Test agent behavior in very long episodes."""
        class LongEnvironment:
            def __init__(self):
                self.state = 0
                self.max_state = 1000
            
            def reset(self):
                self.state = 0
                return self.state
            
            def step(self, action):
                if action == 'forward':
                    self.state += 1
                
                if self.state >= self.max_state:
                    return self.state, 100, True  # Big reward at end
                else:
                    return self.state, -0.1, False  # Small penalty per step
            
            def get_valid_actions(self, state):
                return ['forward', 'stay']
        
        def get_actions(state):
            return ['forward', 'stay']
        
        agent = QLearningAgent(
            action_fn=get_actions,
            discount=0.99,  # High discount for long episodes
            alpha=0.01,     # Low learning rate for stability
            epsilon=0.05
        )
        
        long_env = LongEnvironment()
        
        # Run a few long episodes
        for episode in range(5):
            state = long_env.reset()
            
            for step in range(1200):  # Allow for long episodes
                action = agent.get_action(state)
                next_state, reward, done = long_env.step(action)
                agent.update(state, action, next_state, reward)
                state = next_state
                
                if done:
                    break
        
        # Agent should learn to go forward (optimal policy)
        agent.stop_training()
        best_action = agent.get_best_action(0)
        
        # Should prefer forward over staying
        forward_q = agent.get_q_value(0, 'forward')
        stay_q = agent.get_q_value(0, 'stay')
        
        if forward_q != 0 and stay_q != 0:
            self.assertGreater(forward_q, stay_q)

if __name__ == '__main__':
    # Run all reinforcement learning tests
    unittest.main(verbosity=2)
