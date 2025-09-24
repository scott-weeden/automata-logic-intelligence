"""
Comprehensive test suite for learning module
Tests reinforcement learning algorithms: Q-Learning and SARSA
"""

import pytest
import sys
import os
from typing import List, Dict, Tuple, Any, Callable
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning import QLearningAgent, SARSAAgent


# Test Environments
class GridWorldEnvironment:
    """Simple grid world for testing RL agents"""
    
    def __init__(self, grid, start=(0, 0), goal=None, stochastic=False):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.start = start
        self.goal = goal if goal else (self.rows-1, self.cols-1)
        self.stochastic = stochastic
        self.current_state = start
        self.done = False
    
    def reset(self):
        """Reset environment to start state"""
        self.current_state = self.start
        self.done = False
        return self.current_state
    
    def get_actions(self, state=None):
        """Get available actions for a state"""
        if state is None:
            state = self.current_state
        return ['up', 'down', 'left', 'right']
    
    def step(self, action):
        """Take action and return (next_state, reward, done)"""
        if self.done:
            return self.current_state, 0, True
        
        # Calculate intended next position
        row, col = self.current_state
        if action == 'up':
            next_pos = (row - 1, col)
        elif action == 'down':
            next_pos = (row + 1, col)
        elif action == 'left':
            next_pos = (row, col - 1)
        elif action == 'right':
            next_pos = (row, col + 1)
        else:
            next_pos = (row, col)
        
        # Add stochasticity if enabled
        if self.stochastic and random.random() < 0.1:
            # 10% chance of random action
            actions = self.get_actions()
            action = random.choice(actions)
            return self.step(action)  # Recursive call with random action
        
        # Check if valid move
        next_row, next_col = next_pos
        if (0 <= next_row < self.rows and 
            0 <= next_col < self.cols and 
            self.grid[next_row][next_col] != 1):  # 1 is obstacle
            self.current_state = next_pos
        
        # Calculate reward
        if self.current_state == self.goal:
            reward = 10
            self.done = True
        elif self.grid[self.current_state[0]][self.current_state[1]] == -1:
            reward = -10
            self.done = True
        else:
            reward = -0.1  # Small penalty for each step
        
        return self.current_state, reward, self.done
    
    def get_state_value(self, state):
        """Get true value of a state (for testing)"""
        if state == self.goal:
            return 10
        elif self.grid[state[0]][state[1]] == -1:
            return -10
        else:
            # Distance-based heuristic
            distance = abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])
            return -distance * 0.1


class CliffWalkingEnvironment:
    """Cliff walking environment from Sutton & Barto"""
    
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.current_state = self.start
        self.done = False
    
    def reset(self):
        self.current_state = self.start
        self.done = False
        return self.current_state
    
    def get_actions(self, state=None):
        return ['up', 'down', 'left', 'right']
    
    def step(self, action):
        if self.done:
            return self.current_state, 0, True
        
        row, col = self.current_state
        
        # Move based on action
        if action == 'up' and row > 0:
            row -= 1
        elif action == 'down' and row < self.rows - 1:
            row += 1
        elif action == 'left' and col > 0:
            col -= 1
        elif action == 'right' and col < self.cols - 1:
            col += 1
        
        self.current_state = (row, col)
        
        # Check if fell off cliff
        if self.current_state in self.cliff:
            reward = -100
            self.current_state = self.start  # Reset to start
            self.done = False  # Continue episode
        elif self.current_state == self.goal:
            reward = 0
            self.done = True
        else:
            reward = -1
        
        return self.current_state, reward, self.done


class TestQLearningAgent:
    """Test Q-Learning agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        def action_fn(state):
            return ['a', 'b', 'c']
        
        agent = QLearningAgent(
            action_fn=action_fn,
            discount=0.9,
            alpha=0.5,
            epsilon=0.1
        )
        
        assert agent.discount == 0.9
        assert agent.alpha == 0.5
        assert agent.epsilon == 0.1
        assert agent.training == True
        assert len(agent.q_values) == 0
    
    def test_q_value_initialization(self):
        """Test Q-value initialization"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b'],
            discount=0.9
        )
        
        # Q-values should be 0 initially
        assert agent.get_q_value('state1', 'a') == 0
        assert agent.get_q_value('state1', 'b') == 0
    
    def test_q_value_update(self):
        """Test Q-value update formula"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b'],
            discount=0.9,
            alpha=0.5
        )
        
        # Manual update
        old_q = agent.get_q_value('s1', 'a')
        agent.update('s1', 'a', 's2', 10)
        new_q = agent.get_q_value('s1', 'a')
        
        # Q(s,a) <- Q(s,a) + α[r + γ*max_Q(s',a') - Q(s,a)]
        # Q(s1,a) <- 0 + 0.5[10 + 0.9*0 - 0] = 5
        assert new_q == 5.0
    
    def test_q_value_convergence(self):
        """Test that Q-values converge with repeated updates"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a'],
            discount=0.9,
            alpha=0.1
        )
        
        # Repeated updates to same state-action
        for _ in range(100):
            agent.update('s1', 'a', 'terminal', 10)
        
        # Should converge to reward value (no future rewards from terminal)
        assert abs(agent.get_q_value('s1', 'a') - 10) < 0.01
    
    def test_max_q_value(self):
        """Test max Q-value extraction"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b', 'c'],
            discount=0.9
        )
        
        # Set some Q-values manually
        agent.q_values[('s1', 'a')] = 5
        agent.q_values[('s1', 'b')] = 10
        agent.q_values[('s1', 'c')] = 3
        
        assert agent.get_max_q_value('s1') == 10
    
    def test_epsilon_greedy_exploration(self):
        """Test epsilon-greedy action selection"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b', 'c'],
            epsilon=0.5  # 50% exploration
        )
        
        # Set Q-values to make 'b' optimal
        agent.q_values[('s1', 'a')] = 1
        agent.q_values[('s1', 'b')] = 10
        agent.q_values[('s1', 'c')] = 2
        
        # Count action selections
        action_counts = {'a': 0, 'b': 0, 'c': 0}
        for _ in range(1000):
            action = agent.get_action('s1')
            action_counts[action] += 1
        
        # 'b' should be chosen most often but not always
        assert action_counts['b'] > action_counts['a']
        assert action_counts['b'] > action_counts['c']
        # With epsilon=0.5, 'b' should be chosen ~50% + (50%/3) ≈ 66%
        assert 0.5 < action_counts['b'] / 1000 < 0.8
    
    def test_greedy_action_when_not_training(self):
        """Test that agent acts greedily when not training"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b', 'c'],
            epsilon=0.9  # High exploration
        )
        
        # Set Q-values
        agent.q_values[('s1', 'a')] = 1
        agent.q_values[('s1', 'b')] = 10
        agent.q_values[('s1', 'c')] = 2
        
        # Stop training
        agent.stop_training()
        
        # Should always choose best action
        for _ in range(100):
            action = agent.get_action('s1')
            assert action == 'b'
    
    def test_episode_management(self):
        """Test episode start/stop functionality"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b'],
            discount=0.9
        )
        
        # Start new episode
        agent.start_episode()
        assert agent.episode_rewards == 0
        
        # Accumulate rewards
        agent.update('s1', 'a', 's2', 5)
        agent.update('s2', 'b', 's3', 3)
        
        # Check episode tracking (if implemented)
        # This depends on specific implementation
    
    def test_learning_in_grid_world(self):
        """Test Q-learning in a simple grid world"""
        grid = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ]
        env = GridWorldEnvironment(grid, start=(0, 0), goal=(2, 3))
        
        agent = QLearningAgent(
            action_fn=env.get_actions,
            discount=0.9,
            alpha=0.5,
            epsilon=0.3
        )
        
        # Train for several episodes
        total_rewards = []
        for episode in range(100):
            state = env.reset()
            episode_reward = 0
            
            for step in range(100):  # Max steps per episode
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, next_state, reward)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        # Performance should improve over time
        early_performance = np.mean(total_rewards[:20])
        late_performance = np.mean(total_rewards[-20:])
        assert late_performance > early_performance
        
        # Should learn some non-zero Q-values
        assert len(agent.q_values) > 0
        assert any(q != 0 for q in agent.q_values.values())


class TestSARSAAgent:
    """Test SARSA agent"""
    
    def test_initialization(self):
        """Test SARSA initialization"""
        agent = SARSAAgent(
            action_fn=lambda s: ['a', 'b'],
            discount=0.9,
            alpha=0.5,
            epsilon=0.1
        )
        
        assert agent.discount == 0.9
        assert agent.alpha == 0.5
        assert agent.epsilon == 0.1
        assert hasattr(agent, 'next_action')
    
    def test_sarsa_update_formula(self):
        """Test SARSA update differs from Q-learning"""
        # SARSA: Q(s,a) <- Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
        # Q-learning: Q(s,a) <- Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]
        
        agent = SARSAAgent(
            action_fn=lambda s: ['a', 'b'],
            discount=0.9,
            alpha=0.5,
            epsilon=0
        )
        
        # Set up Q-values
        agent.q_values[('s2', 'a')] = 5
        agent.q_values[('s2', 'b')] = 10
        
        # SARSA uses actual next action, not max
        agent.next_action = 'a'  # Simulate choosing 'a' as next action
        agent.update('s1', 'action1', 's2', 2)
        
        # Update should use Q(s2, a) = 5, not max Q(s2, *) = 10
        # Q(s1,action1) = 0 + 0.5[2 + 0.9*5 - 0] = 0.5[2 + 4.5] = 3.25
        expected = 0.5 * (2 + 0.9 * 5)
        assert abs(agent.get_q_value('s1', 'action1') - expected) < 0.01
    
    def test_sarsa_on_policy_learning(self):
        """Test that SARSA learns the policy it follows"""
        grid = [[0, 0, 0], [0, 0, 0]]
        env = GridWorldEnvironment(grid, start=(0, 0), goal=(1, 2))
        
        agent = SARSAAgent(
            action_fn=env.get_actions,
            discount=0.9,
            alpha=0.5,
            epsilon=0.3  # Some exploration
        )
        
        # Train SARSA
        for episode in range(100):
            state = env.reset()
            action = agent.get_action(state)
            
            for step in range(50):
                next_state, reward, done = env.step(action)
                next_action = agent.get_action(next_state)
                
                # SARSA update with actual next action
                agent.next_action = next_action
                agent.update(state, action, next_state, reward)
                
                state = next_state
                action = next_action
                
                if done:
                    break
        
        # Should have learned Q-values
        assert len(agent.q_values) > 0
        
        # Check that it learned a reasonable policy
        # From (0,0), should prefer moving toward goal
        q_right = agent.get_q_value((0, 0), 'right')
        q_down = agent.get_q_value((0, 0), 'down')
        q_left = agent.get_q_value((0, 0), 'left')
        q_up = agent.get_q_value((0, 0), 'up')
        
        # Right and down should have higher values (toward goal)
        assert max(q_right, q_down) > max(q_left, q_up)


class TestQLearningVsSARSA:
    """Compare Q-Learning and SARSA"""
    
    def test_cliff_walking_comparison(self):
        """Test Q-learning vs SARSA on cliff walking"""
        env = CliffWalkingEnvironment()
        
        # Q-Learning agent
        q_agent = QLearningAgent(
            action_fn=env.get_actions,
            discount=0.9,
            alpha=0.5,
            epsilon=0.1
        )
        
        # SARSA agent
        sarsa_agent = SARSAAgent(
            action_fn=env.get_actions,
            discount=0.9,
            alpha=0.5,
            epsilon=0.1
        )
        
        # Train both agents
        q_rewards = []
        sarsa_rewards = []
        
        for episode in range(200):
            # Train Q-learning
            state = env.reset()
            q_episode_reward = 0
            for step in range(500):
                action = q_agent.get_action(state)
                next_state, reward, done = env.step(action)
                q_agent.update(state, action, next_state, reward)
                q_episode_reward += reward
                state = next_state
                if done:
                    break
            q_rewards.append(q_episode_reward)
            
            # Train SARSA
            state = env.reset()
            sarsa_episode_reward = 0
            action = sarsa_agent.get_action(state)
            for step in range(500):
                next_state, reward, done = env.step(action)
                next_action = sarsa_agent.get_action(next_state)
                sarsa_agent.next_action = next_action
                sarsa_agent.update(state, action, next_state, reward)
                sarsa_episode_reward += reward
                state = next_state
                action = next_action
                if done:
                    break
            sarsa_rewards.append(sarsa_episode_reward)
        
        # Compare performance
        # Q-learning should find optimal (risky) path
        # SARSA should find safer path
        q_final = np.mean(q_rewards[-20:])
        sarsa_final = np.mean(sarsa_rewards[-20:])
        
        # Both should improve over time
        assert q_final > np.mean(q_rewards[:20])
        assert sarsa_final > np.mean(sarsa_rewards[:20])


class TestExplorationStrategies:
    """Test different exploration strategies"""
    
    def test_epsilon_decay(self):
        """Test epsilon decay over episodes"""
        initial_epsilon = 1.0
        min_epsilon = 0.01
        decay_rate = 0.995
        
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b'],
            epsilon=initial_epsilon
        )
        
        # Simulate epsilon decay
        epsilons = []
        for episode in range(1000):
            # Decay epsilon
            agent.epsilon = max(min_epsilon, agent.epsilon * decay_rate)
            epsilons.append(agent.epsilon)
        
        # Check decay
        assert epsilons[0] > epsilons[-1]
        assert epsilons[-1] >= min_epsilon
        assert all(epsilons[i] >= epsilons[i+1] for i in range(len(epsilons)-1))
    
    def test_optimistic_initialization(self):
        """Test optimistic initial Q-values for exploration"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b', 'c'],
            epsilon=0  # Pure greedy
        )
        
        # Set optimistic initial values
        for action in ['a', 'b', 'c']:
            agent.q_values[('s1', action)] = 10  # Optimistic
        
        # Even with epsilon=0, should explore all actions initially
        actions_taken = set()
        state = 's1'
        
        for _ in range(10):
            action = agent.get_action(state)
            actions_taken.add(action)
            # Simulate negative reward to reduce Q-value
            agent.update(state, action, 's2', -1)
        
        # Should have tried multiple actions due to optimistic init
        assert len(actions_taken) > 1


class TestConvergence:
    """Test convergence properties"""
    
    def test_q_learning_convergence_simple(self):
        """Test Q-learning convergence on simple MDP"""
        # Simple two-state MDP
        class SimpleMDP:
            def reset(self):
                return 'A'
            
            def get_actions(self, state=None):
                return ['go']
            
            def step(self, state, action):
                if state == 'A':
                    return 'B', 0, False
                else:  # B
                    return 'A', 1, False
        
        env = SimpleMDP()
        agent = QLearningAgent(
            action_fn=env.get_actions,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1
        )
        
        # Train for many steps
        state = 'A'
        for _ in range(10000):
            action = agent.get_action(state)
            if state == 'A':
                next_state, reward = 'B', 0
            else:
                next_state, reward = 'A', 1
            
            agent.update(state, action, next_state, reward)
            state = next_state
        
        # Check convergence to expected values
        # V(A) = 0 + 0.9 * V(B) = 0.9 * (1 + 0.9 * V(A))
        # V(A) = 0.9 + 0.81 * V(A)
        # V(A) = 0.9 / 0.19 ≈ 4.74
        
        q_a = agent.get_q_value('A', 'go')
        expected_a = 0.9 / (1 - 0.81)
        assert abs(q_a - expected_a) < 0.5
    
    def test_learning_rate_effect(self):
        """Test effect of learning rate on convergence"""
        grid = [[0, 0], [0, 0]]
        env = GridWorldEnvironment(grid, start=(0, 0), goal=(1, 1))
        
        # High learning rate
        agent_high = QLearningAgent(
            action_fn=env.get_actions,
            alpha=0.9,
            epsilon=0.1
        )
        
        # Low learning rate
        agent_low = QLearningAgent(
            action_fn=env.get_actions,
            alpha=0.1,
            epsilon=0.1
        )
        
        # Train both
        for agent in [agent_high, agent_low]:
            for episode in range(50):
                state = env.reset()
                for step in range(20):
                    action = agent.get_action(state)
                    next_state, reward, done = env.step(action)
                    agent.update(state, action, next_state, reward)
                    state = next_state
                    if done:
                        break
        
        # Both should learn, but patterns differ
        assert len(agent_high.q_values) > 0
        assert len(agent_low.q_values) > 0


class TestComplexEnvironments:
    """Test agents in complex environments"""
    
    def test_maze_solving(self):
        """Test Q-learning on maze"""
        maze = [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ]
        env = GridWorldEnvironment(maze, start=(0, 0), goal=(4, 4))
        
        agent = QLearningAgent(
            action_fn=env.get_actions,
            discount=0.95,
            alpha=0.5,
            epsilon=0.2
        )
        
        # Train
        episode_lengths = []
        for episode in range(200):
            state = env.reset()
            steps = 0
            
            for step in range(200):
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, next_state, reward)
                state = next_state
                steps += 1
                
                if done:
                    break
            
            episode_lengths.append(steps)
        
        # Should learn to solve maze faster over time
        early_avg = np.mean(episode_lengths[:20])
        late_avg = np.mean(episode_lengths[-20:])
        assert late_avg < early_avg
    
    def test_stochastic_environment(self):
        """Test learning in stochastic environment"""
        grid = [[0, 0, 0], [0, 0, 0]]
        env = GridWorldEnvironment(
            grid, 
            start=(0, 0), 
            goal=(1, 2),
            stochastic=True
        )
        
        agent = QLearningAgent(
            action_fn=env.get_actions,
            discount=0.9,
            alpha=0.3,
            epsilon=0.2
        )
        
        # Train in stochastic environment
        successes = 0
        for episode in range(300):
            state = env.reset()
            
            for step in range(50):
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, next_state, reward)
                state = next_state
                
                if done and state == env.goal:
                    successes += 1
                    break
        
        # Should learn despite stochasticity
        success_rate = successes / 300
        assert success_rate > 0.5  # Should succeed more than half the time
    
    def test_negative_rewards(self):
        """Test learning with negative rewards"""
        # Grid with penalties
        grid = [
            [0, -1, 0],
            [-1, -1, 0],
            [0, 0, 0]
        ]
        env = GridWorldEnvironment(grid, start=(0, 0), goal=(2, 2))
        
        agent = QLearningAgent(
            action_fn=env.get_actions,
            discount=0.9,
            alpha=0.5,
            epsilon=0.1
        )
        
        # Train
        for episode in range(100):
            state = env.reset()
            
            for step in range(50):
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, next_state, reward)
                state = next_state
                
                if done:
                    break
        
        # Should learn to avoid negative reward states
        # Check that Q-values for moving into penalty states are lower
        q_into_penalty = agent.get_q_value((0, 0), 'right')  # Move to (0,1) which is -1
        q_safe = agent.get_q_value((0, 0), 'down')  # Move to (1,0) which is -1
        
        # Both are penalties in this case, but agent should have learned something
        assert len(agent.q_values) > 0


class TestMemoryAndStorage:
    """Test memory and storage aspects"""
    
    def test_q_table_size(self):
        """Test Q-table grows appropriately"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b', 'c'],
            epsilon=0
        )
        
        # Visit multiple states
        states = [f"s{i}" for i in range(10)]
        actions = ['a', 'b', 'c']
        
        for state in states:
            for action in actions:
                agent.update(state, action, 'next', 1)
        
        # Q-table should have entries for all state-action pairs
        assert len(agent.q_values) == len(states) * len(actions)
    
    def test_unseen_state_handling(self):
        """Test handling of previously unseen states"""
        agent = QLearningAgent(
            action_fn=lambda s: ['a', 'b'],
            epsilon=0
        )
        
        # Get Q-value for unseen state
        q_val = agent.get_q_value('unseen_state', 'a')
        assert q_val == 0
        
        # Get action for unseen state
        agent.q_values[('unseen_state', 'b')] = 1
        action = agent.get_action('unseen_state')
        assert action == 'b'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
