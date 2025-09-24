# Markov Decision Processes & Reinforcement Learning

## Overview

MDPs model sequential decision making under uncertainty. Reinforcement learning algorithms learn optimal policies through interaction with the environment.

## Markov Decision Process (MDP)

### Components
- **States** S: Set of possible states
- **Actions** A(s): Available actions in state s  
- **Transition Model** P(s'|s,a): Probability of reaching s' from s via a
- **Reward Function** R(s): Immediate reward for being in state s
- **Discount Factor** γ ∈ [0,1]: Future reward preference

### Markov Property
```
P(S_{t+1}|S_t, A_t, S_{t-1}, ..., S_0) = P(S_{t+1}|S_t, A_t)
```

## Value Functions

### State Value Function
```python
def state_value(state, policy, mdp):
    """V^π(s) = E[Σ γ^t R(S_t) | S_0=s, π]"""
    return expected_discounted_reward(state, policy, mdp)
```

### Action Value Function  
```python
def action_value(state, action, policy, mdp):
    """Q^π(s,a) = E[Σ γ^t R(S_t) | S_0=s, A_0=a, π]"""
    return expected_reward_after_action(state, action, policy, mdp)
```

## Bellman Equations

### Bellman Equation for V*
```
V*(s) = R(s) + γ max_a Σ P(s'|s,a) V*(s')
```

### Bellman Equation for Q*
```
Q*(s,a) = R(s) + γ Σ P(s'|s,a) max_a' Q*(s',a')
```

## Dynamic Programming

### Value Iteration
```python
from mdp.algorithms import ValueIteration

vi = ValueIteration(mdp, epsilon=0.01)
utilities = vi.solve()
policy = vi.extract_policy(utilities)
```

**Algorithm:**
1. Initialize V(s) arbitrarily
2. Repeat until convergence:
   - V_{k+1}(s) = R(s) + γ max_a Σ P(s'|s,a) V_k(s')
3. Extract policy: π(s) = argmax_a Σ P(s'|s,a) V(s')

### Policy Iteration
```python
from mdp.algorithms import PolicyIteration

pi = PolicyIteration(mdp)
policy = pi.solve()
```

**Algorithm:**
1. Initialize policy π arbitrarily
2. Repeat:
   - **Policy Evaluation**: Solve V^π(s) = R(s) + γ Σ P(s'|s,π(s)) V^π(s')
   - **Policy Improvement**: π'(s) = argmax_a Σ P(s'|s,a) V^π(s')
3. Stop when policy unchanged

## Reinforcement Learning

### Temporal Difference Learning
```python
from learning.algorithms import TDLearning

td = TDLearning(alpha=0.1, gamma=0.9)
for episode in episodes:
    td.update(state, reward, next_state)
```

**Update Rule:**
```
V(s) ← V(s) + α[r + γV(s') - V(s)]
```

### Q-Learning (Off-Policy)
```python
from learning.algorithms import QLearning

q_agent = QLearning(alpha=0.1, gamma=0.9, epsilon=0.1)
for episode in episodes:
    action = q_agent.get_action(state)
    q_agent.update(state, action, reward, next_state)
```

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

### SARSA (On-Policy)
```python
from learning.algorithms import SARSA

sarsa = SARSA(alpha=0.1, gamma=0.9, epsilon=0.1)
action = sarsa.get_action(state)
# Take action, observe reward and next_state
next_action = sarsa.get_action(next_state)
sarsa.update(state, action, reward, next_state, next_action)
```

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

## Exploration Strategies

### ε-Greedy
```python
def epsilon_greedy(q_values, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(actions)  # Explore
    else:
        return argmax(q_values)        # Exploit
```

### Softmax (Boltzmann)
```python
def softmax_action(q_values, temperature=1.0):
    probs = np.exp(q_values / temperature)
    probs /= np.sum(probs)
    return np.random.choice(actions, p=probs)
```

## Grid World Example

```python
from mdp.gridworld import GridWorld
from mdp.algorithms import ValueIteration

# Create 4x3 grid with rewards
grid = GridWorld(
    grid=[[0, 0, 0, 1],    # +1 reward at (0,3)
          [0, None, 0, -1], # -1 reward at (1,3), wall at (1,1)  
          [0, 0, 0, 0]],
    living_penalty=-0.04,
    noise=0.2  # 20% chance of perpendicular movement
)

# Solve with value iteration
vi = ValueIteration(grid, gamma=0.9)
utilities = vi.solve()
policy = vi.extract_policy(utilities)

print("Optimal Policy:")
for state in grid.states:
    print(f"{state}: {policy[state]}")
```

## Performance Comparison

| Algorithm | Type | Convergence | Memory | Best Use |
|-----------|------|-------------|---------|----------|
| Value Iteration | DP | O(S²A/ε) | O(S) | Known MDP |
| Policy Iteration | DP | O(S³A) | O(S) | Few actions |
| Q-Learning | RL | Guaranteed* | O(SA) | Unknown MDP |
| SARSA | RL | Guaranteed* | O(SA) | Safe exploration |
| TD Learning | RL | Guaranteed* | O(S) | State values only |

*Under appropriate conditions

## Convergence Guarantees

### Q-Learning Convergence
Requires:
1. All state-action pairs visited infinitely often
2. Learning rate α_t satisfies: Σα_t = ∞, Σα_t² < ∞
3. Bounded rewards

### Value Iteration Convergence
- **Contraction mapping**: ||T V - T U|| ≤ γ||V - U||
- **Convergence rate**: Linear with factor γ
- **Error bound**: ||V_k - V*|| ≤ γ^k ||V_0 - V*||

## Advanced Topics

### Function Approximation
```python
class LinearQFunction:
    def __init__(self, features):
        self.weights = np.zeros(len(features))
        self.features = features
    
    def q_value(self, state, action):
        return np.dot(self.weights, self.features(state, action))
    
    def update(self, state, action, target):
        features = self.features(state, action)
        prediction = self.q_value(state, action)
        error = target - prediction
        self.weights += self.alpha * error * features
```

### Deep Q-Networks (DQN)
- Neural network approximation of Q-function
- Experience replay buffer
- Target network for stability
- Handles high-dimensional state spaces

## Key Insights

1. **MDPs** provide mathematical framework for sequential decisions
2. **Dynamic programming** solves known MDPs optimally
3. **Reinforcement learning** handles unknown environments
4. **Exploration vs exploitation** fundamental tradeoff
5. **Function approximation** enables large state spaces
6. **Convergence guarantees** require careful algorithm design

## Practical Applications

- **Game playing**: AlphaGo, poker bots
- **Robotics**: Navigation, manipulation
- **Finance**: Portfolio optimization, trading
- **Healthcare**: Treatment planning
- **Autonomous vehicles**: Path planning, control
