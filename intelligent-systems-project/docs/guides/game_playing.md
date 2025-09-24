# Game Playing

## Overview

Adversarial search algorithms for two-player, zero-sum, perfect information games. Implements minimax, alpha-beta pruning, and expectimax with significant performance optimizations.

## Game Interface

```python
class Game:
    def actions(self, state): pass           # Legal actions
    def result(self, state, action): pass    # Apply action
    def terminal_test(self, state): pass     # Game over?
    def utility(self, state, player): pass   # Final score
    def to_move(self, state): pass          # Current player
```

## Algorithms

### Minimax
- **Strategy**: Maximize own utility, minimize opponent's
- **Complete**: Yes (finite games)
- **Optimal**: Yes (against optimal opponent)
- **Time**: O(b^m), **Space**: O(bm)

```python
from games.agents import MinimaxAgent

agent = MinimaxAgent(depth=4)
action = agent.get_action(game, state)
```

### Alpha-Beta Pruning
- **Strategy**: Minimax with branch elimination
- **Performance**: O(b^(m/2)) best case vs O(b^m)
- **Pruning**: 1,078 vs 18,729 nodes (94% reduction)

```python
from games.agents import AlphaBetaAgent

agent = AlphaBetaAgent(depth=6)
action = agent.get_action(game, state)
print(f"Nodes expanded: {agent.nodes_expanded}")
```

### Expectimax
- **Strategy**: Handle chance nodes with expected values
- **Use case**: Games with randomness or uncertain opponents

```python
from games.agents import ExpectimaxAgent

agent = ExpectimaxAgent(depth=4)
action = agent.get_action(game, state)
```

## TicTacToe Implementation

```python
from games.tictactoe import TicTacToe

game = TicTacToe()
state = game.initial

# Play with minimax
minimax = MinimaxAgent(depth=9)
while not game.terminal_test(state):
    action = minimax.get_action(game, state)
    state = game.result(state, action)
    game.display(state)
```

## Performance Analysis

### Node Expansion Comparison
- **Minimax**: 18,729 nodes
- **Alpha-Beta**: 1,078 nodes  
- **Improvement**: 94% reduction

### Depth Capabilities
- **Minimax**: Depth 4-5 practical
- **Alpha-Beta**: Depth 8-10 practical
- **Effective doubling** of search depth

## Evaluation Functions

### Material Count
```python
def material_evaluation(state, player):
    return sum(piece_values[piece] for piece in player_pieces)
```

### Positional Features
```python
def positional_evaluation(state, player):
    score = material_score(state, player)
    score += mobility_score(state, player)
    score += center_control(state, player)
    return score
```

## Optimization Techniques

### Move Ordering
```python
def order_moves(game, state):
    moves = game.actions(state)
    # Order by: captures, checks, center moves
    return sorted(moves, key=move_priority, reverse=True)
```

### Transposition Tables
```python
class TranspositionTable:
    def __init__(self):
        self.table = {}
    
    def lookup(self, state, depth):
        key = hash(state)
        return self.table.get((key, depth))
    
    def store(self, state, depth, value):
        key = hash(state)
        self.table[(key, depth)] = value
```

## Game Tree Properties

### Branching Factor (b)
- **TicTacToe**: ~5 average
- **Chess**: ~35 average
- **Go**: ~250 average

### Game Length (m)
- **TicTacToe**: 9 moves max
- **Chess**: ~40 moves average
- **Checkers**: ~70 moves average

## Example: Perfect TicTacToe

```python
from games.tictactoe import TicTacToe
from games.agents import AlphaBetaAgent

game = TicTacToe()
agent = AlphaBetaAgent(depth=9)  # Perfect play

def play_perfect_game():
    state = game.initial
    while not game.terminal_test(state):
        if game.to_move(state) == 'X':
            action = agent.get_action(game, state)
        else:
            action = agent.get_action(game, state)
        state = game.result(state, action)
        game.display(state)
    
    return game.utility(state, 'X')

# Result: Always draw with perfect play
result = play_perfect_game()  # Returns 0 (draw)
```

## Key Insights

1. **Alpha-beta pruning** provides massive performance gains
2. **Move ordering** critical for pruning effectiveness  
3. **Evaluation functions** determine playing strength
4. **Depth vs breadth** tradeoff in search
5. **Perfect play** achievable for simple games

## Advanced Topics

- **Iterative deepening** for time management
- **Quiescence search** for tactical positions
- **Opening books** and **endgame tables**
- **Monte Carlo Tree Search** for complex games
- **Neural network** evaluation functions
