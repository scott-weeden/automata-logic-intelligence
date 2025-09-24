"""
Search Problem Formulation

Defines abstract search problem interface based on CS 5368 Week 1-2 material.
A search problem consists of:
- Initial state
- Actions available in each state  
- Transition model (result of actions)
- Goal test
- Path cost function
"""

class SearchProblem:
    """
    Abstract base class for search problems.
    Subclasses must implement actions, result, goal_test methods.
    """
    
    def __init__(self, initial_state):
        """Initialize with starting state."""
        self.initial_state = initial_state
    
    def actions(self, state):
        """
        Return list of actions available in given state.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement actions method")
    
    def result(self, state, action):
        """
        Return state that results from executing action in state.
        Also called transition model.
        """
        raise NotImplementedError("Subclass must implement result method")
    
    def goal_test(self, state):
        """
        Return True if state is a goal state.
        Default implementation checks against self.goal if it exists.
        """
        if hasattr(self, 'goal'):
            return state == self.goal
        raise NotImplementedError("Subclass must implement goal_test method")
    
    def path_cost(self, cost_so_far, state1, action, state2):
        """
        Return cost of path from start to state2 via state1 and action.
        Default implementation assumes cost 1 for every step.
        """
        return cost_so_far + 1
    
    def value(self, state):
        """
        Return value of state for optimization problems.
        Used in hill-climbing and simulated annealing.
        """
        raise NotImplementedError("Subclass must implement value method")

class GridSearchProblem(SearchProblem):
    """
    Example search problem: pathfinding on 2D grid.
    Demonstrates problem formulation for navigation tasks.
    """
    
    def __init__(self, grid, start, goal):
        """
        Initialize grid search problem.
        grid: 2D array where 0=free, 1=obstacle
        start: (row, col) starting position
        goal: (row, col) goal position
        """
        super().__init__(start)
        self.grid = grid
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
    
    def actions(self, state):
        """Return valid moves from current position."""
        row, col = state
        actions = []
        
        # Check four directions: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.rows and 
                0 <= new_col < self.cols and 
                self.grid[new_row][new_col] == 0):
                actions.append((dr, dc))
        
        return actions
    
    def result(self, state, action):
        """Apply action to get new state."""
        row, col = state
        dr, dc = action
        return (row + dr, col + dc)
    
    def path_cost(self, cost_so_far, state1, action, state2):
        """Uniform cost for grid movement."""
        return cost_so_far + 1
