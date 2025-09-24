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
    Subclasses must implement get_start_state, is_goal_state, get_successors methods.
    """
    
    def get_start_state(self):
        """Return the start state for the search problem."""
        raise NotImplementedError("Subclass must implement get_start_state method")
    
    def is_goal_state(self, state):
        """Return True if state is a goal state."""
        raise NotImplementedError("Subclass must implement is_goal_state method")
    
    def get_successors(self, state):
        """
        Return list of (successor, action, stepCost) tuples.
        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor.
        """
        raise NotImplementedError("Subclass must implement get_successors method")
    
    def get_cost_of_actions(self, actions):
        """
        Return the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        return len(actions)

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
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
    
    def get_start_state(self):
        """Return starting position."""
        return self.start
    
    def is_goal_state(self, state):
        """Return True if state is the goal."""
        return state == self.goal
    
    def get_successors(self, state):
        """Return valid moves from current position."""
        successors = []
        row, col = state
        
        # Check four directions: up, down, left, right
        moves = [(-1, 0, 'North'), (1, 0, 'South'), (0, -1, 'West'), (0, 1, 'East')]
        
        for dr, dc, action in moves:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.rows and 
                0 <= new_col < self.cols and 
                self.grid[new_row][new_col] != '#' and
                self.grid[new_row][new_col] != 1):
                successors.append(((new_row, new_col), action, 1))
        
        return successors
