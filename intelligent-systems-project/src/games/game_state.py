"""
Game State Representation

Abstract base class for game states and example implementations.
Based on CS 5368 game playing framework with states, actions, and utilities.
"""

class Game:
    """
    Abstract base class for two-player games.
    Defines interface that minimax and alpha-beta algorithms expect.
    """
    
    def actions(self, state):
        """Return list of legal actions in given state."""
        raise NotImplementedError
    
    def result(self, state, action):
        """Return state that results from making action in state."""
        raise NotImplementedError
    
    def terminal_test(self, state):
        """Return True if state is terminal (game over)."""
        raise NotImplementedError
    
    def utility(self, state, player):
        """Return utility value for player in terminal state."""
        raise NotImplementedError
    
    def to_move(self, state):
        """Return player whose turn it is in given state."""
        raise NotImplementedError
    
    def display(self, state):
        """Print or display the state."""
        print(state)

class TicTacToe(Game):
    """
    Tic-Tac-Toe game implementation.
    Demonstrates complete game interface for 3x3 grid.
    """
    
    def __init__(self):
        """Initialize empty 3x3 board."""
        self.initial = GameState(
            board=[[' ' for _ in range(3)] for _ in range(3)],
            to_move='X'
        )
    
    def actions(self, state):
        """Return list of (row, col) positions for empty squares."""
        actions = []
        for i in range(3):
            for j in range(3):
                if state.board[i][j] == ' ':
                    actions.append((i, j))
        return actions
    
    def result(self, state, action):
        """Place current player's mark at action position."""
        if action not in self.actions(state):
            raise ValueError(f"Illegal action {action}")
        
        new_board = [row[:] for row in state.board]  # Deep copy
        row, col = action
        new_board[row][col] = state.to_move
        
        next_player = 'O' if state.to_move == 'X' else 'X'
        return GameState(board=new_board, to_move=next_player)
    
    def terminal_test(self, state):
        """Check if game is over (win or draw)."""
        return self.utility(state, 'X') != 0 or len(self.actions(state)) == 0
    
    def utility(self, state, player):
        """Return +1 if player wins, -1 if loses, 0 for draw/ongoing."""
        lines = (
            # Rows
            [state.board[0][0], state.board[0][1], state.board[0][2]],
            [state.board[1][0], state.board[1][1], state.board[1][2]], 
            [state.board[2][0], state.board[2][1], state.board[2][2]],
            # Columns
            [state.board[0][0], state.board[1][0], state.board[2][0]],
            [state.board[0][1], state.board[1][1], state.board[2][1]],
            [state.board[0][2], state.board[1][2], state.board[2][2]],
            # Diagonals
            [state.board[0][0], state.board[1][1], state.board[2][2]],
            [state.board[0][2], state.board[1][1], state.board[2][0]]
        )
        
        for line in lines:
            if line == ['X', 'X', 'X']:
                return 1 if player == 'X' else -1
            elif line == ['O', 'O', 'O']:
                return 1 if player == 'O' else -1
        
        return 0  # No winner yet
    
    def to_move(self, state):
        """Return whose turn it is."""
        return state.to_move
    
    def display(self, state):
        """Display the board."""
        for i, row in enumerate(state.board):
            print(' | '.join(row))
            if i < 2:
                print('---------')

class GameState:
    """
    Generic game state representation.
    Stores board configuration and current player.
    """
    
    def __init__(self, board=None, to_move=None, **kwargs):
        """Initialize game state with board and current player."""
        self.board = board
        self.to_move = to_move
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        """String representation for debugging."""
        return f"GameState(to_move={self.to_move})"
