"""Game-playing demonstration using the Tic-Tac-Toe domain."""

from __future__ import annotations

from typing import List, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from games import MinimaxAgent, TicTacToe, TicTacToeState

Move = Tuple[int, int]
Turn = Tuple[int, Move]


def _render_board(state: TicTacToeState) -> str:
    rows = []
    for row in state.board:
        rows.append(" | ".join(row))
    return "\n---------\n".join(rows)


def get_human_move(state: TicTacToeState) -> Move:
    """Get move from human player with input validation."""
    while True:
        try:
            print("Enter your move as 'row,col' (0-2): ", end="")
            move_input = input().strip()
            row, col = map(int, move_input.split(','))
            
            if 0 <= row <= 2 and 0 <= col <= 2:
                if state.board[row][col] == ' ':
                    return (row, col)
                else:
                    print("That position is already taken!")
            else:
                print("Invalid position! Use 0-2 for row and column.")
        except (ValueError, IndexError):
            print("Invalid input! Use format: row,col (e.g., 1,2)")


def play_human_vs_ai() -> None:
    """Human vs AI interactive game."""
    game = TicTacToe()
    ai_agent = MinimaxAgent(index=1, depth=9)  # AI plays as O
    
    print("=== Human vs AI Tic-Tac-Toe ===")
    print("You are X, AI is O")
    print("Enter moves as row,col (0-2)")
    print()
    
    state = game.initial
    print("Initial board:")
    print(_render_board(state))
    print()
    
    while not game.terminal_test(state):
        current_player = game.to_move(state)
        
        if current_player == 0:  # Human turn (X)
            print("Your turn (X):")
            action = get_human_move(state)
            print(f"You played: {action}")
        else:  # AI turn (O)
            print("AI thinking...")
            action = ai_agent.get_action(state)
            print(f"AI played: {action}")
        
        state = state.generate_successor(current_player, action)
        print(_render_board(state))
        print()
    
    # Game over
    utility_x = game.utility(state, 0)
    if utility_x > 0:
        print("🎉 You win!")
    elif utility_x < 0:
        print("🤖 AI wins!")
    else:
        print("🤝 It's a draw!")


def play_optimal_tictactoe(show_output: bool = True) -> Tuple[TicTacToeState, List[Turn]]:
    """AI vs AI demonstration game."""
    game = TicTacToe()
    agent_x = MinimaxAgent(index=0, depth=9)
    agent_o = MinimaxAgent(index=1, depth=9)

    state = game.initial
    history: List[Turn] = []
    agents = {0: agent_x, 1: agent_o}

    if show_output:
        print("Initial board:\n" + _render_board(state) + "\n")

    while not game.terminal_test(state):
        current_index = game.to_move(state)
        agent = agents[current_index]
        action = agent.get_action(state)
        if action is None:
            break
        history.append((current_index, action))
        state = state.generate_successor(current_index, action)
        if show_output:
            print(f"Player {'X' if current_index == 0 else 'O'} -> {action}")
            print(_render_board(state))
            print()

    if show_output:
        utility_x = game.utility(state, 0)
        outcome = "draw" if utility_x == 0 else ("X wins" if utility_x > 0 else "O wins")
        print("Game over:", outcome)

    return state, history


if __name__ == "__main__":  # pragma: no cover - manual demo
    print("Choose game mode:")
    print("1. Human vs AI")
    print("2. AI vs AI demo")
    
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            play_human_vs_ai()
            break
        elif choice == "2":
            play_optimal_tictactoe(show_output=True)
            break
        else:
            print("Invalid choice! Enter 1 or 2.")
