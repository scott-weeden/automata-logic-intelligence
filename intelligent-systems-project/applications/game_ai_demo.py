"""Game-playing demonstration using the Tic-Tac-Toe domain."""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.games import MinimaxAgent, TicTacToe, TicTacToeState

Move = Tuple[int, int]
Turn = Tuple[int, Move]


def _render_board(state: TicTacToeState) -> str:
    rows = []
    for row in state.board:
        rows.append(" | ".join(row))
    return "\n---------\n".join(rows)


def _prompt_human_move(state: TicTacToeState) -> Move:
    legal = set(state.get_legal_actions())
    prompt = (
        "Enter your move as row,col (0-based indexing). For example, '1,2' places "
        "a mark on the second row, third column."
    )
    print(prompt)
    while True:
        raw = input("Your move (or 'q' to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt("Game ended by user")
        separators = [",", " "]
        for sep in separators:
            if sep in raw:
                parts = raw.split(sep)
                break
        else:
            parts = [raw]
        if len(parts) != 2:
            print("Please provide two numbers separated by a comma.")
            continue
        try:
            row = int(parts[0])
            col = int(parts[1])
        except ValueError:
            print("Row and column must be integers between 0 and 2.")
            continue
        move = (row, col)
        if move not in legal:
            print(f"Illegal move {move}. Available moves: {sorted(legal)}")
            continue
        return move


def play_optimal_tictactoe(show_output: bool = True) -> Tuple[TicTacToeState, List[Turn]]:
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


def play_human_vs_ai(human_symbol: str = "X", show_output: bool = True) -> TicTacToeState:
    human_symbol = human_symbol.upper()
    if human_symbol not in {"X", "O"}:
        raise ValueError("human_symbol must be 'X' or 'O'")

    game = TicTacToe()
    state = game.initial
    ai_index = 0 if human_symbol == "O" else 1
    human_index = 1 - ai_index
    ai_agent = MinimaxAgent(index=ai_index, depth=9)

    if show_output:
        print("You are playing as", human_symbol)
        print("Initial board:\n" + _render_board(state) + "\n")

    try:
        while not game.terminal_test(state):
            current_index = game.to_move(state)
            if current_index == human_index:
                if show_output:
                    print("Your turn (player", human_symbol, ")")
                action = _prompt_human_move(state)
                if show_output:
                    print(f"You -> {action}")
            else:
                action = ai_agent.get_action(state)
                if show_output:
                    print(f"AI ({'X' if ai_index == 0 else 'O'}) -> {action}")
            state = state.generate_successor(current_index, action)
            if show_output:
                print(_render_board(state))
                print()
    except KeyboardInterrupt:
        if show_output:
            print("Game aborted by human player.")
        return state

    if show_output:
        utility = game.utility(state, human_index)
        if utility > 0:
            print("You win! Congratulations!")
        elif utility < 0:
            print("The AI wins. Better luck next time!")
        else:
            print("It's a draw.")
    return state


if __name__ == "__main__":  # pragma: no cover - manual demo
    print("Select mode:\n 1) Human vs AI\n 2) AI vs AI (perfect play)")
    choice = input("Choice [1/2]: ").strip() or "1"
    if choice.startswith("2"):
        play_optimal_tictactoe(show_output=True)
    else:
        symbol = input("Play as X or O? [X]: ").strip().upper() or "X"
        try:
            play_human_vs_ai(human_symbol=symbol, show_output=True)
        except ValueError as exc:
            print(exc)
