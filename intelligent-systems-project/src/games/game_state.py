"""Core game abstractions used by the adversarial search modules.

Phase 1 formalises the interface between generic game-playing algorithms
(minimax, alpha-beta, expectimax) and concrete games.  The :class:`GameState`
base class mirrors the interface used in the lectures: it provides legal moves,
creates successor states, tests for termination, and reports the utility of a
terminal position for a specified agent.  :class:`GameAgent` provides the shared
book-keeping for search-based agents.

A compact Tic-Tac-Toe implementation is included for demonstrations and unit
tests.  It reuses the abstractions above and is deliberately straightforward so
students can focus on the algorithms rather than domain intricacies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Move = Tuple[int, int]


class GameState:
    """Abstract representation of a deterministic, perfect-information game.

    Subclasses must override all four interface methods; they intentionally do
    not provide default behaviour to prevent silent errors when an algorithm
    relies on a missing override.
    """

    def get_legal_actions(self, agent_index: int = 0) -> Sequence:
        raise NotImplementedError

    def generate_successor(self, agent_index: int, action) -> "GameState":
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError

    def get_utility(self, agent_index: int = 0) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        attrs = []
        if hasattr(self, "to_move"):
            attrs.append(f"to_move={getattr(self, 'to_move')!r}")
        if hasattr(self, "last_move") and getattr(self, "last_move") is not None:
            attrs.append(f"last_move={getattr(self, 'last_move')!r}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"


class GameAgent:
    """Base class for adversarial search agents."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.nodes_explored = 0

    def reset_statistics(self) -> None:
        """Reset counters recorded during the previous search."""
        self.nodes_explored = 0

    def get_action(self, game_state: GameState):
        raise NotImplementedError


class Game:
    """Abstract game definition used by the demo applications."""

    def actions(self, state: GameState) -> Sequence:
        raise NotImplementedError

    def result(self, state: GameState, action) -> GameState:
        raise NotImplementedError

    def terminal_test(self, state: GameState) -> bool:
        raise NotImplementedError

    def utility(self, state: GameState, player: int) -> float:
        raise NotImplementedError

    def to_move(self, state: GameState):
        raise NotImplementedError

    def display(self, state: GameState) -> None:  # pragma: no cover - console output
        print(state)


@dataclass(frozen=True)
class TicTacToeState(GameState):
    """Immutable 3x3 Tic-Tac-Toe board state."""

    board: Tuple[Tuple[str, str, str], Tuple[str, str, str], Tuple[str, str, str]]
    to_move: str = "X"
    last_move: Optional[Move] = None

    def __post_init__(self) -> None:
        for row in self.board:
            if len(row) != 3:
                raise ValueError("TicTacToe board must be 3x3")
            for cell in row:
                if cell not in {"X", "O", " "}:
                    raise ValueError(f"Invalid cell value {cell!r}")
        if self.to_move not in {"X", "O"}:
            raise ValueError("to_move must be 'X' or 'O'")

    def get_legal_actions(self, agent_index: int = 0) -> List[Move]:
        if self.is_terminal():
            return []
        actions: List[Move] = []
        for r, row in enumerate(self.board):
            for c, cell in enumerate(row):
                if cell == " ":
                    actions.append((r, c))
        return actions

    def generate_successor(self, agent_index: int, action: Move) -> "TicTacToeState":
        if self.is_terminal():
            raise ValueError("Cannot expand a terminal state")
        row, col = action
        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError(f"Action {action} outside of board")
        if self.board[row][col] != " ":
            raise ValueError(f"Cell {action} already occupied")

        expected_symbol = self.to_move
        symbol = "X" if agent_index == 0 else "O"
        if symbol != expected_symbol:
            raise ValueError(
                f"Agent index {agent_index} cannot move: expected player {expected_symbol}"
            )

        next_board = [list(row) for row in self.board]
        next_board[row][col] = symbol
        next_symbol = "O" if symbol == "X" else "X"
        return TicTacToeState(
            board=tuple(tuple(row) for row in next_board),
            to_move=next_symbol,
            last_move=action,
        )

    def is_terminal(self) -> bool:
        return self._winner() is not None or not any(" " in row for row in self.board)

    def get_utility(self, agent_index: int = 0) -> float:
        winner = self._winner()
        if winner is None:
            return 0.0
        if agent_index not in (0, 1):
            raise ValueError("agent_index must be 0 or 1 for TicTacToe")
        agent_symbol = "X" if agent_index == 0 else "O"
        if winner == agent_symbol:
            return 1.0
        return -1.0

    def _winner(self) -> Optional[str]:
        lines = []
        # Rows and columns
        for i in range(3):
            lines.append(self.board[i])
            lines.append((self.board[0][i], self.board[1][i], self.board[2][i]))
        # Diagonals
        lines.append((self.board[0][0], self.board[1][1], self.board[2][2]))
        lines.append((self.board[0][2], self.board[1][1], self.board[2][0]))
        for line in lines:
            if line[0] != " " and line[0] == line[1] == line[2]:
                return line[0]
        return None


class TicTacToe(Game):
    """Convenience wrapper exposing the Tic-Tac-Toe domain as a :class:`Game`."""

    def __init__(self) -> None:
        empty_row = (" ", " ", " ")
        self.initial = TicTacToeState(board=(empty_row, empty_row, empty_row))

    def actions(self, state: TicTacToeState) -> Sequence[Move]:
        return state.get_legal_actions(0 if state.to_move == "X" else 1)

    def result(self, state: TicTacToeState, action: Move) -> TicTacToeState:
        player_index = 0 if state.to_move == "X" else 1
        return state.generate_successor(player_index, action)

    def terminal_test(self, state: TicTacToeState) -> bool:
        return state.is_terminal()

    def utility(self, state: TicTacToeState, player: int) -> float:
        return state.get_utility(player)

    def to_move(self, state: TicTacToeState) -> int:
        return 0 if state.to_move == "X" else 1

    def display(self, state: TicTacToeState) -> None:  # pragma: no cover - console output
        for i, row in enumerate(state.board):
            print(" | ".join(row))
            if i < 2:
                print("---------")
