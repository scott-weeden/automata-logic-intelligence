"""Entry points for the example applications included with the project."""

from .pathfinding_demo import run_pathfinding_demo
from .game_ai_demo import play_optimal_tictactoe
from .mdp_robot_navigation import analyse_robot_navigation
from .reinforcement_learning_trader import train_trading_agent
from .medical_diagnosis_bayes import diagnose_patient

__all__ = [
    "run_pathfinding_demo",
    "play_optimal_tictactoe",
    "analyse_robot_navigation",
    "train_trading_agent",
    "diagnose_patient",
]
