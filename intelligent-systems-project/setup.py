"""Setup script for the Intelligent Systems teaching library."""

from __future__ import annotations

from pathlib import Path
from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")

packages = find_packages(where="src")
if "applications" not in packages:
    packages.append("applications")

setup(
    name="intelligent-systems",
    version="1.0.0",
    description="Algorithms and demos for search, games, MDPs, and reinforcement learning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/example/intelligent-systems-project",
    author="CS 5368 Teaching Team",
    license="MIT",
    packages=packages,
    package_dir={"": "src", "applications": "applications"},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "coverage>=7.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "networkx>=2.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "is-pathfinding-demo=applications.pathfinding_demo:run_pathfinding_demo",
            "is-game-demo=applications.game_ai_demo:play_optimal_tictactoe",
            "is-game-human-demo=applications.game_ai_demo:play_human_vs_ai",
            "is-mdp-demo=applications.mdp_robot_navigation:analyse_robot_navigation",
            "is-rl-trader=applications.reinforcement_learning_trader:train_trading_agent",
            "is-medical-bayes=applications.medical_diagnosis_bayes:diagnose_patient",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Education",
    ],
    project_urls={
        "API Reference": "https://github.com/example/intelligent-systems-project/blob/main/docs/api_reference.md",
        "Issue Tracker": "https://github.com/example/intelligent-systems-project/issues",
    },
)
