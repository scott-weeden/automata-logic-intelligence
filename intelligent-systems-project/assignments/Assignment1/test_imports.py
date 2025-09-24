#!/usr/bin/env python3
"""Test script to verify the compiled intelligent-systems module can be imported."""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test importing all modules and key classes."""

    print("Testing module imports...")

    # Test basic module imports
    try:
        import search
        import games
        import mdp
        import learning
        print("✓ Basic module imports successful")
    except ImportError as e:
        print(f"✗ Basic module import failed: {e}")
        return False

    # Test specific class imports
    try:
        from search.algorithms import AStarSearch, BreadthFirstSearch, DepthFirstSearch
        print("✓ Search algorithms imported")

        from games.minimax import MinimaxAgent, AlphaBetaAgent
        print("✓ Game algorithms imported")

        from learning.qlearning import QLearningAgent, SARSAAgent
        print("✓ Learning algorithms imported")

    except ImportError as e:
        print(f"✗ Specific class import failed: {e}")
        return False

    # Test instantiation
    try:
        search_agent = AStarSearch()
        minimax_agent = MinimaxAgent(depth=3)
        qlearning_agent = QLearningAgent()
        print("✓ Class instantiation successful")

    except Exception as e:
        print(f"✗ Class instantiation failed: {e}")
        return False

    print("All imports and instantiations successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)