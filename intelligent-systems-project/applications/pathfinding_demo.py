"""
Pathfinding Demo Application

Interactive demonstration of search algorithms on grid worlds.
Shows practical application of BFS, DFS, UCS, and A* for navigation.
Based on CS 5368 search algorithm implementations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from search import (
    breadth_first_search, depth_first_search, uniform_cost_search, astar_search,
    GridSearchProblem, GridHeuristic
)
import time

def create_maze():
    """Create example maze for pathfinding demonstration."""
    # 0 = free space, 1 = obstacle
    maze = [
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ]
    return maze

def visualize_path(maze, path, start, goal):
    """Visualize maze with solution path."""
    # Create display grid
    display = []
    for row in maze:
        display.append(['.' if cell == 0 else '#' for cell in row])
    
    # Mark start and goal
    display[start[0]][start[1]] = 'S'
    display[goal[0]][goal[1]] = 'G'
    
    # Mark path
    if path:
        current_pos = start
        for action in path:
            dr, dc = action
            current_pos = (current_pos[0] + dr, current_pos[1] + dc)
            if current_pos != goal:  # Don't overwrite goal marker
                display[current_pos[0]][current_pos[1]] = '*'
    
    # Print grid
    print("Maze: S=start, G=goal, *=path, #=obstacle, .=free")
    for row in display:
        print(' '.join(row))
    print()

def benchmark_algorithms(problem, start, goal):
    """Compare performance of different search algorithms."""
    algorithms = [
        ('BFS', breadth_first_search, None),
        ('DFS', depth_first_search, None),
        ('UCS', uniform_cost_search, None),
        ('A*', astar_search, GridHeuristic(goal, '4-way'))
    ]
    
    print("Algorithm Performance Comparison:")
    print("Algorithm | Time (ms) | Path Length | Optimal?")
    print("-" * 50)
    
    optimal_length = None
    
    for name, algorithm, heuristic in algorithms:
        start_time = time.time()
        
        try:
            if heuristic:
                solution = algorithm(problem, heuristic)
            else:
                solution = algorithm(problem)
            
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            
            if solution:
                path_length = len(solution)
                if optimal_length is None:
                    optimal_length = path_length
                is_optimal = path_length == optimal_length
                
                print(f"{name:9} | {elapsed_ms:8.2f} | {path_length:11} | {is_optimal}")
            else:
                print(f"{name:9} | {elapsed_ms:8.2f} | {'No solution':11} | False")
                
        except Exception as e:
            print(f"{name:9} | Error: {str(e)}")

def interactive_demo():
    """Run interactive pathfinding demonstration."""
    print("=== Pathfinding Algorithm Demo ===")
    print("Comparing BFS, DFS, UCS, and A* on maze navigation\n")
    
    # Create problem
    maze = create_maze()
    start = (0, 0)
    goal = (9, 9)
    problem = GridSearchProblem(maze, start, goal)
    
    print("Original maze:")
    visualize_path(maze, None, start, goal)
    
    # Test each algorithm
    algorithms = [
        ('Breadth-First Search', breadth_first_search, None),
        ('A* Search', astar_search, GridHeuristic(goal, '4-way'))
    ]
    
    for name, algorithm, heuristic in algorithms:
        print(f"=== {name} ===")
        
        start_time = time.time()
        if heuristic:
            solution = algorithm(problem, heuristic)
        else:
            solution = algorithm(problem)
        end_time = time.time()
        
        if solution:
            print(f"Solution found! Path length: {len(solution)}")
            print(f"Time: {(end_time - start_time) * 1000:.2f} ms")
            visualize_path(maze, solution, start, goal)
        else:
            print("No solution found!")
        
        print()
    
    # Performance comparison
    benchmark_algorithms(problem, start, goal)

if __name__ == '__main__':
    interactive_demo()
