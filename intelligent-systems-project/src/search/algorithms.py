"""
Search Algorithms Implementation

Implements uninformed and informed search strategies:
- Breadth-First Search (BFS): Complete, optimal for unit costs
- Depth-First Search (DFS): Space-efficient but not optimal
- Uniform Cost Search (UCS): Optimal for varying step costs
- A* Search: Optimal with admissible heuristics

Based on CS 5368 Week 3-4 material on search problem formulation.
"""

from collections import deque
import heapq
from .problem import SearchProblem
from .utils import Node

def breadth_first_search(problem):
    """
    BFS using FIFO queue. Guarantees shortest path in unweighted graphs.
    Time: O(b^d), Space: O(b^d) where b=branching factor, d=depth
    """
    if problem.goal_test(problem.initial_state):
        return []
    
    frontier = deque([Node(problem.initial_state)])
    explored = set()
    
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child.solution()
                frontier.append(child)
    
    return None  # No solution found

def depth_first_search(problem):
    """
    DFS using LIFO stack (recursion). Space-efficient: O(bm).
    Not complete in infinite spaces, not optimal.
    """
    def dfs_recursive(node, explored):
        if problem.goal_test(node.state):
            return node.solution()
        
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            if child.state not in explored:
                result = dfs_recursive(child, explored)
                if result is not None:
                    return result
        return None
    
    return dfs_recursive(Node(problem.initial_state), set())

def uniform_cost_search(problem):
    """
    UCS using priority queue ordered by path cost g(n).
    Optimal for problems with varying step costs.
    """
    frontier = [(0, Node(problem.initial_state))]
    explored = set()
    
    while frontier:
        cost, node = heapq.heappop(frontier)
        
        if problem.goal_test(node.state):
            return node.solution()
        
        if node.state in explored:
            continue
            
        explored.add(node.state)
        
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            if child.state not in explored:
                heapq.heappush(frontier, (child.path_cost, child))
    
    return None

def astar_search(problem, heuristic):
    """
    A* search using f(n) = g(n) + h(n) evaluation function.
    Optimal if heuristic is admissible (never overestimates).
    """
    frontier = [(heuristic(problem.initial_state), 0, Node(problem.initial_state))]
    explored = set()
    
    while frontier:
        f_cost, g_cost, node = heapq.heappop(frontier)
        
        if problem.goal_test(node.state):
            return node.solution()
        
        if node.state in explored:
            continue
            
        explored.add(node.state)
        
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            if child.state not in explored:
                h_cost = heuristic(child.state)
                f_cost = child.path_cost + h_cost
                heapq.heappush(frontier, (f_cost, child.path_cost, child))
    
    return None

def greedy_best_first_search(problem, heuristic):
    """
    Greedy search using only h(n). Fast but not optimal.
    Expands node that appears closest to goal.
    """
    frontier = [(heuristic(problem.initial_state), Node(problem.initial_state))]
    explored = set()
    
    while frontier:
        h_cost, node = heapq.heappop(frontier)
        
        if problem.goal_test(node.state):
            return node.solution()
        
        explored.add(node.state)
        
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            if child.state not in explored:
                h_cost = heuristic(child.state)
                heapq.heappush(frontier, (h_cost, child))
    
    return None
