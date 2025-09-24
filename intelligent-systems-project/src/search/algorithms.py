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
from .utils import Node

class SearchAgent:
    """Base class for search agents."""
    
    def __init__(self):
        self.nodes_expanded = 0
    
    def search(self, problem):
        """Search for solution. Must be implemented by subclasses."""
        raise NotImplementedError

class BreadthFirstSearch(SearchAgent):
    """BFS using FIFO queue. Guarantees shortest path in unweighted graphs."""
    
    def search(self, problem):
        """Search using breadth-first strategy."""
        self.nodes_expanded = 0
        
        if problem.is_goal_state(problem.get_start_state()):
            return []
        
        frontier = deque([Node(problem.get_start_state())])
        explored = set()
        
        while frontier:
            node = frontier.popleft()
            
            if node.state in explored:
                continue
                
            explored.add(node.state)
            self.nodes_expanded += 1
            
            for successor, action, cost in problem.get_successors(node.state):
                if successor not in explored:
                    child = Node(successor, node, action, node.path_cost + cost)
                    
                    if problem.is_goal_state(successor):
                        return child.solution()
                    
                    frontier.append(child)
        
        return None

class DepthFirstSearch(SearchAgent):
    """DFS using LIFO stack (recursion). Space-efficient: O(bm)."""
    
    def search(self, problem):
        """Search using depth-first strategy."""
        self.nodes_expanded = 0
        
        def dfs_recursive(node, explored):
            if problem.is_goal_state(node.state):
                return node.solution()
            
            if node.state in explored:
                return None
                
            explored.add(node.state)
            self.nodes_expanded += 1
            
            for successor, action, cost in problem.get_successors(node.state):
                if successor not in explored:
                    child = Node(successor, node, action, node.path_cost + cost)
                    result = dfs_recursive(child, explored)
                    if result is not None:
                        return result
            
            explored.remove(node.state)  # Backtrack
            return None
        
        return dfs_recursive(Node(problem.get_start_state()), set())

class UniformCostSearch(SearchAgent):
    """UCS using priority queue ordered by path cost g(n)."""
    
    def search(self, problem):
        """Search using uniform cost strategy."""
        self.nodes_expanded = 0
        
        frontier = [(0, Node(problem.get_start_state()))]
        explored = set()
        
        while frontier:
            cost, node = heapq.heappop(frontier)
            
            if problem.is_goal_state(node.state):
                return node.solution()
            
            if node.state in explored:
                continue
                
            explored.add(node.state)
            self.nodes_expanded += 1
            
            for successor, action, step_cost in problem.get_successors(node.state):
                if successor not in explored:
                    child = Node(successor, node, action, node.path_cost + step_cost)
                    heapq.heappush(frontier, (child.path_cost, child))
        
        return None

class AStarSearch(SearchAgent):
    """A* search using f(n) = g(n) + h(n) evaluation function."""
    
    def __init__(self, heuristic=None):
        super().__init__()
        self.heuristic = heuristic or (lambda state, problem: 0)
    
    def search(self, problem):
        """Search using A* strategy."""
        self.nodes_expanded = 0
        
        start_node = Node(problem.get_start_state())
        h_cost = self.heuristic(start_node.state, problem)
        frontier = [(h_cost, 0, start_node)]
        explored = set()
        
        while frontier:
            f_cost, g_cost, node = heapq.heappop(frontier)
            
            if problem.is_goal_state(node.state):
                return node.solution()
            
            if node.state in explored:
                continue
                
            explored.add(node.state)
            self.nodes_expanded += 1
            
            for successor, action, step_cost in problem.get_successors(node.state):
                if successor not in explored:
                    child = Node(successor, node, action, node.path_cost + step_cost)
                    h_cost = self.heuristic(child.state, problem)
                    f_cost = child.path_cost + h_cost
                    heapq.heappush(frontier, (f_cost, child.path_cost, child))
        
        return None

class GreedyBestFirstSearch(SearchAgent):
    """Greedy search using only h(n). Fast but not optimal."""
    
    def __init__(self, heuristic=None):
        super().__init__()
        self.heuristic = heuristic or (lambda state, problem: 0)
    
    def search(self, problem):
        """Search using greedy best-first strategy."""
        self.nodes_expanded = 0
        
        start_node = Node(problem.get_start_state())
        h_cost = self.heuristic(start_node.state, problem)
        frontier = [(h_cost, start_node)]
        explored = set()
        
        while frontier:
            h_cost, node = heapq.heappop(frontier)
            
            if problem.is_goal_state(node.state):
                return node.solution()
            
            if node.state in explored:
                continue
                
            explored.add(node.state)
            self.nodes_expanded += 1
            
            for successor, action, step_cost in problem.get_successors(node.state):
                if successor not in explored:
                    child = Node(successor, node, action, node.path_cost + step_cost)
                    h_cost = self.heuristic(child.state, problem)
                    heapq.heappush(frontier, (h_cost, child))
        
        return None

class IterativeDeepeningSearch(SearchAgent):
    """Iterative deepening search combines benefits of BFS and DFS."""
    
    def search(self, problem, max_depth=50):
        """Search using iterative deepening strategy."""
        self.nodes_expanded = 0
        
        for depth in range(max_depth):
            result = self._depth_limited_search(problem, depth)
            if result is not None:
                return result
        
        return None
    
    def _depth_limited_search(self, problem, limit):
        """Depth-limited search helper."""
        def dls_recursive(node, depth):
            if problem.is_goal_state(node.state):
                return node.solution()
            
            if depth >= limit:
                return None
            
            self.nodes_expanded += 1
            
            for successor, action, cost in problem.get_successors(node.state):
                child = Node(successor, node, action, node.path_cost + cost)
                result = dls_recursive(child, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        return dls_recursive(Node(problem.get_start_state()), 0)
