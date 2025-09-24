"""
Search Utilities

Helper classes and functions for search algorithms.
Includes Node class for representing search tree nodes.
"""

class Node:
    """
    Node in search tree. Contains state, parent, action, path cost.
    Used by all search algorithms to track exploration.
    """
    
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Initialize search node."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1
    
    def child_node(self, problem, action):
        """Create child node by applying action."""
        next_state = problem.result(self.state, action)
        next_cost = problem.path_cost(self.path_cost, self.state, action, next_state)
        return Node(next_state, self, action, next_cost)
    
    def solution(self):
        """Return sequence of actions from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))
    
    def path(self):
        """Return sequence of states from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))
    
    def __eq__(self, other):
        """Nodes are equal if they have same state."""
        return isinstance(other, Node) and self.state == other.state
    
    def __hash__(self):
        """Hash based on state for use in sets/dicts."""
        return hash(self.state)
    
    def __lt__(self, other):
        """Compare nodes by path cost for priority queue."""
        return self.path_cost < other.path_cost
    
    def __repr__(self):
        """String representation for debugging."""
        return f"Node({self.state}, cost={self.path_cost})"

def memoize(fn, slot=None):
    """
    Memoization decorator for expensive function calls.
    Useful for caching heuristic computations.
    """
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]
        memoized_fn.cache = {}
    return memoized_fn

class PriorityQueue:
    """
    Priority queue implementation for search algorithms.
    Supports insertion, removal, and membership testing.
    """
    
    def __init__(self, order='min', f=lambda x: x):
        """
        Initialize priority queue.
        order: 'min' for min-heap, 'max' for max-heap
        f: function to extract priority from items
        """
        self.heap = []
        self.f = f
        self.order = order
    
    def append(self, item):
        """Add item to queue."""
        import heapq
        priority = self.f(item)
        if self.order == 'max':
            priority = -priority
        heapq.heappush(self.heap, (priority, item))
    
    def extend(self, items):
        """Add multiple items to queue."""
        for item in items:
            self.append(item)
    
    def pop(self):
        """Remove and return item with highest priority."""
        import heapq
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')
    
    def __len__(self):
        """Return number of items in queue."""
        return len(self.heap)
    
    def __contains__(self, key):
        """Check if key is in queue."""
        return any(item == key for _, item in self.heap)
    
    def __getitem__(self, key):
        """Get item with given key."""
        for _, item in self.heap:
            if item == key:
                return item
        raise KeyError(str(key) + " is not in the priority queue")
    
    def __delitem__(self, key):
        """Remove item with given key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
            import heapq
            heapq.heapify(self.heap)
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
