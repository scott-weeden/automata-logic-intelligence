import util

class DFS(object):
    def depthFirstSearch(self, problem):
        """
        Search the deepest nodes in the search tree first
        [2nd Edition: p 75, 3rd Edition: p 87]

        Your search algorithm needs to return a list of actions that reaches
        the goal.  Make sure to implement a graph search algorithm
        [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:

        print "Start:", problem.getStartState()
        print "Is the start a goal?", problem.isGoalState(problem.getStartState())
        print "Start's successors:", problem.getSuccessors(problem.getStartState())
        """
        frontier = util.Stack()
        start_state = problem.getStartState()
        frontier.push((start_state, []))
        visited = set()

        while not frontier.isEmpty():
            state, actions = frontier.pop()
            if state in visited:
                continue
            if problem.isGoalState(state):
                return actions
            visited.add(state)

            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    frontier.push((successor, actions + [action]))

        return []

class BFS(object):
    def breadthFirstSearch(self, problem):
        start_state = problem.getStartState()
        if problem.isGoalState(start_state):
            return []

        frontier = util.Queue()
        frontier.push((start_state, []))
        visited = set()

        while not frontier.isEmpty():
            state, actions = frontier.pop()
            if state in visited:
                continue
            if problem.isGoalState(state):
                return actions
            visited.add(state)

            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    frontier.push((successor, actions + [action]))

        return []

class UCS(object):
    def uniformCostSearch(self, problem):
        frontier = util.PriorityQueue()
        start_state = problem.getStartState()
        frontier.push((start_state, [], 0), 0)
        visited = {}

        while not frontier.isEmpty():
            state, actions, cost = frontier.pop()
            if state in visited and visited[state] <= cost:
                continue
            visited[state] = cost

            if problem.isGoalState(state):
                return actions

            for successor, action, step_cost in problem.getSuccessors(state):
                new_cost = cost + step_cost
                recorded_cost = visited.get(successor)
                if recorded_cost is None or new_cost < recorded_cost:
                    frontier.push((successor, actions + [action], new_cost), new_cost)

        return []

class aSearch (object):
    def nullHeuristic( state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0
    def aStarSearch(self,problem, heuristic=nullHeuristic):
        "Search the node that has the lowest combined cost and heuristic first."
        frontier = util.PriorityQueue()
        start_state = problem.getStartState()
        start_cost = heuristic(start_state, problem)
        frontier.push((start_state, [], 0), start_cost)
        visited = {}

        while not frontier.isEmpty():
            state, actions, cost = frontier.pop()
            if state in visited and visited[state] <= cost:
                continue
            visited[state] = cost

            if problem.isGoalState(state):
                return actions

            for successor, action, step_cost in problem.getSuccessors(state):
                new_cost = cost + step_cost
                recorded_cost = visited.get(successor)
                if recorded_cost is None or new_cost < recorded_cost:
                    priority = new_cost + heuristic(successor, problem)
                    frontier.push((successor, actions + [action], new_cost), priority)

        return []

