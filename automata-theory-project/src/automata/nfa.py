"""Non-deterministic Finite Automaton implementation"""

class NFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
    
    def accepts(self, string):
        """Check if string is accepted by NFA"""
        def epsilon_closure(states):
            closure = set(states)
            stack = list(states)
            while stack:
                state = stack.pop()
                if (state, '') in self.transitions:
                    for next_state in self.transitions[(state, '')]:
                        if next_state not in closure:
                            closure.add(next_state)
                            stack.append(next_state)
            return closure
        
        current_states = epsilon_closure({self.start_state})
        
        for symbol in string:
            next_states = set()
            for state in current_states:
                if (state, symbol) in self.transitions:
                    next_states.update(self.transitions[(state, symbol)])
            current_states = epsilon_closure(next_states)
        
        return bool(current_states & self.accept_states)
