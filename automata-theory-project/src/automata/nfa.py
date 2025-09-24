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
        # TODO: Implement NFA acceptance
        pass
