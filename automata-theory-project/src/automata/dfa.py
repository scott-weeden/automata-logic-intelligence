"""Deterministic Finite Automaton implementation"""

class DFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
    
    def accepts(self, string):
        """Check if string is accepted by DFA"""
        current_state = self.start_state
        for symbol in string:
            if (current_state, symbol) not in self.transitions:
                return False
            current_state = self.transitions[(current_state, symbol)]
        return current_state in self.accept_states
