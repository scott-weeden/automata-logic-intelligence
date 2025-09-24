"""Pushdown Automaton implementation (optional scaffold)"""

class PDA:
    def __init__(self, states, alphabet, stack_alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.stack_alphabet = stack_alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
    
    def accepts(self, string):
        """Check if string is accepted by PDA"""
        # TODO: Implement PDA acceptance
        pass
