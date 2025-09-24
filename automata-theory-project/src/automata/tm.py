"""Turing Machine implementation"""

class TuringMachine:
    def __init__(self, states, alphabet, tape_alphabet, transitions, start_state, accept_states, reject_states):
        self.states = states
        self.alphabet = alphabet
        self.tape_alphabet = tape_alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
        self.reject_states = reject_states
    
    def run(self, input_string, max_steps=1000):
        """Run Turing Machine on input string"""
        # TODO: Implement TM execution
        pass
