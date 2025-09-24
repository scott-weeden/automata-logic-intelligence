"""Quantum Finite Automaton implementation"""
import math
import cmath

class QFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states, threshold=0.5):
        self.states = list(states)
        self.alphabet = alphabet
        self.transitions = transitions  # Dict: (state, symbol) -> list of (next_state, amplitude)
        self.start_state = start_state
        self.accept_states = accept_states
        self.threshold = threshold
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
    
    def accepts(self, string):
        """Check if string is accepted by QFA with probability > threshold"""
        accept_prob = self.acceptance_probability(string)
        return accept_prob > self.threshold
    
    def acceptance_probability(self, string):
        """Calculate probability of acceptance for given string"""
        # Initialize state vector (all amplitude in start state)
        n_states = len(self.states)
        state_vector = [0.0] * n_states
        start_idx = self.state_to_index[self.start_state]
        state_vector[start_idx] = 1.0
        
        # Process each symbol
        for symbol in string:
            state_vector = self._apply_transition(state_vector, symbol)
        
        # Calculate acceptance probability
        accept_prob = 0.0
        for state in self.accept_states:
            idx = self.state_to_index[state]
            amplitude = state_vector[idx]
            accept_prob += abs(amplitude) ** 2
        
        return accept_prob
    
    def _apply_transition(self, state_vector, symbol):
        """Apply quantum transition for given symbol"""
        n_states = len(self.states)
        new_vector = [0.0] * n_states
        
        for i, state in enumerate(self.states):
            if abs(state_vector[i]) > 1e-10:  # Only process non-zero amplitudes
                if (state, symbol) in self.transitions:
                    for next_state, amplitude in self.transitions[(state, symbol)]:
                        next_idx = self.state_to_index[next_state]
                        new_vector[next_idx] += state_vector[i] * amplitude
        
        return new_vector
    
    def run_with_trace(self, string):
        """Run QFA with detailed trace of quantum states"""
        n_states = len(self.states)
        state_vector = [0.0] * n_states
        start_idx = self.state_to_index[self.start_state]
        state_vector[start_idx] = 1.0
        
        trace = []
        trace.append({
            'step': 'initial',
            'symbol': None,
            'state_vector': state_vector[:],
            'probabilities': [abs(amp)**2 for amp in state_vector]
        })
        
        for i, symbol in enumerate(string):
            state_vector = self._apply_transition(state_vector, symbol)
            trace.append({
                'step': i,
                'symbol': symbol,
                'state_vector': state_vector[:],
                'probabilities': [abs(amp)**2 for amp in state_vector]
            })
        
        # Final acceptance probability
        accept_prob = sum(abs(state_vector[self.state_to_index[state]])**2 
                         for state in self.accept_states)
        
        return accept_prob, trace


class MeasureOnceQFA(QFA):
    """Measure-once Quantum Finite Automaton"""
    
    def __init__(self, states, alphabet, transitions, start_state, accept_states, 
                 non_accept_states, threshold=0.5):
        super().__init__(states, alphabet, transitions, start_state, accept_states, threshold)
        self.non_accept_states = non_accept_states
    
    def accepts(self, string):
        """Check acceptance with measure-once semantics"""
        # Process string
        n_states = len(self.states)
        state_vector = [0.0] * n_states
        start_idx = self.state_to_index[self.start_state]
        state_vector[start_idx] = 1.0
        
        for symbol in string:
            state_vector = self._apply_transition(state_vector, symbol)
        
        # Measure: probability of being in accept states
        accept_prob = sum(abs(state_vector[self.state_to_index[state]])**2 
                         for state in self.accept_states)
        
        return accept_prob > self.threshold


class ReversibleQFA(QFA):
    """Reversible Quantum Finite Automaton"""
    
    def __init__(self, states, alphabet, unitary_transitions, start_state, accept_states, threshold=0.5):
        self.states = list(states)
        self.alphabet = alphabet
        self.unitary_transitions = unitary_transitions  # Dict: symbol -> unitary matrix
        self.start_state = start_state
        self.accept_states = accept_states
        self.threshold = threshold
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
    
    def _apply_unitary(self, state_vector, unitary_matrix):
        """Apply unitary transformation to state vector"""
        n = len(state_vector)
        new_vector = [0.0] * n
        
        for i in range(n):
            for j in range(n):
                new_vector[i] += unitary_matrix[i][j] * state_vector[j]
        
        return new_vector
    
    def acceptance_probability(self, string):
        """Calculate acceptance probability using unitary evolution"""
        n_states = len(self.states)
        state_vector = [0.0] * n_states
        start_idx = self.state_to_index[self.start_state]
        state_vector[start_idx] = 1.0
        
        # Apply unitary transformations
        for symbol in string:
            if symbol in self.unitary_transitions:
                state_vector = self._apply_unitary(state_vector, self.unitary_transitions[symbol])
        
        # Calculate acceptance probability
        accept_prob = sum(abs(state_vector[self.state_to_index[state]])**2 
                         for state in self.accept_states)
        
        return accept_prob


def create_hadamard_qfa():
    """Create a simple QFA using Hadamard-like transitions"""
    # Simple 2-state QFA with quantum superposition
    states = {'q0', 'q1'}
    alphabet = {'0', '1'}
    
    # Hadamard-like transitions: create superposition
    h = 1/math.sqrt(2)
    transitions = {
        ('q0', '0'): [('q0', h), ('q1', h)],
        ('q0', '1'): [('q0', h), ('q1', -h)],
        ('q1', '0'): [('q0', h), ('q1', -h)],
        ('q1', '1'): [('q0', -h), ('q1', h)]
    }
    
    return QFA(states, alphabet, transitions, 'q0', {'q1'})


def create_deutsch_qfa():
    """Create QFA implementing Deutsch's algorithm concept"""
    states = {'q0', 'q1', 'q2', 'q3'}
    alphabet = {'x', 'f'}
    
    # Simplified quantum circuit for function evaluation
    h = 1/math.sqrt(2)
    transitions = {
        ('q0', 'x'): [('q1', h), ('q2', h)],  # Hadamard on input
        ('q1', 'f'): [('q3', 1.0)],           # Constant function
        ('q2', 'f'): [('q3', -1.0)],          # Balanced function
    }
    
    return QFA(states, alphabet, transitions, 'q0', {'q3'})
