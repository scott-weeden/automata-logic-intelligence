"""Tests for Quantum Finite Automaton (QFA)"""

import pytest
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from automata import QFA, MeasureOnceQFA, ReversibleQFA, create_hadamard_qfa, create_deutsch_qfa

class TestQFABasic:
    """Basic QFA functionality tests"""
    
    def test_qfa_initialization(self):
        """Test QFA can be initialized"""
        qfa = QFA(
            states={'q0', 'q1'},
            alphabet={'0', '1'},
            transitions={
                ('q0', '0'): [('q1', 0.5), ('q0', 0.5)],
                ('q0', '1'): [('q0', 1.0)]
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert qfa.states == ['q0', 'q1']
        assert qfa.alphabet == {'0', '1'}
        assert qfa.start_state == 'q0'
        assert qfa.accept_states == {'q1'}
    
    def test_qfa_acceptance_probability(self):
        """Test QFA acceptance probability calculation"""
        # Simple QFA with deterministic transitions
        qfa = QFA(
            states={'q0', 'q1'},
            alphabet={'a'},
            transitions={
                ('q0', 'a'): [('q1', 1.0)]
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        prob = qfa.acceptance_probability('a')
        assert abs(prob - 1.0) < 1e-10
        
        prob = qfa.acceptance_probability('')
        assert abs(prob - 0.0) < 1e-10
    
    def test_qfa_superposition(self):
        """Test QFA with quantum superposition"""
        h = 1/math.sqrt(2)  # Hadamard coefficient
        
        qfa = QFA(
            states={'q0', 'q1'},
            alphabet={'0'},
            transitions={
                ('q0', '0'): [('q0', h), ('q1', h)]
            },
            start_state='q0',
            accept_states={'q1'},
            threshold=0.4
        )
        
        prob = qfa.acceptance_probability('0')
        assert abs(prob - 0.5) < 1e-10  # |h|^2 = 0.5
        
        assert qfa.accepts('0') == True  # 0.5 > 0.4

class TestQFAQuantumEffects:
    """Test quantum-specific effects in QFA"""
    
    def test_interference(self):
        """Test quantum interference effects"""
        # QFA with interference
        h = 1/math.sqrt(2)
        
        qfa = QFA(
            states={'q0', 'q1', 'q2'},
            alphabet={'a', 'b'},
            transitions={
                ('q0', 'a'): [('q1', h), ('q2', h)],
                ('q1', 'b'): [('q0', h)],
                ('q2', 'b'): [('q0', -h)]  # Negative amplitude
            },
            start_state='q0',
            accept_states={'q0'}
        )
        
        prob = qfa.acceptance_probability('ab')
        # After 'a': |q1⟩ + |q2⟩ (unnormalized)
        # After 'b': h*h|q0⟩ + h*(-h)|q0⟩ = 0|q0⟩
        assert abs(prob - 0.0) < 1e-10
    
    def test_amplitude_evolution(self):
        """Test quantum amplitude evolution"""
        qfa = QFA(
            states={'q0', 'q1'},
            alphabet={'x'},
            transitions={
                ('q0', 'x'): [('q1', 0.8)],
                ('q1', 'x'): [('q0', 0.6)]
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        # After one 'x': amplitude 0.8 in q1
        prob1 = qfa.acceptance_probability('x')
        assert abs(prob1 - 0.64) < 1e-10  # |0.8|^2
        
        # After two 'x': amplitude 0.8*0.6 = 0.48 in q0
        prob2 = qfa.acceptance_probability('xx')
        assert abs(prob2 - 0.0) < 1e-10  # q0 not accepting

class TestQFATracing:
    """Test QFA execution tracing"""
    
    def test_trace_quantum_states(self):
        """Test QFA trace functionality"""
        h = 1/math.sqrt(2)
        
        qfa = QFA(
            states={'q0', 'q1'},
            alphabet={'0'},
            transitions={
                ('q0', '0'): [('q0', h), ('q1', h)]
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        prob, trace = qfa.run_with_trace('0')
        
        assert len(trace) == 2  # Initial + after '0'
        assert trace[0]['step'] == 'initial'
        assert trace[1]['step'] == 0
        assert trace[1]['symbol'] == '0'
        
        # Check probability conservation
        for step in trace:
            total_prob = sum(step['probabilities'])
            assert abs(total_prob - 1.0) < 1e-10

class TestMeasureOnceQFA:
    """Test Measure-once QFA"""
    
    def test_measure_once_semantics(self):
        """Test measure-once QFA behavior"""
        h = 1/math.sqrt(2)
        
        mqfa = MeasureOnceQFA(
            states={'q0', 'q1', 'q2'},
            alphabet={'a'},
            transitions={
                ('q0', 'a'): [('q1', h), ('q2', h)]
            },
            start_state='q0',
            accept_states={'q1'},
            non_accept_states={'q2'}
        )
        
        prob = mqfa.acceptance_probability('a')
        assert abs(prob - 0.5) < 1e-10

class TestReversibleQFA:
    """Test Reversible QFA"""
    
    def test_unitary_evolution(self):
        """Test reversible QFA with unitary matrices"""
        # 2x2 Hadamard matrix
        h = 1/math.sqrt(2)
        hadamard = [[h, h], [h, -h]]
        
        rqfa = ReversibleQFA(
            states={'q0', 'q1'},
            alphabet={'H'},
            unitary_transitions={'H': hadamard},
            start_state='q0',
            accept_states={'q1'}
        )
        
        prob = rqfa.acceptance_probability('H')
        assert abs(prob - 0.5) < 1e-10
        
        # Two Hadamards should return to original state
        prob = rqfa.acceptance_probability('HH')
        assert abs(prob - 0.0) < 1e-10

class TestQFAExamples:
    """Test predefined QFA examples"""
    
    def test_hadamard_qfa(self):
        """Test Hadamard QFA example"""
        qfa = create_hadamard_qfa()
        
        assert qfa.states == ['q0', 'q1']
        assert qfa.alphabet == {'0', '1'}
        assert qfa.start_state == 'q0'
        assert qfa.accept_states == {'q1'}
        
        # Test some basic properties
        prob0 = qfa.acceptance_probability('0')
        prob1 = qfa.acceptance_probability('1')
        
        # Both should have same magnitude due to Hadamard symmetry
        assert abs(abs(prob0) - abs(prob1)) < 1e-10
    
    def test_deutsch_qfa(self):
        """Test Deutsch algorithm QFA example"""
        qfa = create_deutsch_qfa()
        
        assert qfa.states == ['q0', 'q1', 'q2', 'q3']
        assert qfa.alphabet == {'x', 'f'}
        assert qfa.start_state == 'q0'
        assert qfa.accept_states == {'q3'}

class TestQFAEdgeCases:
    """Test QFA edge cases"""
    
    def test_no_transitions(self):
        """Test QFA with no valid transitions"""
        qfa = QFA(
            states={'q0', 'q1'},
            alphabet={'a', 'b'},
            transitions={
                ('q0', 'a'): [('q1', 1.0)]
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        # No transition for 'b'
        prob = qfa.acceptance_probability('b')
        assert abs(prob - 0.0) < 1e-10
    
    def test_amplitude_normalization(self):
        """Test behavior with non-normalized amplitudes"""
        qfa = QFA(
            states={'q0', 'q1'},
            alphabet={'a'},
            transitions={
                ('q0', 'a'): [('q1', 2.0)]  # Non-normalized amplitude
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        prob = qfa.acceptance_probability('a')
        assert abs(prob - 4.0) < 1e-10  # |2.0|^2 = 4.0
    
    def test_complex_amplitudes(self):
        """Test QFA with complex amplitudes"""
        import cmath
        
        qfa = QFA(
            states={'q0', 'q1'},
            alphabet={'i'},
            transitions={
                ('q0', 'i'): [('q1', 1j)]  # Imaginary amplitude
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        prob = qfa.acceptance_probability('i')
        assert abs(prob - 1.0) < 1e-10  # |1j|^2 = 1.0
