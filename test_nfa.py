"""Tests for Non-deterministic Finite Automaton (NFA)"""

import pytest
from automata import NFA


class TestNFABasic:
    """Basic NFA functionality tests"""
    
    def test_nfa_initialization(self):
        """Test NFA can be initialized with correct parameters"""
        nfa = NFA(
            states={'q0', 'q1'},
            alphabet={'0', '1'},
            transitions={
                ('q0', '0'): {'q0', 'q1'},
                ('q0', '1'): {'q0'},
                ('q1', '1'): {'q1'}
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert nfa.states == {'q0', 'q1'}
        assert nfa.alphabet == {'0', '1'}
        assert nfa.start_state == 'q0'
        assert nfa.accept_states == {'q1'}
    
    def test_nfa_empty_string(self):
        """Test NFA with empty string"""
        # NFA that accepts empty string
        nfa = NFA(
            states={'q0'},
            alphabet={'0', '1'},
            transitions={},
            start_state='q0',
            accept_states={'q0'}
        )
        
        assert nfa.accepts('') == True
    
    def test_nfa_single_character(self):
        """Test NFA with single character transitions"""
        # NFA that accepts strings ending in '1'
        nfa = NFA(
            states={'q0', 'q1'},
            alphabet={'0', '1'},
            transitions={
                ('q0', '0'): {'q0'},
                ('q0', '1'): {'q0', 'q1'},
                ('q1', '0'): set(),
                ('q1', '1'): set()
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert nfa.accepts('1') == True
        assert nfa.accepts('01') == True
        assert nfa.accepts('0') == False


class TestNFANondeterminism:
    """Test NFA non-deterministic behavior"""
    
    def test_multiple_transitions(self):
        """Test NFA with multiple transitions from same state"""
        # NFA that accepts strings containing '01'
        nfa = NFA(
            states={'q0', 'q1', 'q2'},
            alphabet={'0', '1'},
            transitions={
                ('q0', '0'): {'q0', 'q1'},
                ('q0', '1'): {'q0'},
                ('q1', '1'): {'q2'},
                ('q2', '0'): {'q2'},
                ('q2', '1'): {'q2'}
            },
            start_state='q0',
            accept_states={'q2'}
        )
        
        assert nfa.accepts('01') == True
        assert nfa.accepts('001') == True
        assert nfa.accepts('0110') == True
        assert nfa.accepts('10') == False
        assert nfa.accepts('0') == False
    
    def test_epsilon_transitions(self):
        """Test NFA with epsilon (empty) transitions"""
        # NFA with epsilon transitions
        nfa = NFA(
            states={'q0', 'q1', 'q2'},
            alphabet={'a', 'b'},
            transitions={
                ('q0', ''): {'q1'},  # epsilon transition
                ('q0', 'a'): {'q0'},
                ('q1', 'b'): {'q2'},
                ('q2', 'b'): {'q2'}
            },
            start_state='q0',
            accept_states={'q2'}
        )
        
        assert nfa.accepts('b') == True  # via epsilon transition
        assert nfa.accepts('ab') == True
        assert nfa.accepts('aabb') == True
        assert nfa.accepts('a') == False


class TestNFAComplexPatterns:
    """Test NFA with complex acceptance patterns"""
    
    def test_union_pattern(self):
        """Test NFA that accepts union of two patterns"""
        # Accepts strings ending in '00' OR containing '11'
        nfa = NFA(
            states={'q0', 'q1', 'q2', 'q3', 'q4'},
            alphabet={'0', '1'},
            transitions={
                ('q0', '0'): {'q0', 'q1'},
                ('q0', '1'): {'q0', 'q3'},
                ('q1', '0'): {'q2'},  # path for '00'
                ('q1', '1'): {'q0'},
                ('q2', '0'): {'q2'},
                ('q2', '1'): {'q0'},
                ('q3', '1'): {'q4'},  # path for '11'
                ('q3', '0'): {'q0'},
                ('q4', '0'): {'q4'},
                ('q4', '1'): {'q4'}
            },
            start_state='q0',
            accept_states={'q2', 'q4'}
        )
        
        assert nfa.accepts('00') == True    # ends in '00'
        assert nfa.accepts('100') == True   # ends in '00'
        assert nfa.accepts('11') == True    # contains '11'
        assert nfa.accepts('011') == True   # contains '11'
        assert nfa.accepts('1100') == True  # both patterns
        assert nfa.accepts('01') == False   # neither pattern
    
    def test_kleene_star_pattern(self):
        """Test NFA implementing Kleene star pattern"""
        # Accepts (ab)*
        nfa = NFA(
            states={'q0', 'q1'},
            alphabet={'a', 'b'},
            transitions={
                ('q0', 'a'): {'q1'},
                ('q1', 'b'): {'q0'}
            },
            start_state='q0',
            accept_states={'q0'}
        )
        
        assert nfa.accepts('') == True
        assert nfa.accepts('ab') == True
        assert nfa.accepts('abab') == True
        assert nfa.accepts('ababab') == True
        assert nfa.accepts('a') == False
        assert nfa.accepts('aba') == False
        assert nfa.accepts('ba') == False


class TestNFAEdgeCases:
    """Test NFA edge cases and error conditions"""
    
    def test_no_transitions(self):
        """Test NFA with no transitions"""
        nfa = NFA(
            states={'q0'},
            alphabet={'0', '1'},
            transitions={},
            start_state='q0',
            accept_states={'q0'}
        )
        
        assert nfa.accepts('') == True
        assert nfa.accepts('0') == False
        assert nfa.accepts('1') == False
    
    def test_unreachable_accept_state(self):
        """Test NFA with unreachable accept state"""
        nfa = NFA(
            states={'q0', 'q1'},
            alphabet={'0', '1'},
            transitions={
                ('q0', '0'): {'q0'},
                ('q0', '1'): {'q0'}
            },
            start_state='q0',
            accept_states={'q1'}  # q1 is unreachable
        )
        
        assert nfa.accepts('') == False
        assert nfa.accepts('01') == False
        assert nfa.accepts('10') == False
    
    def test_dead_end_transitions(self):
        """Test NFA with transitions to empty set"""
        nfa = NFA(
            states={'q0', 'q1'},
            alphabet={'0', '1'},
            transitions={
                ('q0', '0'): {'q1'},
                ('q0', '1'): set(),  # dead end
                ('q1', '0'): set(),  # dead end
                ('q1', '1'): set()   # dead end
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert nfa.accepts('0') == True
        assert nfa.accepts('1') == False
        assert nfa.accepts('00') == False
        assert nfa.accepts('01') == False


class TestNFASpecialCases:
    """Test special NFA configurations"""
    
    def test_single_state_nfa(self):
        """Test NFA with single state"""
        nfa = NFA(
            states={'q0'},
            alphabet={'a'},
            transitions={
                ('q0', 'a'): {'q0'}
            },
            start_state='q0',
            accept_states={'q0'}
        )
        
        assert nfa.accepts('') == True
        assert nfa.accepts('a') == True
        assert nfa.accepts('aa') == True
        assert nfa.accepts('aaa') == True
    
    def test_all_states_accepting(self):
        """Test NFA where all states are accepting"""
        nfa = NFA(
            states={'q0', 'q1', 'q2'},
            alphabet={'0', '1'},
            transitions={
                ('q0', '0'): {'q1'},
                ('q0', '1'): {'q2'},
                ('q1', '0'): {'q0'},
                ('q1', '1'): {'q2'},
                ('q2', '0'): {'q1'},
                ('q2', '1'): {'q0'}
            },
            start_state='q0',
            accept_states={'q0', 'q1', 'q2'}
        )
        
        assert nfa.accepts('') == True
        assert nfa.accepts('0') == True
        assert nfa.accepts('1') == True
        assert nfa.accepts('01') == True
        assert nfa.accepts('10') == True
    
    def test_no_accepting_states(self):
        """Test NFA with no accepting states"""
        nfa = NFA(
            states={'q0', 'q1'},
            alphabet={'0', '1'},
            transitions={
                ('q0', '0'): {'q1'},
                ('q0', '1'): {'q0'},
                ('q1', '0'): {'q0'},
                ('q1', '1'): {'q1'}
            },
            start_state='q0',
            accept_states=set()
        )
        
        assert nfa.accepts('') == False
        assert nfa.accepts('0') == False
        assert nfa.accepts('1') == False
        assert nfa.accepts('01') == False
