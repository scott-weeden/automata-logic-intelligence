"""Tests for Pushdown Automaton (PDA)"""

import pytest
from automata import PDA


class TestPDABasic:
    """Basic PDA functionality tests"""
    
    def test_pda_initialization(self):
        """Test PDA can be initialized with correct parameters"""
        pda = PDA(
            states={'q0', 'q1', 'q2'},
            alphabet={'a', 'b'},
            stack_alphabet={'Z', 'A'},
            transitions={
                ('q0', 'a', 'Z'): {('q1', 'AZ')},
                ('q1', 'b', 'A'): {('q2', '')}
            },
            start_state='q0',
            accept_states={'q2'}
        )
        
        assert pda.states == {'q0', 'q1', 'q2'}
        assert pda.alphabet == {'a', 'b'}
        assert pda.stack_alphabet == {'Z', 'A'}
        assert pda.start_state == 'q0'
        assert pda.accept_states == {'q2'}
    
    def test_pda_empty_string(self):
        """Test PDA with empty string"""
        # PDA that accepts empty string
        pda = PDA(
            states={'q0'},
            alphabet={'a'},
            stack_alphabet={'Z'},
            transitions={},
            start_state='q0',
            accept_states={'q0'}
        )
        
        assert pda.accepts('') == True
    
    def test_pda_simple_acceptance(self):
        """Test PDA with simple acceptance pattern"""
        # PDA that accepts 'a'
        pda = PDA(
            states={'q0', 'q1'},
            alphabet={'a'},
            stack_alphabet={'Z'},
            transitions={
                ('q0', 'a', 'Z'): {('q1', 'Z')}
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert pda.accepts('a') == True
        assert pda.accepts('') == False
        assert pda.accepts('aa') == False


class TestPDAStackOperations:
    """Test PDA stack manipulation"""
    
    def test_push_operation(self):
        """Test PDA push operations"""
        # PDA that pushes 'A' for each 'a'
        pda = PDA(
            states={'q0', 'q1'},
            alphabet={'a'},
            stack_alphabet={'Z', 'A'},
            transitions={
                ('q0', 'a', 'Z'): {('q0', 'AZ')},
                ('q0', 'a', 'A'): {('q0', 'AA')},
                ('q0', '', 'Z'): {('q1', 'Z')}  # epsilon transition
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert pda.accepts('') == True
        assert pda.accepts('a') == True
        assert pda.accepts('aa') == True
        assert pda.accepts('aaa') == True
    
    def test_pop_operation(self):
        """Test PDA pop operations"""
        # PDA that accepts a^n b^n (equal a's and b's)
        pda = PDA(
            states={'q0', 'q1', 'q2'},
            alphabet={'a', 'b'},
            stack_alphabet={'Z', 'A'},
            transitions={
                ('q0', 'a', 'Z'): {('q0', 'AZ')},
                ('q0', 'a', 'A'): {('q0', 'AA')},
                ('q0', 'b', 'A'): {('q1', '')},
                ('q1', 'b', 'A'): {('q1', '')},
                ('q1', '', 'Z'): {('q2', 'Z')}
            },
            start_state='q0',
            accept_states={'q2'}
        )
        
        assert pda.accepts('') == True
        assert pda.accepts('ab') == True
        assert pda.accepts('aabb') == True
        assert pda.accepts('aaabbb') == True
        assert pda.accepts('a') == False
        assert pda.accepts('b') == False
        assert pda.accepts('aab') == False
        assert pda.accepts('abb') == False
    
    def test_stack_replacement(self):
        """Test PDA stack symbol replacement"""
        # PDA that replaces stack symbols
        pda = PDA(
            states={'q0', 'q1'},
            alphabet={'a', 'b'},
            stack_alphabet={'Z', 'A', 'B'},
            transitions={
                ('q0', 'a', 'Z'): {('q0', 'AZ')},
                ('q0', 'b', 'A'): {('q0', 'BA')},
                ('q0', '', 'B'): {('q1', '')}
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert pda.accepts('ab') == True
        assert pda.accepts('aabb') == True
        assert pda.accepts('a') == False
        assert pda.accepts('b') == False


class TestPDAContextFreeLanguages:
    """Test PDA recognition of context-free languages"""
    
    def test_balanced_parentheses(self):
        """Test PDA for balanced parentheses"""
        # L = {w | w has balanced parentheses}
        pda = PDA(
            states={'q0', 'q1'},
            alphabet={'(', ')'},
            stack_alphabet={'Z', 'P'},
            transitions={
                ('q0', '(', 'Z'): {('q0', 'PZ')},
                ('q0', '(', 'P'): {('q0', 'PP')},
                ('q0', ')', 'P'): {('q0', '')},
                ('q0', '', 'Z'): {('q1', 'Z')}
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert pda.accepts('') == True
        assert pda.accepts('()') == True
        assert pda.accepts('(())') == True
        assert pda.accepts('()()') == True
        assert pda.accepts('((()))') == True
        assert pda.accepts('(') == False
        assert pda.accepts(')') == False
        assert pda.accepts('(()') == False
        assert pda.accepts('())') == False
    
    def test_palindromes(self):
        """Test PDA for palindromes with center marker"""
        # L = {wcw^R | w ∈ {a,b}*}
        pda = PDA(
            states={'q0', 'q1', 'q2'},
            alphabet={'a', 'b', 'c'},
            stack_alphabet={'Z', 'A', 'B'},
            transitions={
                ('q0', 'a', 'Z'): {('q0', 'AZ')},
                ('q0', 'a', 'A'): {('q0', 'AA')},
                ('q0', 'a', 'B'): {('q0', 'AB')},
                ('q0', 'b', 'Z'): {('q0', 'BZ')},
                ('q0', 'b', 'A'): {('q0', 'BA')},
                ('q0', 'b', 'B'): {('q0', 'BB')},
                ('q0', 'c', 'Z'): {('q1', 'Z')},
                ('q0', 'c', 'A'): {('q1', 'A')},
                ('q0', 'c', 'B'): {('q1', 'B')},
                ('q1', 'a', 'A'): {('q1', '')},
                ('q1', 'b', 'B'): {('q1', '')},
                ('q1', '', 'Z'): {('q2', 'Z')}
            },
            start_state='q0',
            accept_states={'q2'}
        )
        
        assert pda.accepts('c') == True
        assert pda.accepts('aca') == True
        assert pda.accepts('bcb') == True
        assert pda.accepts('abcba') == True
        assert pda.accepts('aabcbaa') == True
        assert pda.accepts('abc') == False
        assert pda.accepts('abcab') == False
    
    def test_copy_language(self):
        """Test PDA for copy language ww"""
        # L = {ww | w ∈ {a,b}*} with separator
        pda = PDA(
            states={'q0', 'q1', 'q2'},
            alphabet={'a', 'b', '#'},
            stack_alphabet={'Z', 'A', 'B'},
            transitions={
                ('q0', 'a', 'Z'): {('q0', 'AZ')},
                ('q0', 'a', 'A'): {('q0', 'AA')},
                ('q0', 'a', 'B'): {('q0', 'AB')},
                ('q0', 'b', 'Z'): {('q0', 'BZ')},
                ('q0', 'b', 'A'): {('q0', 'BA')},
                ('q0', 'b', 'B'): {('q0', 'BB')},
                ('q0', '#', 'Z'): {('q1', 'Z')},
                ('q0', '#', 'A'): {('q1', 'A')},
                ('q0', '#', 'B'): {('q1', 'B')},
                ('q1', 'a', 'A'): {('q1', '')},
                ('q1', 'b', 'B'): {('q1', '')},
                ('q1', '', 'Z'): {('q2', 'Z')}
            },
            start_state='q0',
            accept_states={'q2'}
        )
        
        assert pda.accepts('#') == True
        assert pda.accepts('a#a') == True
        assert pda.accepts('b#b') == True
        assert pda.accepts('ab#ab') == True
        assert pda.accepts('aa#aa') == True
        assert pda.accepts('a#b') == False
        assert pda.accepts('ab#ba') == False


class TestPDANondeterminism:
    """Test PDA non-deterministic behavior"""
    
    def test_multiple_transitions(self):
        """Test PDA with multiple transitions from same configuration"""
        # Non-deterministic choice between two paths
        pda = PDA(
            states={'q0', 'q1', 'q2'},
            alphabet={'a', 'b'},
            stack_alphabet={'Z'},
            transitions={
                ('q0', 'a', 'Z'): {('q1', 'Z'), ('q2', 'Z')},  # non-deterministic
                ('q1', 'b', 'Z'): {('q1', 'Z')},
                ('q2', 'b', 'Z'): {('q2', 'Z')}
            },
            start_state='q0',
            accept_states={'q1', 'q2'}
        )
        
        assert pda.accepts('ab') == True
        assert pda.accepts('abb') == True
        assert pda.accepts('a') == False
        assert pda.accepts('b') == False
    
    def test_epsilon_transitions(self):
        """Test PDA with epsilon transitions"""
        pda = PDA(
            states={'q0', 'q1', 'q2'},
            alphabet={'a'},
            stack_alphabet={'Z', 'A'},
            transitions={
                ('q0', '', 'Z'): {('q1', 'Z')},  # epsilon transition
                ('q0', 'a', 'Z'): {('q0', 'AZ')},
                ('q1', 'a', 'Z'): {('q2', 'Z')}
            },
            start_state='q0',
            accept_states={'q1', 'q2'}
        )
        
        assert pda.accepts('') == True  # via epsilon transition
        assert pda.accepts('a') == True
        assert pda.accepts('aa') == False


class TestPDAEdgeCases:
    """Test PDA edge cases and error conditions"""
    
    def test_empty_stack_access(self):
        """Test PDA behavior when trying to access empty stack"""
        pda = PDA(
            states={'q0', 'q1'},
            alphabet={'a'},
            stack_alphabet={'Z'},
            transitions={
                ('q0', 'a', 'Z'): {('q1', '')},  # pop Z
                ('q1', 'a', 'Z'): {('q1', 'Z')}  # try to read Z again
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert pda.accepts('a') == True
        assert pda.accepts('aa') == False  # stack should be empty
    
    def test_no_valid_transitions(self):
        """Test PDA with no valid transitions for input"""
        pda = PDA(
            states={'q0'},
            alphabet={'a', 'b'},
            stack_alphabet={'Z'},
            transitions={
                ('q0', 'a', 'Z'): {('q0', 'Z')}
            },
            start_state='q0',
            accept_states={'q0'}
        )
        
        assert pda.accepts('a') == True
        assert pda.accepts('b') == False  # no transition for 'b'
        assert pda.accepts('ab') == False
    
    def test_unreachable_accept_state(self):
        """Test PDA with unreachable accept state"""
        pda = PDA(
            states={'q0', 'q1'},
            alphabet={'a'},
            stack_alphabet={'Z'},
            transitions={
                ('q0', 'a', 'Z'): {('q0', 'Z')}
            },
            start_state='q0',
            accept_states={'q1'}  # unreachable
        )
        
        assert pda.accepts('') == False
        assert pda.accepts('a') == False
        assert pda.accepts('aa') == False


class TestPDASpecialConfigurations:
    """Test special PDA configurations"""
    
    def test_acceptance_by_empty_stack(self):
        """Test PDA that accepts by empty stack"""
        # Note: This tests the concept, actual implementation may vary
        pda = PDA(
            states={'q0'},
            alphabet={'a', 'b'},
            stack_alphabet={'Z', 'A'},
            transitions={
                ('q0', 'a', 'Z'): {('q0', 'AZ')},
                ('q0', 'a', 'A'): {('q0', 'AA')},
                ('q0', 'b', 'A'): {('q0', '')},
                ('q0', 'b', 'Z'): {('q0', '')}
            },
            start_state='q0',
            accept_states={'q0'}
        )
        
        # These tests depend on implementation details
        assert pda.accepts('ab') == True
        assert pda.accepts('aabb') == True
    
    def test_single_state_pda(self):
        """Test PDA with single state"""
        pda = PDA(
            states={'q0'},
            alphabet={'a'},
            stack_alphabet={'Z', 'A'},
            transitions={
                ('q0', 'a', 'Z'): {('q0', 'AZ')},
                ('q0', 'a', 'A'): {('q0', 'AA')}
            },
            start_state='q0',
            accept_states={'q0'}
        )
        
        assert pda.accepts('') == True
        assert pda.accepts('a') == True
        assert pda.accepts('aa') == True
    
    def test_no_stack_operations(self):
        """Test PDA that doesn't modify stack"""
        pda = PDA(
            states={'q0', 'q1'},
            alphabet={'a', 'b'},
            stack_alphabet={'Z'},
            transitions={
                ('q0', 'a', 'Z'): {('q1', 'Z')},
                ('q1', 'b', 'Z'): {('q0', 'Z')}
            },
            start_state='q0',
            accept_states={'q0'}
        )
        
        assert pda.accepts('') == True
        assert pda.accepts('ab') == True
        assert pda.accepts('abab') == True
        assert pda.accepts('a') == False
        assert pda.accepts('b') == False
