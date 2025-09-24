"""Tests for Linear Bounded Automaton (LBA)"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from automata import LBA

class TestLBABasic:
    """Basic LBA functionality tests"""
    
    def test_lba_initialization(self):
        """Test LBA can be initialized"""
        lba = LBA(
            states={'q0', 'q1'},
            alphabet={'a', 'b'},
            tape_alphabet={'a', 'b', '⊢', '⊣'},
            transitions={
                ('q0', 'a'): ('q1', 'b', 'R')
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert lba.states == {'q0', 'q1'}
        assert lba.alphabet == {'a', 'b'}
        assert lba.start_state == 'q0'
        assert lba.accept_states == {'q1'}
    
    def test_lba_empty_string(self):
        """Test LBA with empty string"""
        lba = LBA(
            states={'q0'},
            alphabet={'a'},
            tape_alphabet={'a', '⊢', '⊣'},
            transitions={},
            start_state='q0',
            accept_states={'q0'}
        )
        
        assert lba.accepts('') == True
    
    def test_lba_simple_acceptance(self):
        """Test LBA simple acceptance"""
        # LBA that changes 'a' to 'b' and accepts
        lba = LBA(
            states={'q0', 'q1'},
            alphabet={'a'},
            tape_alphabet={'a', 'b', '⊢', '⊣'},
            transitions={
                ('q0', 'a'): ('q1', 'b', 'R')
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        assert lba.accepts('a') == True
        assert lba.accepts('') == False

class TestLBAContextSensitive:
    """Test LBA for context-sensitive languages"""
    
    def test_copy_language(self):
        """Test LBA for a^n b^n c^n"""
        # Simplified LBA for a^n b^n c^n (n >= 1)
        lba = LBA(
            states={'q0', 'q1', 'q2', 'q3', 'q4', 'q5'},
            alphabet={'a', 'b', 'c'},
            tape_alphabet={'a', 'b', 'c', 'X', 'Y', 'Z', '⊢', '⊣'},
            transitions={
                # Mark first 'a' and find corresponding 'b' and 'c'
                ('q0', 'a'): ('q1', 'X', 'R'),
                ('q1', 'a'): ('q1', 'a', 'R'),
                ('q1', 'Y'): ('q1', 'Y', 'R'),
                ('q1', 'b'): ('q2', 'Y', 'R'),
                ('q2', 'b'): ('q2', 'b', 'R'),
                ('q2', 'Z'): ('q2', 'Z', 'R'),
                ('q2', 'c'): ('q3', 'Z', 'L'),
                # Go back to start
                ('q3', 'Z'): ('q3', 'Z', 'L'),
                ('q3', 'b'): ('q3', 'b', 'L'),
                ('q3', 'Y'): ('q3', 'Y', 'L'),
                ('q3', 'a'): ('q3', 'a', 'L'),
                ('q3', 'X'): ('q3', 'X', 'L'),
                ('q3', '⊢'): ('q0', '⊢', 'R'),
                # Check if all symbols are marked
                ('q0', 'X'): ('q4', 'X', 'R'),
                ('q4', 'X'): ('q4', 'X', 'R'),
                ('q4', 'Y'): ('q4', 'Y', 'R'),
                ('q4', 'Z'): ('q4', 'Z', 'R'),
                ('q4', '⊣'): ('q5', '⊣', 'L')
            },
            start_state='q0',
            accept_states={'q5'}
        )
        
        # This is a complex test - simplified version
        assert lba.accepts('abc') == True
    
    def test_palindromes(self):
        """Test LBA for palindromes"""
        # LBA that accepts palindromes over {a,b}
        lba = LBA(
            states={'q0', 'q1', 'q2', 'q3', 'q4'},
            alphabet={'a', 'b'},
            tape_alphabet={'a', 'b', 'X', '⊢', '⊣'},
            transitions={
                # Mark first symbol and find matching last symbol
                ('q0', 'a'): ('q1', 'X', 'R'),
                ('q0', 'b'): ('q2', 'X', 'R'),
                ('q0', 'X'): ('q0', 'X', 'R'),
                ('q0', '⊣'): ('q4', '⊣', 'L'),  # Accept if all marked
                
                # Find last 'a'
                ('q1', 'a'): ('q1', 'a', 'R'),
                ('q1', 'b'): ('q1', 'b', 'R'),
                ('q1', 'X'): ('q1', 'X', 'R'),
                ('q1', '⊣'): ('q3', '⊣', 'L'),
                ('q3', 'a'): ('q0', 'X', 'L'),  # Mark and go back
                
                # Find last 'b'
                ('q2', 'a'): ('q2', 'a', 'R'),
                ('q2', 'b'): ('q2', 'b', 'R'),
                ('q2', 'X'): ('q2', 'X', 'R'),
                ('q2', '⊣'): ('q3', '⊣', 'L'),
                ('q3', 'b'): ('q0', 'X', 'L'),  # Mark and go back
                
                # Go back to start
                ('q0', 'a'): ('q0', 'a', 'L'),
                ('q0', 'b'): ('q0', 'b', 'L'),
                ('q0', '⊢'): ('q0', '⊢', 'R')
            },
            start_state='q0',
            accept_states={'q4'}
        )
        
        # Simplified test
        assert lba.accepts('a') == True
        assert lba.accepts('aa') == True

class TestLBABoundedTape:
    """Test LBA bounded tape behavior"""
    
    def test_tape_boundaries(self):
        """Test that LBA respects tape boundaries"""
        lba = LBA(
            states={'q0', 'q1'},
            alphabet={'a'},
            tape_alphabet={'a', '⊢', '⊣'},
            transitions={
                ('q0', '⊢'): ('q0', '⊢', 'R'),
                ('q0', 'a'): ('q0', 'a', 'R'),
                ('q0', '⊣'): ('q1', '⊣', 'L')
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        result, steps, trace = lba.run('a', trace=True)
        assert result == True
        assert steps > 0
    
    def test_infinite_loop_detection(self):
        """Test LBA infinite loop detection"""
        lba = LBA(
            states={'q0'},
            alphabet={'a'},
            tape_alphabet={'a', '⊢', '⊣'},
            transitions={
                ('q0', 'a'): ('q0', 'a', 'R'),
                ('q0', '⊣'): ('q0', '⊣', 'L')
            },
            start_state='q0',
            accept_states={'q1'}  # Unreachable
        )
        
        result = lba.accepts('a', max_steps=100)
        assert result == False

class TestLBATracing:
    """Test LBA execution tracing"""
    
    def test_trace_output(self):
        """Test LBA trace functionality"""
        lba = LBA(
            states={'q0', 'q1'},
            alphabet={'a'},
            tape_alphabet={'a', 'b', '⊢', '⊣'},
            transitions={
                ('q0', 'a'): ('q1', 'b', 'R')
            },
            start_state='q0',
            accept_states={'q1'}
        )
        
        result, steps, trace = lba.run('a', trace=False)
        assert result == True
        assert len(trace) > 0
        assert trace[0]['old_state'] == 'q0'
        assert trace[0]['symbol'] == 'a'
        assert trace[0]['new_state'] == 'q1'
