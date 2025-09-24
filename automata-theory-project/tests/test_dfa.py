"""Tests for DFA implementation"""

import pytest
from automata import DFA

def test_dfa_accepts():
    """Test basic DFA acceptance"""
    # DFA that accepts strings with even number of 0s
    dfa = DFA(
        states={'q0', 'q1'},
        alphabet={'0', '1'},
        transitions={
            ('q0', '0'): 'q1',
            ('q0', '1'): 'q0',
            ('q1', '0'): 'q0',
            ('q1', '1'): 'q1',
        },
        start_state='q0',
        accept_states={'q0'}
    )
    
    assert dfa.accepts('') == True
    assert dfa.accepts('00') == True
    assert dfa.accepts('0') == False
