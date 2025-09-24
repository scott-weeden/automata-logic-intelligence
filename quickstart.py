#!/usr/bin/env python3
"""
Quickstart demo for the Automata Theory Library
"""

from automata import DFA

def main():
    print("🤖 Automata Theory Library - Quickstart Demo")
    print("=" * 50)
    
    # Create DFA that accepts strings with even number of 0s
    print("\n1. Creating DFA that accepts strings with even number of 0s...")
    
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
    
    # Test cases
    test_cases = [
        ('0011', True),   # Two 0s - should accept
        ('000', False),   # Three 0s - should reject
        ('1111', True),   # Zero 0s - should accept
        ('0101', True),   # Two 0s - should accept
        ('010', True),    # Two 0s - should accept
    ]
    
    print("\n2. Testing the DFA:")
    print("-" * 30)
    
    for input_string, expected in test_cases:
        result = dfa.accepts(input_string)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"Input: '{input_string}' -> {result} {status}")
    
    print("\n🎉 Quickstart demo complete!")
    print("\nNext steps:")
    print("- Run 'pytest' to see test feedback")
    print("- Try 'automata --help' for CLI commands")
    print("- Check out the student exercises in student_solutions/")

if __name__ == "__main__":
    main()
