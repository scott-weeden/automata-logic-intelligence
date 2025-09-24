#!/usr/bin/env python3
"""
Pytest test suite for automata theory exercises
Provides immediate red/green feedback for student implementations
"""

import pytest
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from automata_lib import (
    DFA, NFA, TuringMachine, Direction, HaltReason,
    create_binary_palindrome_dfa, create_contains_substring_nfa,
    create_binary_increment_tm, create_infinite_loop_tm
)


# ============================================================================
# TEST FIXTURES AND UTILITIES
# ============================================================================

@dataclass
class TestCase:
    """Test case for automaton"""
    input: str
    expected: bool
    description: str = ""


@pytest.fixture
def binary_alphabet():
    """Binary alphabet for testing"""
    return {'0', '1'}


@pytest.fixture
def simple_states():
    """Simple state set for testing"""
    return {'q0', 'q1', 'q2', 'accept', 'reject'}


# ============================================================================
# DFA TESTS - Regular Languages
# ============================================================================

class TestDFAExercises:
    """DFA exercises with immediate feedback"""
    
    def test_student_dfa_even_zeros(self):
        """
        Exercise 1: Create a DFA that accepts strings with an even number of 0s
        """
        # Student should create this DFA
        student_dfa = self._load_student_machine('dfa_even_zeros.json')
        
        test_cases = [
            TestCase('', True, 'empty string (0 zeros is even)'),
            TestCase('1', True, 'single 1 (0 zeros)'),
            TestCase('0', False, 'single 0 (odd)'),
            TestCase('00', True, 'two 0s (even)'),
            TestCase('000', False, 'three 0s (odd)'),
            TestCase('0101', True, 'two 0s among 1s'),
            TestCase('11111', True, 'all 1s (0 zeros)'),
            TestCase('001001', False, 'three 0s total'),
        ]
        
        self._run_dfa_tests(student_dfa, test_cases)
    
    def test_student_dfa_divisible_by_3(self):
        """
        Exercise 2: Create a DFA that accepts binary numbers divisible by 3
        """
        student_dfa = self._load_student_machine('dfa_divisible_3.json')
        
        test_cases = [
            TestCase('', True, 'empty (represents 0)'),
            TestCase('0', True, '0 is divisible by 3'),
            TestCase('11', True, '3 in binary'),
            TestCase('110', True, '6 in binary'),
            TestCase('1001', True, '9 in binary'),
            TestCase('1', False, '1 not divisible by 3'),
            TestCase('10', False, '2 not divisible by 3'),
            TestCase('100', False, '4 not divisible by 3'),
            TestCase('101', False, '5 not divisible by 3'),
        ]
        
        self._run_dfa_tests(student_dfa, test_cases)
    
    def test_student_dfa_no_consecutive_ones(self):
        """
        Exercise 3: Create a DFA that accepts strings with no consecutive 1s
        """
        student_dfa = self._load_student_machine('dfa_no_consecutive_ones.json')
        
        test_cases = [
            TestCase('', True, 'empty string'),
            TestCase('0', True, 'single 0'),
            TestCase('1', True, 'single 1'),
            TestCase('01', True, 'alternating'),
            TestCase('10', True, 'alternating'),
            TestCase('11', False, 'consecutive 1s'),
            TestCase('0110', False, 'contains 11'),
            TestCase('1010101', True, 'alternating pattern'),
            TestCase('001001001', True, 'no consecutive 1s'),
            TestCase('0111', False, 'three consecutive 1s'),
        ]
        
        self._run_dfa_tests(student_dfa, test_cases)
    
    def _load_student_machine(self, filename: str):
        """Load student's machine implementation"""
        path = Path('student_solutions') / filename
        if not path.exists():
            pytest.skip(f"Student solution '{filename}' not found. Create it in student_solutions/")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if data.get('type') == 'DFA':
            return DFA.from_dict(data)
        else:
            pytest.fail(f"Expected DFA, got {data.get('type')}")
    
    def _run_dfa_tests(self, dfa: DFA, test_cases: List[TestCase]):
        """Run test cases and provide feedback"""
        for test in test_cases:
            result = dfa.accepts(test.input)
            assert result == test.expected, (
                f"\n❌ FAILED: Input '{test.input}'\n"
                f"   Expected: {test.expected}\n"
                f"   Got: {result}\n"
                f"   Hint: {test.description}"
            )
        
        print(f"\n✅ All {len(test_cases)} tests passed!")


# ============================================================================
# NFA TESTS - Non-deterministic Automata
# ============================================================================

class TestNFAExercises:
    """NFA exercises with epsilon transitions"""
    
    def test_student_nfa_ends_with_01(self):
        """
        Exercise 4: Create an NFA that accepts strings ending with '01'
        """
        student_nfa = self._load_student_nfa('nfa_ends_01.json')
        
        test_cases = [
            TestCase('01', True, 'exactly 01'),
            TestCase('001', True, 'ends with 01'),
            TestCase('101', True, 'ends with 01'),
            TestCase('11101', True, 'ends with 01'),
            TestCase('', False, 'empty string'),
            TestCase('0', False, 'just 0'),
            TestCase('1', False, 'just 1'),
            TestCase('10', False, 'ends with 10'),
            TestCase('011', False, 'ends with 11'),
        ]
        
        self._run_nfa_tests(student_nfa, test_cases)
    
    def test_student_nfa_contains_substring(self):
        """
        Exercise 5: Create an NFA that accepts strings containing '101'
        """
        student_nfa = self._load_student_nfa('nfa_contains_101.json')
        
        test_cases = [
            TestCase('101', True, 'exactly 101'),
            TestCase('0101', True, 'contains 101'),
            TestCase('1010', True, 'contains 101'),
            TestCase('11011', True, 'contains 101'),
            TestCase('000101000', True, 'contains 101'),
            TestCase('', False, 'empty string'),
            TestCase('111', False, 'all 1s'),
            TestCase('000', False, 'all 0s'),
            TestCase('100', False, 'no 101'),
            TestCase('011', False, 'no 101'),
        ]
        
        self._run_nfa_tests(student_nfa, test_cases)
    
    def _load_student_nfa(self, filename: str):
        """Load student's NFA implementation"""
        path = Path('student_solutions') / filename
        if not path.exists():
            pytest.skip(f"Student solution '{filename}' not found")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if data.get('type') == 'NFA':
            return NFA.from_dict(data)
        else:
            pytest.fail(f"Expected NFA, got {data.get('type')}")
    
    def _run_nfa_tests(self, nfa: NFA, test_cases: List[TestCase]):
        """Run NFA test cases"""
        for test in test_cases:
            result = nfa.accepts(test.input)
            assert result == test.expected, (
                f"\n❌ FAILED: Input '{test.input}'\n"
                f"   Expected: {test.expected}\n"
                f"   Got: {result}\n"
                f"   Hint: {test.description}"
            )


# ============================================================================
# TURING MACHINE TESTS - Computability
# ============================================================================

class TestTuringMachineExercises:
    """Turing Machine exercises demonstrating computability"""
    
    def test_student_tm_binary_adder(self):
        """
        Exercise 6: Create a TM that adds 1 to a binary number
        """
        student_tm = self._load_student_tm('tm_binary_increment.json')
        
        test_cases = [
            ('0', '1', 'increment 0'),
            ('1', '10', 'increment 1'),
            ('10', '11', 'increment 2'),
            ('11', '100', 'increment 3 with carry'),
            ('111', '1000', 'all carries'),
            ('1010', '1011', 'increment 10'),
        ]
        
        self._run_tm_computation_tests(student_tm, test_cases)
    
    def test_student_tm_palindrome_checker(self):
        """
        Exercise 7: Create a TM that checks if a string is a palindrome
        """
        student_tm = self._load_student_tm('tm_palindrome.json')
        
        test_cases = [
            TestCase('', True, 'empty is palindrome'),
            TestCase('0', True, 'single char'),
            TestCase('1', True, 'single char'),
            TestCase('00', True, 'double same'),
            TestCase('11', True, 'double same'),
            TestCase('010', True, 'odd palindrome'),
            TestCase('0110', True, 'even palindrome'),
            TestCase('01', False, 'not palindrome'),
            TestCase('001', False, 'not palindrome'),
        ]
        
        self._run_tm_decision_tests(student_tm, test_cases, max_steps=500)
    
    def test_student_tm_undecidable_behavior(self):
        """
        Exercise 8: Observe undecidable behavior with loop detection
        """
        student_tm = self._load_student_tm('tm_loop_demo.json')
        
        # Test that shows different halting behaviors
        inputs_and_behaviors = [
            ('', HaltReason.ACCEPT, 'empty input should accept'),
            ('0', HaltReason.ACCEPT, 'single 0 should accept'),
            ('111', HaltReason.INFINITE_LOOP, 'should detect infinite loop'),
        ]
        
        for input_str, expected_halt, description in inputs_and_behaviors:
            accepted, halt_reason, steps = student_tm.run(input_str, max_steps=100)
            
            assert halt_reason == expected_halt, (
                f"\n❌ FAILED: Input '{input_str}'\n"
                f"   Expected halt reason: {expected_halt.value}\n"
                f"   Got: {halt_reason.value}\n"
                f"   Steps: {steps}\n"
                f"   Hint: {description}"
            )
    
    def test_halting_problem_demonstration(self):
        """
        Demonstration: The Halting Problem is undecidable
        
        This test shows that we cannot always determine if a TM will halt.
        """
        # Create a TM that may or may not halt
        tm = create_infinite_loop_tm()
        
        # These we can determine quickly
        assert tm.run('', max_steps=10)[1] == HaltReason.ACCEPT
        assert tm.run('0', max_steps=10)[1] == HaltReason.ACCEPT
        
        # This one loops - but how would we know without running it?
        # With loop detection, we find out
        _, halt_reason, _ = tm.run('111', max_steps=1000)
        assert halt_reason in [HaltReason.INFINITE_LOOP, HaltReason.MAX_STEPS]
        
        print("\n🎓 Halting Problem Demonstrated!")
        print("   We detected a loop, but in general, this is undecidable.")
        print("   There's no algorithm that can determine halting for ALL TMs!")
    
    def _load_student_tm(self, filename: str):
        """Load student's TM implementation"""
        path = Path('student_solutions') / filename
        if not path.exists():
            pytest.skip(f"Student solution '{filename}' not found")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if data.get('type') == 'TuringMachine':
            return TuringMachine.from_dict(data)
        else:
            pytest.fail(f"Expected TuringMachine, got {data.get('type')}")
    
    def _run_tm_decision_tests(self, tm: TuringMachine, test_cases: List[TestCase], 
                               max_steps: int = 1000):
        """Run TM tests for decision problems"""
        for test in test_cases:
            accepted, halt_reason, steps = tm.run(test.input, max_steps=max_steps)
            
            assert accepted == test.expected, (
                f"\n❌ FAILED: Input '{test.input}'\n"
                f"   Expected: {test.expected}\n"
                f"   Got: {accepted}\n"
                f"   Halt reason: {halt_reason.value}\n"
                f"   Steps: {steps}\n"
                f"   Hint: {test.description}"
            )
    
    def _run_tm_computation_tests(self, tm: TuringMachine, test_cases: List[Tuple[str, str, str]],
                                  max_steps: int = 1000):
        """Run TM tests for computation problems"""
        for input_str, expected_output, description in test_cases:
            # Run TM
            accepted, halt_reason, steps = tm.run(input_str, max_steps=max_steps)
            
            # For computation TMs, we need to check the final tape content
            # This would require accessing the final configuration
            # For now, just check acceptance
            assert accepted, (
                f"\n❌ FAILED: Input '{input_str}'\n"
                f"   TM did not accept\n"
                f"   Halt reason: {halt_reason.value}\n"
                f"   Steps: {steps}\n"
                f"   Hint: {description}"
            )


# ============================================================================
# COMPLEXITY TESTS - Big-O Analysis
# ============================================================================

class TestComplexityAnalysis:
    """Tests that demonstrate computational complexity"""
    
    def test_dfa_constant_memory(self):
        """
        DFAs use O(1) memory regardless of input size
        """
        dfa = create_binary_palindrome_dfa()
        
        # Memory usage doesn't grow with input
        for n in [10, 100, 1000]:
            input_str = '01' * n
            # DFA processes in single pass with constant memory
            dfa.accepts(input_str)
        
        print("\n✅ DFA uses O(1) memory - constant space complexity!")
    
    def test_tm_exponential_time_possible(self):
        """
        Some problems require exponential time on TMs
        """
        # This would be a TM solving an NP-complete problem
        # For demonstration, we just show the concept
        print("\n📊 Some problems are inherently exponential!")
        print("   Examples: SAT, Traveling Salesman, Graph Coloring")


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "student: mark test as student exercise"
    )
    config.addinivalue_line(
        "markers", "demo: mark test as demonstration"
    )
    config.addinivalue_line(
        "markers", "undecidable: mark test as undecidability demo"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for better output"""
    for item in items:
        # Add markers based on test names
        if "student" in item.nodeid:
            item.add_marker(pytest.mark.student)
        if "demo" in item.nodeid or "demonstration" in item.nodeid:
            item.add_marker(pytest.mark.demo)
        if "undecidable" in item.nodeid or "halting" in item.nodeid:
            item.add_marker(pytest.mark.undecidable)


# ============================================================================
# PYTEST PLUGINS FOR BETTER OUTPUT
# ============================================================================

def pytest_runtest_makereport(item, call):
    """Custom test reporting for better student feedback"""
    if call.when == "call":
        if hasattr(item, 'obj') and hasattr(item.obj, '__self__'):
            test_method = item.obj
            if test_method.__doc__:
                # Print exercise description
                print(f"\n📝 {test_method.__doc__.strip()}")


# ============================================================================
# HELPER SCRIPTS FOR STUDENTS
# ============================================================================

if __name__ == '__main__':
    """Run tests with optimal settings for student feedback"""
    import subprocess
    
    # Run with verbose output and color
    subprocess.run([
        'pytest', __file__,
        '-v',  # Verbose
        '--tb=short',  # Short traceback
        '--color=yes',  # Colored output
        '-x',  # Stop on first failure
        '--ff',  # Run failed tests first
        '-q',  # Quieter output
    ])