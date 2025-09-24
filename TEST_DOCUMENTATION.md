# Test Documentation for Automata Theory Library

## Overview

This document describes the comprehensive test suites generated for the NFA and PDA implementations in the automata theory library.

## Test Files Generated

### 1. `test_nfa.py` - Non-deterministic Finite Automaton Tests

**Current Status**: ❌ All tests failing (NFA.accepts() returns None - not implemented)

#### Test Categories:

**TestNFABasic** - Basic NFA functionality
- `test_nfa_initialization()` - Verifies NFA can be created with correct parameters
- `test_nfa_empty_string()` - Tests empty string acceptance
- `test_nfa_single_character()` - Tests single character transitions

**TestNFANondeterminism** - Non-deterministic behavior
- `test_multiple_transitions()` - Tests multiple transitions from same state
- `test_epsilon_transitions()` - Tests epsilon (empty) transitions

**TestNFAComplexPatterns** - Complex acceptance patterns
- `test_union_pattern()` - Tests union of two patterns (ends in '00' OR contains '11')
- `test_kleene_star_pattern()` - Tests Kleene star pattern (ab)*

**TestNFAEdgeCases** - Edge cases and error conditions
- `test_no_transitions()` - NFA with no transitions
- `test_unreachable_accept_state()` - Unreachable accept states
- `test_dead_end_transitions()` - Transitions to empty set

**TestNFASpecialCases** - Special configurations
- `test_single_state_nfa()` - Single state NFA
- `test_all_states_accepting()` - All states are accepting
- `test_no_accepting_states()` - No accepting states

#### Key Test Patterns:

1. **String ending patterns**: `'01'`, `'00'`
2. **String containing patterns**: `'11'`, `'101'`
3. **Balanced patterns**: `(ab)*`
4. **Empty string handling**
5. **Non-deterministic choice points**
6. **Epsilon transitions**

### 2. `test_pda.py` - Pushdown Automaton Tests

**Current Status**: ❌ Import error (PDA.accepts() returns None - not implemented)

#### Test Categories:

**TestPDABasic** - Basic PDA functionality
- `test_pda_initialization()` - Verifies PDA creation with stack alphabet
- `test_pda_empty_string()` - Tests empty string acceptance
- `test_pda_simple_acceptance()` - Simple acceptance patterns

**TestPDAStackOperations** - Stack manipulation
- `test_push_operation()` - Tests pushing symbols onto stack
- `test_pop_operation()` - Tests popping symbols from stack
- `test_stack_replacement()` - Tests replacing stack symbols

**TestPDAContextFreeLanguages** - Context-free language recognition
- `test_balanced_parentheses()` - L = {w | w has balanced parentheses}
- `test_palindromes()` - L = {wcw^R | w ∈ {a,b}*}
- `test_copy_language()` - L = {ww | w ∈ {a,b}*} with separator

**TestPDANondeterminism** - Non-deterministic behavior
- `test_multiple_transitions()` - Multiple transitions from same configuration
- `test_epsilon_transitions()` - Epsilon transitions in PDA

**TestPDAEdgeCases** - Edge cases
- `test_empty_stack_access()` - Behavior when accessing empty stack
- `test_no_valid_transitions()` - No valid transitions for input
- `test_unreachable_accept_state()` - Unreachable accept states

**TestPDASpecialConfigurations** - Special configurations
- `test_acceptance_by_empty_stack()` - Acceptance by empty stack
- `test_single_state_pda()` - Single state PDA
- `test_no_stack_operations()` - PDA that doesn't modify stack

#### Key Context-Free Languages Tested:

1. **a^n b^n** - Equal number of a's and b's
2. **Balanced parentheses** - Properly nested parentheses
3. **Palindromes with center marker** - wcw^R patterns
4. **Copy language** - ww patterns with separator
5. **Stack-based counting** - Using stack for counting operations

## Implementation Requirements

### For NFA Implementation:

```python
def accepts(self, string):
    """
    Check if string is accepted by NFA using subset construction
    or recursive exploration of all possible paths.
    
    Must handle:
    - Non-deterministic transitions (multiple next states)
    - Epsilon transitions (empty string transitions)
    - Dead-end paths (transitions to empty set)
    """
```

**Key algorithms needed**:
- Epsilon closure computation
- Subset construction for simulation
- Breadth-first or depth-first path exploration

### For PDA Implementation:

```python
def accepts(self, string):
    """
    Check if string is accepted by PDA using stack-based simulation.
    
    Must handle:
    - Stack operations (push, pop, replace)
    - Non-deterministic transitions
    - Epsilon transitions
    - Stack symbol matching
    - Acceptance by final state (and optionally empty stack)
    """
```

**Key algorithms needed**:
- Configuration-based simulation (state, remaining input, stack)
- Stack manipulation operations
- Non-deterministic path exploration
- Acceptance condition checking

## Running the Tests

### Current Status Check:
```bash
# Check NFA tests (will show failures due to unimplemented accepts())
source venv/bin/activate
python -m pytest test_nfa.py -v

# Check PDA tests (will show import working now)
python -m pytest test_pda.py -v
```

### After Implementation:
```bash
# Run all automata tests
python -m pytest test_nfa.py test_pda.py -v

# Run with coverage
python -m pytest test_nfa.py test_pda.py --cov=automata --cov-report=html
```

## Test-Driven Development Approach

1. **Red Phase**: Tests currently fail (methods return None)
2. **Green Phase**: Implement minimal functionality to pass tests
3. **Refactor Phase**: Optimize and clean up implementations

### Recommended Implementation Order:

#### NFA:
1. Basic string acceptance (no epsilon transitions)
2. Non-deterministic transitions (multiple next states)
3. Epsilon transitions and epsilon closure
4. Complex patterns and edge cases

#### PDA:
1. Basic stack operations (push/pop single symbols)
2. Simple context-free languages (a^n b^n)
3. Complex stack manipulations
4. Advanced context-free languages (palindromes, copy language)

## Educational Value

These tests provide:

1. **Immediate Feedback** - Students see exactly what works/fails
2. **Comprehensive Coverage** - All major automata theory concepts
3. **Edge Case Awareness** - Handles boundary conditions
4. **Real Examples** - Concrete instances of theoretical concepts
5. **Progressive Complexity** - From simple to advanced patterns

## Integration with Course

- Tests can be used as **homework assignments**
- **Gradual release** - unlock test categories as concepts are taught
- **Visual feedback** - Red/green test results show progress
- **Debugging aid** - Specific test failures guide implementation

---

**Note**: Both NFA and PDA implementations need to be completed for tests to pass. The test suites are comprehensive and will thoroughly validate the correctness of the implementations.
