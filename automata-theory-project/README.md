# Automata Theory Educational Project 🤖

A comprehensive educational library and interactive notebooks for learning automata theory, formal languages, and computational complexity.

## 📁 Project Structure

```
automata-theory-project/
├── src/automata/           # Core automata implementations
├── notebooks/              # Educational Jupyter notebooks
├── tests/                  # Comprehensive test suites
├── templates/              # Web interface templates
└── docs/                   # Additional documentation
```

## 🧠 Core Automata Library (`src/automata/`)

### Finite Automata
- **`dfa.py`** - Deterministic Finite Automaton
  - Recognizes regular languages
  - Complete transition function
  - Used for pattern matching, lexical analysis

- **`nfa.py`** - Non-deterministic Finite Automaton  
  - Multiple transitions per state-symbol pair
  - Epsilon (ε) transitions supported
  - Equivalent power to DFA but more intuitive design

### Context-Free Recognition
- **`pda.py`** - Pushdown Automaton
  - Stack-based memory for context-free languages
  - Recognizes nested structures (parentheses, programming languages)
  - Non-deterministic with epsilon transitions

### Context-Sensitive Recognition  
- **`lba.py`** - Linear Bounded Automaton
  - Turing machine with tape bounded by input length
  - Recognizes context-sensitive languages
  - Tape markers (⊢, ⊣) enforce boundaries

### Universal Computation
- **`tm.py`** - Turing Machine
  - Most powerful computational model
  - Infinite tape with read/write head
  - Demonstrates halting problem and undecidability

### Quantum Computing
- **`qfa.py`** - Quantum Finite Automaton
  - Quantum superposition of states
  - Probabilistic acceptance with interference effects
  - Multiple variants: MeasureOnceQFA, ReversibleQFA

## 📚 Educational Notebooks (`notebooks/`)

### `01_regular_languages.ipynb`
**Concepts**: Regular languages, finite automata, regular expressions
- DFA construction and minimization
- NFA to DFA conversion
- Regular expression equivalence
- Pumping lemma demonstrations
- Closure properties of regular languages

### `02_cfg_pda.ipynb` 
**Concepts**: Context-free grammars, pushdown automata
- CFG parsing and derivations
- PDA construction from CFGs
- Stack-based recognition algorithms
- Context-free pumping lemma
- Inherent ambiguity examples

### `03_turing_decidability.ipynb`
**Concepts**: Turing machines, computability, decidability
- TM programming and execution tracing
- Halting problem demonstration
- Rice's theorem illustrations
- Decidable vs undecidable problems
- Reduction techniques

## 🧪 Test Suites (`tests/`)

### `test_dfa.py`
- Basic DFA functionality
- Transition completeness
- Acceptance/rejection testing
- Edge cases and error handling

### `test_nfa.py` 
- Non-deterministic behavior
- Epsilon closure algorithms
- NFA-specific test patterns
- Complex transition scenarios

### `test_pda.py`
- Stack operation correctness
- Context-free language recognition
- Balanced structures (parentheses, brackets)
- Non-deterministic path exploration

### `test_lba.py`
- Tape boundary enforcement
- Context-sensitive language patterns
- Configuration loop detection
- Trace functionality validation

### `test_qfa.py`
- Quantum state evolution
- Interference effect verification
- Probabilistic acceptance testing
- Amplitude conservation checks

### `test_tm.py`
- Turing machine execution
- Halting behavior analysis
- Infinite loop detection
- Trace output validation

## 🌐 Web Interface (`src/`)

### `web_playground.py`
**Interactive automata testing environment**
- Multi-tab interface for all automata types
- Real-time input testing with visual feedback
- GraphViz integration for state diagram generation
- Example library with common patterns
- Execution tracing for educational debugging

### `cli.py`
**Command-line interface for automata operations**
```bash
# Test automata with various inputs
python cli.py test dfa config.json "input_string"
python cli.py test pda config.json "((()))" --trace
python cli.py test qfa config.json "01" --probability

# Launch interactive modes
python cli.py interactive    # Command-line REPL
python cli.py web --port 5000    # Web interface

# Generate examples and documentation
python cli.py examples pda
python cli.py examples qfa
```

## 📖 Theoretical Concepts Covered

### Chomsky Hierarchy
1. **Type 3 (Regular)**: DFA, NFA, Regular Expressions
2. **Type 2 (Context-Free)**: PDA, Context-Free Grammars  
3. **Type 1 (Context-Sensitive)**: LBA, Context-Sensitive Grammars
4. **Type 0 (Unrestricted)**: Turing Machines, Recursively Enumerable

### Computational Complexity
- **Decidability**: Problems with algorithmic solutions
- **Undecidability**: Halting problem, Rice's theorem
- **Time Complexity**: P, NP, PSPACE relationships
- **Space Complexity**: LOGSPACE, PSPACE hierarchies

### Quantum Computing Foundations
- **Superposition**: Multiple states simultaneously
- **Interference**: Constructive and destructive amplitude effects
- **Measurement**: Probabilistic state collapse
- **Unitary Evolution**: Reversible quantum operations

## 🎓 Educational Usage

### For Students
1. **Progressive Learning**: Start with DFA, advance to quantum automata
2. **Interactive Exploration**: Modify examples and observe behavior
3. **Visual Feedback**: Watch step-by-step execution traces
4. **Immediate Testing**: Verify understanding with comprehensive tests

### For Instructors  
1. **Lecture Integration**: Use notebooks for live demonstrations
2. **Assignment Creation**: Modify examples for homework problems
3. **Assessment Tools**: Automated testing of student implementations
4. **Research Projects**: Extend library with new automata types

## 🔬 Advanced Features

### Execution Tracing
- Step-by-step state transitions
- Tape/stack content visualization  
- Configuration history tracking
- Loop detection algorithms

### Error Handling
- Graceful handling of invalid inputs
- Timeout protection for infinite loops
- Detailed error messages with hints
- Robust parsing of configuration files

### Performance Optimization
- Efficient state space exploration
- Memoization for repeated computations
- Early termination conditions
- Memory-conscious data structures

## 🚀 Getting Started

### Installation
```bash
git clone <repository>
cd automata-theory-project
pip install -r requirements.txt
```

### Quick Test
```bash
# Test all automata types
python -m pytest tests/ -v

# Launch web interface
python src/web_playground.py

# Run interactive CLI
python src/cli.py interactive
```

### Example Usage
```python
from automata import DFA, PDA, QFA

# Create and test a DFA
dfa = DFA(states={'q0','q1'}, alphabet={'0','1'}, ...)
print(dfa.accepts('0110'))

# Explore quantum effects
qfa = create_hadamard_qfa()
prob = qfa.acceptance_probability('01')
```

## 📚 Learning Path

1. **Start**: `01_regular_languages.ipynb` - Master finite automata
2. **Progress**: `02_cfg_pda.ipynb` - Understand context-free languages  
3. **Advanced**: `03_turing_decidability.ipynb` - Explore limits of computation
4. **Experiment**: Web playground - Interactive exploration
5. **Extend**: Implement custom automata types

## 🤝 Contributing

- Add new automata variants
- Expand test coverage
- Create additional notebooks
- Improve visualization tools
- Enhance quantum algorithms

## 📄 License

MIT License - Educational use encouraged

---

**"From the deterministic certainty of DFA to the quantum mysteries of QFA - explore the full spectrum of computational models!"** 🎭⚛️
