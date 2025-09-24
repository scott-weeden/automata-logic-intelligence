# Automata Theory Library 🤖

**Making Undecidability Visceral** - Watch infinite loops spiral in real-time!

A comprehensive Python library for learning and experimenting with automata theory, featuring immediate red/green test feedback and visual execution traces that make theoretical computer science tangible.

## 🚀 Features

### Core Automata Types
- **DFA** (Deterministic Finite Automaton)
- **NFA** (Non-deterministic Finite Automaton with ε-transitions)
- **PDA** (Pushdown Automaton) - **NEW!** ✨
- **LBA** (Linear Bounded Automaton) - **NEW!** ✨
- **Turing Machines** with loop detection and step limits
- **Universal Turing Machine** (UTM) simulator
- **Quantum Finite Automata** (QFA) - **NEW!** ✨

### Advanced Quantum Features ⚛️
- **Measure-once QFA** with quantum superposition
- **Reversible QFA** with unitary evolution
- **Quantum interference** and amplitude manipulation
- **Probabilistic acceptance** with configurable thresholds

### Interactive Web Playground 🌐
- **Real-time testing** of all automata types
- **Visual state diagrams** with GraphViz integration
- **Step-by-step execution traces**
- **Example library** with common patterns
- **Multi-tab interface** for different automata

### Visceral Undecidability
- `--trace` flag to watch execution step-by-step
- `--max-steps` to explore infinite loops safely
- Real-time loop detection and visualization
- Visual spiraling of non-halting computations

### Immediate Feedback
- Red/Green pytest integration
- Instant validation when students write machines
- Comprehensive test suites for all exercises
- Clear error messages with hints

## 📦 Installation

### Quick Install
```bash
pip install automata-theory
```

### Development Install
```bash
git clone https://github.com/scott-weeden/automata-theory
cd automata-theory
pip install -e ".[dev]"
```

## 🎯 Quick Start

### 1. Create Your First DFA
```python
from automata_lib import DFA

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

# Test it
print(dfa.accepts('0011'))  # True (two 0s)
print(dfa.accepts('000'))   # False (three 0s)
```

### 2. Build a Pushdown Automaton
```python
from automata_lib import PDA

# PDA for balanced parentheses
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

print(pda.accepts('((()))'))  # True
print(pda.accepts('(()'))     # False
```

### 3. Explore Quantum Computing
```python
from automata_lib import create_hadamard_qfa

# Quantum automaton with superposition
qfa = create_hadamard_qfa()

# Get acceptance probability
prob = qfa.acceptance_probability('01')
print(f"Acceptance probability: {prob}")

# Trace quantum state evolution
prob, trace = qfa.run_with_trace('01')
for step in trace:
    print(f"Step {step['step']}: {step['probabilities']}")
```

### 4. Test Context-Sensitive Languages
```python
from automata_lib import LBA

# Linear Bounded Automaton
lba = LBA(
    states={'q0', 'q1', 'q2'},
    alphabet={'a', 'b'},
    tape_alphabet={'a', 'b', 'X', '⊢', '⊣'},
    transitions={
        ('q0', 'a'): ('q1', 'X', 'R'),
        ('q1', 'b'): ('q2', 'X', 'L')
    },
    start_state='q0',
    accept_states={'q2'}
)

result, steps, trace = lba.run('ab', trace=True)
print(f"Result: {result}, Steps: {steps}")
```

### 5. Launch Interactive Web Playground
```bash
# Start web interface
python -m automata_lib.web_playground

# Open http://localhost:5000
```

### 6. Watch a Turing Machine Loop
```bash
# Create an infinite loop demonstration
automata undecidability

# Watch it spiral with trace
automata run undecidable.tm.json '111' --trace --max-steps 50

# See loop detection in action
automata run undecidable.tm.json '111' --trace --loop-detect
```

### 7. Get Instant Test Feedback
```bash
# Run tests on your implementation
pytest test_automata.py -v --tb=short

# Watch for red/green feedback
✅ test_student_pda_balanced_parens PASSED
✅ test_student_lba_copy_language PASSED
✅ test_student_qfa_superposition PASSED
❌ test_student_qfa_interference FAILED
   Expected: 0.0 for quantum interference
   Got: 0.25
   Hint: Check negative amplitude cancellation
```

## 🧪 New Automata Types

### Pushdown Automata (PDA)
```python
# Context-free language: a^n b^n
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
```

### Linear Bounded Automata (LBA)
```python
# Context-sensitive language: a^n b^n c^n
lba = LBA(
    states={'q0', 'q1', 'q2', 'q3'},
    alphabet={'a', 'b', 'c'},
    tape_alphabet={'a', 'b', 'c', 'X', 'Y', 'Z', '⊢', '⊣'},
    transitions={
        ('q0', 'a'): ('q1', 'X', 'R'),
        ('q1', 'b'): ('q2', 'Y', 'R'),
        ('q2', 'c'): ('q3', 'Z', 'L'),
        # ... more transitions for full implementation
    },
    start_state='q0',
    accept_states={'q3'}
)
```

### Quantum Finite Automata (QFA)
```python
import math

# Quantum superposition
h = 1/math.sqrt(2)
qfa = QFA(
    states={'q0', 'q1'},
    alphabet={'0', '1'},
    transitions={
        ('q0', '0'): [('q0', h), ('q1', h)],
        ('q0', '1'): [('q0', h), ('q1', -h)],
        ('q1', '0'): [('q0', h), ('q1', -h)],
        ('q1', '1'): [('q0', -h), ('q1', h)]
    },
    start_state='q0',
    accept_states={'q1'},
    threshold=0.5
)

# Quantum interference
prob = qfa.acceptance_probability('01')  # May be 0 due to interference!
```

## 🌐 Interactive Web Playground

Launch the web interface to test automata visually:

```bash
cd automata-theory-project/src
python web_playground.py
```

Features:
- **Multi-tab interface** for different automata types
- **Real-time testing** with immediate feedback
- **GraphViz visualization** generation
- **Example library** with common patterns
- **Execution tracing** for debugging
- **Quantum state visualization** for QFA

## 📚 Student Exercises

### Exercise Structure
```
student_solutions/
├── dfa_even_zeros.json       # Exercise 1
├── dfa_divisible_3.json      # Exercise 2
├── nfa_ends_01.json          # Exercise 3
├── pda_balanced_parens.json  # Exercise 4 - NEW!
├── pda_copy_language.json    # Exercise 5 - NEW!
├── lba_palindromes.json      # Exercise 6 - NEW!
├── qfa_superposition.json    # Exercise 7 - NEW!
├── tm_binary_increment.json  # Exercise 8
└── tm_loop_demo.json         # Exercise 9
```

### New PDA Exercise
```json
{
  "type": "PDA",
  "states": ["q0", "q1", "q2"],
  "alphabet": ["a", "b"],
  "stack_alphabet": ["Z", "A"],
  "transitions": {
    "('q0', 'a', 'Z')": [["q0", "AZ"]],
    "('q0', 'a', 'A')": [["q0", "AA"]],
    "('q0', 'b', 'A')": [["q1", ""]],
    "('q1', 'b', 'A')": [["q1", ""]],
    "('q1', '', 'Z')": [["q2", "Z"]]
  },
  "start_state": "q0",
  "accept_states": ["q2"]
}
```

## 🔬 Exploring Advanced Topics

### Quantum Computing Concepts
```python
from automata_lib import ReversibleQFA

# Unitary evolution with Hadamard gates
hadamard = [[1/√2, 1/√2], [1/√2, -1/√2]]

rqfa = ReversibleQFA(
    states={'q0', 'q1'},
    alphabet={'H'},
    unitary_transitions={'H': hadamard},
    start_state='q0',
    accept_states={'q1'}
)

# Two Hadamards return to original state
prob = rqfa.acceptance_probability('HH')  # Should be 0
```

### Context-Sensitive Languages
```python
# LBA for {ww | w ∈ {a,b}*}
lba_copy = LBA(
    states={'q0', 'q1', 'q2', 'q3'},
    alphabet={'a', 'b'},
    tape_alphabet={'a', 'b', 'X', 'Y', '⊢', '⊣'},
    transitions={
        # Mark and compare symbols
        ('q0', 'a'): ('q1', 'X', 'R'),
        ('q0', 'b'): ('q2', 'Y', 'R'),
        # ... implementation for copy language
    },
    start_state='q0',
    accept_states={'q3'}
)
```

### Undecidability Demonstrations
```python
# Halting problem visualization
tm_halt = create_halting_problem_tm()

# Watch it loop infinitely
result, reason, steps = tm_halt.run('', max_steps=100, trace=True)
print(f"Reason: {reason.value}")  # INFINITE_LOOP_DETECTED

# Visualize the spiral
tm_halt.visualize_execution('', max_steps=50, animate=True)
```

## 🛠️ Enhanced CLI Commands

### Core Commands
```bash
# Test any automaton type
automata test pda config.json "(())" --trace
automata test lba config.json "aabbcc" --max-steps 1000
automata test qfa config.json "01" --trace

# Interactive web playground
automata web --port 8080

# Generate examples
automata examples pda
automata examples qfa

# Visualize automata
automata visualize dfa config.json --output diagram.png
```

### New Options
- `--quantum-trace`: Show quantum state evolution
- `--stack-trace`: Show PDA stack operations
- `--tape-trace`: Show LBA tape modifications
- `--probability`: Show QFA acceptance probabilities

## 📊 Advanced Visualization

### Quantum State Evolution
```python
qfa = create_hadamard_qfa()
prob, trace = qfa.run_with_trace('010')

# Visualize quantum amplitudes
import matplotlib.pyplot as plt
steps = [t['step'] for t in trace]
probs = [t['probabilities'] for t in trace]

plt.plot(steps, probs)
plt.title('Quantum State Evolution')
plt.show()
```

### PDA Stack Visualization
```python
pda = create_balanced_parens_pda()
result, trace = pda.run_with_trace('((()))')

for step in trace:
    print(f"Stack: {step['stack']}, Input: {step['remaining_input']}")
```

## 🎓 Educational Usage

### For Instructors
1. **Advanced Assignments**: PDA, LBA, and QFA exercises
2. **Quantum Computing**: Introduction through QFA
3. **Context-Sensitive Languages**: LBA demonstrations
4. **Interactive Lectures**: Web playground for live demos

### For Students
1. **Gradual Complexity**: From DFA to Quantum Automata
2. **Visual Learning**: See stack operations and quantum states
3. **Immediate Feedback**: Know instantly if implementations work
4. **Research Projects**: Implement custom quantum algorithms

## 🔗 Integration with Course

### Docker Support
```bash
# Build with enhanced course Docker image
docker run -v $(pwd):/app scott-weeden/latex automata test pda
```

### Continuous Testing
```yaml
# Enhanced GitHub Actions workflow
name: Test All Automata
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pip install -e ".[quantum]"
      - run: pytest tests/ -v
      - run: python -m automata_lib.cli examples
```

## 🐛 Debugging Tips

### Common Issues

1. **PDA stack operations**: Check push/pop order and epsilon transitions
2. **LBA tape boundaries**: Ensure proper marker handling
3. **QFA normalization**: Verify amplitude conservation
4. **Quantum interference**: Check for negative amplitude cancellation

### Debug Mode
```python
# Enable comprehensive debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Trace all operations
pda.accepts("input", trace=True, debug=True)
lba.run("input", trace=True, max_steps=100)
qfa.run_with_trace("input")
```

## 📚 Resources

- **Documentation**: https://automata-theory.readthedocs.io
- **Course Materials**: See `/course-content`
- **Quantum Computing**: Introduction via QFA
- **Video Tutorials**: YouTube playlist with quantum examples
- **Discord**: Join our study group

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- More quantum algorithms (Grover, Shor concepts)
- Advanced LBA examples
- PDA optimization algorithms
- Web playground enhancements
- Quantum visualization tools

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by Sipser's "Introduction to the Theory of Computation"
- Quantum computing concepts from Nielsen & Chuang
- Built for CS Theory courses worldwide
- Special thanks to students who tested quantum features

## 🚧 Roadmap

### Recently Added ✅
- [x] Pushdown Automata (PDA)
- [x] Linear Bounded Automata (LBA)
- [x] Quantum Finite Automata (QFA)
- [x] Interactive web playground
- [x] Advanced CLI commands
- [x] Comprehensive test suites

### Coming Soon
- [ ] Quantum Turing Machines
- [ ] Probabilistic Automata
- [ ] Tree Automata
- [ ] Timed Automata
- [ ] Advanced quantum algorithms
- [ ] VR/AR visualization
- [ ] LLM integration for hints

## 💡 Tips for Success

1. **Start Simple**: Master DFA before quantum automata
2. **Test Often**: Use pytest after each implementation
3. **Trace Everything**: Watch executions with `--trace`
4. **Explore Quantum**: Understand superposition and interference
5. **Use Web Playground**: Visual feedback accelerates learning
6. **Study Examples**: Learn from provided implementations

---

**Remember**: From the deterministic certainty of DFA to the probabilistic mysteries of quantum automata, every computation tells a story. Welcome to the expanded universe of theoretical computer science! 🎭⚛️