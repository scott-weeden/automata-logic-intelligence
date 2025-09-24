# Automata Theory Library 🤖

**Making Undecidability Visceral** - Watch infinite loops spiral in real-time!

A comprehensive Python library for learning and experimenting with automata theory, featuring immediate red/green test feedback and visual execution traces that make theoretical computer science tangible.

## 🚀 Features

### Core Automata Types
- **DFA** (Deterministic Finite Automaton)
- **NFA** (Non-deterministic Finite Automaton with ε-transitions)
- **Turing Machines** with loop detection and step limits
- **Universal Turing Machine** (UTM) simulator

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

### 2. Watch a Turing Machine Loop
```bash
# Create an infinite loop demonstration
automata undecidability

# Watch it spiral with trace
automata run undecidable.tm.json '111' --trace --max-steps 50

# See loop detection in action
automata run undecidable.tm.json '111' --trace --loop-detect
```

### 3. Get Instant Test Feedback
```bash
# Run tests on your implementation
pytest test_automata.py -v --tb=short

# Watch for red/green feedback
✅ test_student_dfa_even_zeros PASSED
❌ test_student_dfa_divisible_by_3 FAILED
   Expected: True for input '11'
   Got: False
   Hint: Binary 11 is decimal 3
```

## 📚 Student Exercises

### Exercise Structure
```
student_solutions/
├── dfa_even_zeros.json       # Exercise 1
├── dfa_divisible_3.json      # Exercise 2
├── dfa_no_consecutive_ones.json  # Exercise 3
├── nfa_ends_01.json          # Exercise 4
├── nfa_contains_101.json     # Exercise 5
├── tm_binary_increment.json  # Exercise 6
├── tm_palindrome.json        # Exercise 7
└── tm_loop_demo.json         # Exercise 8
```

### Example Exercise File
```json
{
  "type": "DFA",
  "states": ["q0", "q1"],
  "alphabet": ["0", "1"],
  "transitions": {
    "q0,0": "q1",
    "q0,1": "q0",
    "q1,0": "q0",
    "q1,1": "q1"
  },
  "start_state": "q0",
  "accept_states": ["q0"]
}
```

## 🔬 Exploring Undecidability

### The Halting Problem Made Real
```python
from automata_lib import TuringMachine, HaltReason

# Create a TM that might not halt
tm = create_infinite_loop_tm()

# Run with different inputs
accepted, reason, steps = tm.run('', max_steps=100)
print(f"Empty: {reason.value}")  # ACCEPT

accepted, reason, steps = tm.run('111', max_steps=1000)
print(f"'111': {reason.value}")  # INFINITE_LOOP_DETECTED

# Watch it live
tm.run('111', max_steps=50, trace=True, delay=0.1)
```

### Trace Output Example
```
TURING MACHINE EXECUTION TRACE
========================================
Step    0
State: q0          Head:   0
Tape:  [1] 1 1
δ(q0, 1) → (q1, X, R)

Step    1
State: q1          Head:   1
Tape:  X [1] 1
δ(q1, 1) → (q2, X, R)

Step    2
State: q2          Head:   2
Tape:  X X [1]
δ(q2, 1) → (q0, X, L)

... (spiraling continues)

⏱ MAX STEPS (50) exceeded - possible infinite loop

Loop Analysis:
Most Visited States
╭────────┬────────╮
│ State  │ Visits │
├────────┼────────┤
│ q0     │   17   │
│ q1     │   16   │
│ q2     │   16   │
╰────────┴────────╯

💡 Tip: Try running with --max-steps 100 to explore further
```

## 🧪 Testing Your Machines

### Running Tests
```bash
# Run all tests
pytest

# Run specific exercise tests
pytest -k "dfa_even_zeros"

# Run with verbose output
pytest -v --tb=short --color=yes

# Stop on first failure (recommended for exercises)
pytest -x --ff
```

### Test Output
```
📝 Exercise 1: Create a DFA that accepts strings with an even number of 0s

test_automata.py::TestDFAExercises::test_student_dfa_even_zeros 
✅ All 8 tests passed!

📝 Exercise 6: Create a TM that adds 1 to a binary number

test_automata.py::TestTuringMachineExercises::test_student_tm_binary_adder
❌ FAILED: Input '111'
   Expected: 1000
   Got: 0111
   Hint: Handle carry propagation correctly
```

## 📊 Visualization

### Generate State Diagrams
```bash
# Create visual representation of your automaton
automata visualize machine.json

# Outputs: machine.png with GraphViz rendering
```

### Interactive Web UI
```bash
# Start web interface
automata-web

# Open http://localhost:5000
```

## 🛠️ CLI Commands

### Core Commands
```bash
# Run a machine on input
automata run machine.json "input" --trace --max-steps 1000

# Create a new machine interactively
automata create dfa --output my_dfa.json

# Test a machine against test cases
automata test machine.json tests.json

# Visualize a machine
automata visualize machine.json

# Show example machines
automata examples

# Interactive undecidability demo
automata undecidability
```

### Options
- `--trace`: Show step-by-step execution
- `--max-steps N`: Limit execution to N steps
- `--delay N`: Add N second delay between trace steps
- `--loop-detect`: Enable/disable loop detection
- `--output FILE`: Specify output file

## 📖 Theory Connections

### Regular Languages (DFA/NFA)
- Closure properties
- Pumping lemma applications
- Minimization algorithms
- Regular expressions

### Context-Free Languages
- Pushdown automata (coming soon)
- CFG to PDA conversion
- CYK algorithm

### Decidability
- Halting problem
- Rice's theorem
- Recursive vs recursively enumerable
- Reductions

### Complexity
- P vs NP
- NP-completeness
- Space complexity
- Time hierarchies

## 🎓 Educational Usage

### For Instructors
1. **Assignments**: Use provided exercises as homework
2. **Exams**: Generate random test cases
3. **Demonstrations**: Show undecidability live in class
4. **Projects**: Students implement advanced machines

### For Students
1. **Immediate Feedback**: Know instantly if your machine works
2. **Visual Learning**: See exactly how machines process input
3. **Exploration**: Experiment with limits of computation
4. **Understanding**: Make abstract concepts concrete

## 🔗 Integration with Course

### Docker Support
```bash
# Build with course Docker image
docker run -v $(pwd):/app scott-weeden/latex automata test
```

### Continuous Testing
```yaml
# GitHub Actions workflow
name: Test Automata
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pip install -e .
      - run: pytest
```

## 🐛 Debugging Tips

### Common Issues

1. **DFA not complete**: Ensure transitions for all (state, symbol) pairs
2. **NFA epsilon loops**: Check ε-closure computation
3. **TM infinite loops**: Use `--max-steps` and `--trace`
4. **Test failures**: Read hints in error messages

### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Use trace in code
dfa.accepts("input", trace=True)
```

## 📚 Resources

- **Documentation**: https://automata-theory.readthedocs.io
- **Course Materials**: See `/course-content`
- **Video Tutorials**: YouTube playlist (coming soon)
- **Discord**: Join our study group

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- More exercise templates
- Visualization improvements
- Performance optimizations
- Documentation examples
- Test case generators

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by Sipser's "Introduction to the Theory of Computation"
- Built for CS Theory courses worldwide
- Special thanks to students who tested early versions

## 🚧 Roadmap

### Coming Soon
- [ ] Pushdown Automata (PDA)
- [ ] Linear Bounded Automata
- [ ] Quantum Automata
- [ ] Interactive web playground
- [ ] Video trace exports
- [ ] Advanced complexity analysis
- [ ] LLM integration for hints

## 💡 Tips for Success

1. **Start Small**: Build simple machines first
2. **Test Often**: Use pytest after each change
3. **Trace Execution**: Watch your machine run with `--trace`
4. **Explore Limits**: Try inputs that might cause loops
5. **Learn from Failures**: Red tests teach as much as green ones

---

**Remember**: Every loop that spirals endlessly is a reminder that some questions have no algorithmic answer. Welcome to the beautiful world of theoretical computer science! 🎭
