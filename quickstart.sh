#!/bin/bash
# Automata Theory Library - Quick Start Script
# This script sets up everything needed to start learning automata theory

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ASCII Art Banner
echo -e "${BLUE}"
cat << "EOF"
     _         _                        _        
    / \  _   _| |_ ___  _ __ ___   __ _| |_ __ _ 
   / _ \| | | | __/ _ \| '_ ` _ \ / _` | __/ _` |
  / ___ \ |_| | || (_) | | | | | | (_| | || (_| |
 /_/   \_\__,_|\__\___/|_| |_| |_|\__,_|\__\__,_|
                                                  
       _____ _                          
      |_   _| |__   ___  ___  _ __ _   _ 
        | | | '_ \ / _ \/ _ \| '__| | | |
        | | | | | |  __/ (_) | |  | |_| |
        |_| |_| |_|\___|\___/|_|   \__, |
                                    |___/ 
EOF
echo -e "${NC}"
echo -e "${GREEN}Welcome to Automata Theory Library!${NC}"
echo -e "${YELLOW}Making Undecidability Visceral${NC}\n"

# Function to print step
step() {
    echo -e "\n${BLUE}► $1${NC}"
}

# Function to print success
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Check Python version
step "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [ "$(echo "$PYTHON_VERSION >= 3.8" | bc)" -eq 1 ]; then
        success "Python $PYTHON_VERSION found"
    else
        error "Python 3.8+ required, found $PYTHON_VERSION"
    fi
else
    error "Python 3 not found. Please install Python 3.8+"
fi

# Create virtual environment
step "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    success "Virtual environment created"
else
    success "Virtual environment already exists"
fi

# Activate virtual environment
step "Activating virtual environment..."
source venv/bin/activate
success "Virtual environment activated"

# Upgrade pip
step "Upgrading pip..."
pip install --upgrade pip -q
success "pip upgraded"

# Install the library
step "Installing automata-theory library..."
pip install -e . -q
success "Library installed"

# Install development dependencies
step "Installing development dependencies..."
pip install pytest pytest-cov pytest-xdist rich click graphviz -q
success "Dependencies installed"

# Create directory structure
step "Setting up directory structure..."
mkdir -p student_solutions
mkdir -p examples
mkdir -p tests
mkdir -p visualizations
success "Directories created"

# Create example machines
step "Creating example machines..."

# Example 1: Binary Palindrome DFA
cat > examples/palindrome.dfa.json << 'EOF'
{
  "type": "DFA",
  "states": ["q0", "q1", "q2", "q3", "q4"],
  "alphabet": ["0", "1"],
  "transitions": {
    "q0,0": "q1", "q0,1": "q2",
    "q1,0": "q3", "q1,1": "q4",
    "q2,0": "q4", "q2,1": "q3",
    "q3,0": "q4", "q3,1": "q4",
    "q4,0": "q4", "q4,1": "q4"
  },
  "start_state": "q0",
  "accept_states": ["q3"]
}
EOF

# Example 2: Binary Increment TM
cat > examples/increment.tm.json << 'EOF'
{
  "type": "TuringMachine",
  "states": ["start", "scan_right", "add_one", "carry", "done", "accept", "reject"],
  "input_alphabet": ["0", "1"],
  "tape_alphabet": ["0", "1", "□"],
  "transitions": {
    "start,0": {"state": "scan_right", "write": "0", "move": "R"},
    "start,1": {"state": "scan_right", "write": "1", "move": "R"},
    "start,□": {"state": "add_one", "write": "1", "move": "S"},
    "scan_right,0": {"state": "scan_right", "write": "0", "move": "R"},
    "scan_right,1": {"state": "scan_right", "write": "1", "move": "R"},
    "scan_right,□": {"state": "add_one", "write": "□", "move": "L"},
    "add_one,0": {"state": "done", "write": "1", "move": "L"},
    "add_one,1": {"state": "carry", "write": "0", "move": "L"},
    "add_one,□": {"state": "done", "write": "1", "move": "R"},
    "carry,0": {"state": "done", "write": "1", "move": "L"},
    "carry,1": {"state": "carry", "write": "0", "move": "L"},
    "carry,□": {"state": "done", "write": "1", "move": "R"},
    "done,0": {"state": "done", "write": "0", "move": "L"},
    "done,1": {"state": "done", "write": "1", "move": "L"},
    "done,□": {"state": "accept", "write": "□", "move": "R"}
  },
  "start_state": "start",
  "accept_state": "accept",
  "reject_state": "reject",
  "blank_symbol": "□"
}
EOF

# Example 3: Infinite Loop TM
cat > examples/loop.tm.json << 'EOF'
{
  "type": "TuringMachine",
  "states": ["q0", "q1", "q2", "accept", "reject"],
  "input_alphabet": ["0", "1"],
  "tape_alphabet": ["0", "1", "X", "□"],
  "transitions": {
    "q0,1": {"state": "q1", "write": "X", "move": "R"},
    "q1,1": {"state": "q2", "write": "X", "move": "R"},
    "q2,1": {"state": "q0", "write": "X", "move": "L"},
    "q0,X": {"state": "q1", "write": "X", "move": "R"},
    "q1,X": {"state": "q2", "write": "X", "move": "R"},
    "q2,X": {"state": "q0", "write": "X", "move": "L"},
    "q0,□": {"state": "accept", "write": "□", "move": "S"},
    "q0,0": {"state": "accept", "write": "0", "move": "S"}
  },
  "start_state": "q0",
  "accept_state": "accept",
  "reject_state": "reject",
  "blank_symbol": "□"
}
EOF

success "Example machines created"

# Create starter template for students
step "Creating student exercise templates..."

cat > student_solutions/template.dfa.json << 'EOF'
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
EOF

success "Templates created"

# Create a simple test file
step "Creating test file..."

cat > tests/test_examples.py << 'EOF'
import pytest
import json
from pathlib import Path
import sys
sys.path.append('.')

from automata_lib import DFA, NFA, TuringMachine

def test_example_machines_exist():
    """Test that example machines are created"""
    assert Path('examples/palindrome.dfa.json').exists()
    assert Path('examples/increment.tm.json').exists()
    assert Path('examples/loop.tm.json').exists()

def test_dfa_loading():
    """Test that DFA can be loaded"""
    with open('examples/palindrome.dfa.json', 'r') as f:
        data = json.load(f)
    dfa = DFA.from_dict(data)
    assert dfa is not None

def test_tm_loading():
    """Test that TM can be loaded"""
    with open('examples/increment.tm.json', 'r') as f:
        data = json.load(f)
    tm = TuringMachine.from_dict(data)
    assert tm is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
EOF

success "Test file created"

# Run tests to verify installation
step "Running verification tests..."
python -m pytest tests/test_examples.py -q --tb=no
if [ $? -eq 0 ]; then
    success "All tests passed!"
else
    error "Tests failed. Please check installation."
fi

# Print instructions
echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Installation Complete! 🎉${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Quick Start Commands:${NC}"
echo -e "  ${BLUE}1. Test the DFA example:${NC}"
echo -e "     automata run examples/palindrome.dfa.json '00' --trace"
echo -e ""
echo -e "  ${BLUE}2. Watch a TM increment a binary number:${NC}"
echo -e "     automata run examples/increment.tm.json '111' --trace"
echo -e ""
echo -e "  ${BLUE}3. See an infinite loop (undecidability):${NC}"
echo -e "     automata run examples/loop.tm.json '111' --trace --max-steps 30"
echo -e ""
echo -e "  ${BLUE}4. Create your own DFA interactively:${NC}"
echo -e "     automata create dfa --output my_first_dfa.json"
echo -e ""
echo -e "  ${BLUE}5. Run the undecidability demonstration:${NC}"
echo -e "     automata undecidability"
echo -e ""
echo -e "  ${BLUE}6. Run student exercise tests:${NC}"
echo -e "     pytest test_automata.py -v"

echo -e "\n${YELLOW}Project Structure:${NC}"
echo -e "  📁 student_solutions/  - Your exercise solutions"
echo -e "  📁 examples/          - Example machines"
echo -e "  📁 tests/             - Test files"
echo -e "  📁 visualizations/    - Generated diagrams"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Copy template.dfa.json to create your first exercise"
echo -e "  2. Modify the transitions to solve the problem"
echo -e "  3. Test with: automata test <your_machine> <test_file>"
echo -e "  4. Get immediate feedback with pytest"

echo -e "\n${GREEN}Happy Learning! Remember: Every infinite loop is a lesson in computability! 🔄${NC}\n"

# Optionally run an interactive demo
read -p "Would you like to see the undecidability demo now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Running undecidability demonstration...${NC}\n"
    python -c "
from automata_lib import create_infinite_loop_tm
import time

tm = create_infinite_loop_tm()
print('🔄 Watch this Turing Machine spiral into an infinite loop!')
print('Input: 111')
print('Max steps: 30')
print('')
time.sleep(2)
tm.run('111', max_steps=30, trace=True, delay=0.1)
"
fi

echo -e "\n${BLUE}To activate the virtual environment in future sessions:${NC}"
echo -e "  source venv/bin/activate"
echo -e "\n${GREEN}Enjoy exploring the limits of computation! 🚀${NC}\n"