# Assignment 0 Autograder Documentation

## Overview
The autograder for CS5368 Intelligent Systems is designed to automatically evaluate student submissions for programming assignments. This document provides comprehensive information about the autograder's functionality, usage, and implementation details.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Assignment Structure](#assignment-structure)
5. [Grading Criteria](#grading-criteria)
6. [Test Cases](#test-cases)
7. [Common Issues](#common-issues)
8. [API Reference](#api-reference)

## System Requirements

- Python 3.7 or higher
- Required packages:
  - `numpy >= 1.19.0`
  - `matplotlib >= 3.3.0`
  - `pytest >= 6.0.0`
  - `timeout-decorator >= 0.5.0`

## Installation

```bash
# Clone the repository
git clone https://github.com/course/intelligent-systems.git

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py install
```

## Usage

### Running the Autograder

```bash
# Basic usage
python autograder.py -q <question> -s <student_code>

# Run all questions
python autograder.py --all

# Run with timeout
python autograder.py -q q1 --timeout 30

# Generate grade report
python autograder.py --student <student_id> --report
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-q, --question` | Question number to grade | None |
| `-s, --student` | Path to student code | `./` |
| `--all` | Run all questions | False |
| `--timeout` | Timeout in seconds | 60 |
| `--verbose` | Verbose output | False |
| `--report` | Generate HTML report | False |
| `--no-graphics` | Disable graphics | False |

## Assignment Structure

### Directory Layout
```
assignment0/
├── autograder.py          # Main autograder script
├── testCases/            # Test cases for each question
│   ├── q1/
│   │   ├── test_1.txt
│   │   └── test_2.txt
│   └── q2/
├── solutions/            # Reference solutions
├── grading.py           # Grading logic
└── util.py             # Utility functions
```

### Student Code Structure
Students must implement specific functions in designated files:

```python
# searchAgents.py
class SearchAgent:
    def __init__(self, fn='depthFirstSearch'):
        self.searchFunction = fn
    
    def getAction(self, state):
        # Student implementation
        pass

# search.py
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Returns a list of actions that reaches the goal.
    """
    # Student implementation
    pass
```

## Grading Criteria

### Point Distribution
- **Correctness (70%)**: Algorithm produces correct output
- **Efficiency (20%)**: Time and space complexity
- **Code Quality (10%)**: Style, documentation, clarity

### Rubric Details

#### Question 1: DFS Implementation (6 points)
- Correct path found: 3 points
- Optimal path length: 2 points
- Proper use of data structures: 1 point

#### Question 2: BFS Implementation (6 points)
- Shortest path found: 3 points
- Correct implementation: 2 points
- Efficiency: 1 point

#### Question 3: UCS Implementation (8 points)
- Optimal cost path: 4 points
- Correct priority queue usage: 2 points
- Handles different costs: 2 points

## Test Cases

### Test Case Format
Each test case is defined in a `.test` file:

```yaml
# test_case.test
class: SearchTest
algorithm: depthFirstSearch
timeout: 30

# Layout definition
layout: """
%%%%%%%
%     %
% %%% %
% %   %
% % % %
%   % %
%%%%%%%
"""

# Expected solution
solution: ['South', 'South', 'West', 'West']
```

### Test Categories

1. **Basic Tests**: Simple mazes with single solutions
2. **Edge Cases**: Empty paths, no solutions, single-state problems
3. **Performance Tests**: Large mazes testing efficiency
4. **Correctness Tests**: Multiple paths, optimal path verification

### Example Test Execution

```python
class SearchTest:
    def __init__(self, question, test_dict):
        self.question = question
        self.test_dict = test_dict
    
    def execute(self, student_module):
        """Execute test against student code"""
        try:
            # Create problem from test layout
            problem = self.create_problem()
            
            # Run student algorithm
            algorithm = getattr(student_module, self.test_dict['algorithm'])
            solution = algorithm(problem)
            
            # Verify solution
            return self.check_solution(solution)
        except Exception as e:
            return TestResult(False, str(e))
```

## Common Issues

### Issue 1: Import Errors
**Problem**: Student code cannot import required modules
**Solution**: Ensure all files are in correct directories and PYTHONPATH is set

### Issue 2: Infinite Loops
**Problem**: Student algorithm doesn't terminate
**Solution**: Autograder implements timeout mechanism (default 60s)

### Issue 3: Incorrect Output Format
**Problem**: Student returns wrong data type
**Solution**: Check that actions are returned as list of strings

### Issue 4: Memory Errors
**Problem**: Algorithm uses too much memory
**Solution**: Implement proper graph search with closed set

## API Reference

### Core Classes

#### `Autograder`
Main autograder class that orchestrates testing.

```python
class Autograder:
    def __init__(self, config_file='config.json'):
        """Initialize autograder with configuration"""
        
    def grade_question(self, question, student_code):
        """Grade a single question"""
        
    def grade_all(self, student_code):
        """Grade all questions"""
        
    def generate_report(self, results):
        """Generate grading report"""
```

#### `TestCase`
Represents a single test case.

```python
class TestCase:
    def __init__(self, test_file):
        """Load test from file"""
        
    def run(self, student_function, timeout=60):
        """Run test with timeout"""
        
    def validate_output(self, output):
        """Validate student output"""
```

#### `GradingResult`
Stores grading results.

```python
@dataclass
class GradingResult:
    question: str
    points_earned: float
    points_possible: float
    feedback: str
    test_results: List[TestResult]
```

### Utility Functions

#### `load_test_case(filename)`
Loads a test case from file.

#### `create_problem(layout)`
Creates a search problem from text layout.

#### `check_solution(path, problem)`
Verifies that a path solves the problem.

#### `measure_performance(function, *args)`
Measures time and memory usage of function execution.

## Advanced Features

### Custom Test Cases
Instructors can add custom test cases:

```python
# custom_test.py
class CustomSearchTest(TestCase):
    def __init__(self):
        super().__init__()
        self.points = 10
        
    def run_test(self, student_module):
        # Custom test logic
        problem = self.create_custom_problem()
        solution = student_module.search(problem)
        return self.evaluate_custom_criteria(solution)
```

### Performance Profiling
The autograder can profile student code:

```python
from autograder import profile_function

@profile_function
def test_performance(student_function):
    # Returns time, memory, and call statistics
    stats = run_with_profiling(student_function, test_input)
    return stats
```

### Partial Credit
Autograder supports partial credit for partially correct solutions:

```python
def calculate_partial_credit(solution, expected):
    """Calculate partial credit based on solution quality"""
    if solution == expected:
        return 1.0
    elif is_valid_path(solution):
        return 0.5
    else:
        return 0.0
```

## Security Considerations

1. **Sandboxing**: Student code runs in restricted environment
2. **Resource Limits**: CPU, memory, and disk usage are limited
3. **Import Restrictions**: Only allowed modules can be imported
4. **File Access**: Students cannot access files outside project

## Troubleshooting

### Debug Mode
Run autograder in debug mode for detailed output:
```bash
python autograder.py -q q1 --debug --verbose
```

### Log Files
Check logs for detailed error information:
- `autograder.log`: Main execution log
- `student_errors.log`: Student code errors
- `test_results.json`: Detailed test results

## Contact and Support

For issues or questions regarding the autograder:
- Course Forum: Piazza
- Email: instructor@university.edu
- Office Hours: See course website

## Appendix

### Sample Grading Report
```
==========================================
CS5368 Intelligent Systems - Assignment 0
Student: student_id
Date: 2025-10-01
==========================================

Question 1: DFS Implementation
Points: 5/6
- Test 1: PASSED (1/1)
- Test 2: PASSED (1/1)
- Test 3: FAILED (0/1)
  Error: Path not optimal
- Test 4: PASSED (1/1)
- Test 5: PASSED (1/1)
- Test 6: PASSED (1/1)

Question 2: BFS Implementation  
Points: 6/6
- All tests passed

Total Score: 11/12 (91.67%)
==========================================
```