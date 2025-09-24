#!/usr/bin/env python3
"""Command-line interface for automata theory library"""

import argparse
import json
import sys
from automata import DFA, NFA, PDA, TuringMachine, LBA, QFA, create_hadamard_qfa, create_deutsch_qfa

def main():
    parser = argparse.ArgumentParser(description='Automata Theory CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test an automaton')
    test_parser.add_argument('type', choices=['dfa', 'nfa', 'pda', 'tm', 'lba', 'qfa'])
    test_parser.add_argument('config', help='JSON configuration file')
    test_parser.add_argument('input', help='Input string to test')
    test_parser.add_argument('--trace', action='store_true', help='Show execution trace')
    test_parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps for TM/LBA')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run automaton with detailed output')
    run_parser.add_argument('type', choices=['dfa', 'nfa', 'pda', 'tm', 'lba', 'qfa'])
    run_parser.add_argument('config', help='JSON configuration file')
    run_parser.add_argument('input', help='Input string')
    run_parser.add_argument('--trace', action='store_true', help='Show execution trace')
    run_parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps')
    
    # Examples command
    examples_parser = subparsers.add_parser('examples', help='Show example automata')
    examples_parser.add_argument('type', nargs='?', choices=['dfa', 'nfa', 'pda', 'tm', 'lba', 'qfa'])
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start web playground')
    web_parser.add_argument('--port', type=int, default=5000, help='Port number')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        test_automaton(args)
    elif args.command == 'run':
        run_automaton(args)
    elif args.command == 'examples':
        show_examples(args.type)
    elif args.command == 'interactive':
        interactive_mode()
    elif args.command == 'web':
        start_web_playground(args.port)
    else:
        parser.print_help()

def test_automaton(args):
    """Test an automaton with given input"""
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        automaton = create_automaton(args.type, config)
        
        if args.type in ['tm', 'lba']:
            result, steps, trace = automaton.run(args.input, max_steps=args.max_steps, trace=args.trace)
            print(f"Result: {'ACCEPTED' if result else 'REJECTED'}")
            print(f"Steps: {steps}")
            if args.trace and trace:
                print("\nTrace:")
                for step in trace[:10]:  # Show first 10 steps
                    print(f"  Step {step['step']}: {step}")
        elif args.type == 'qfa':
            result = automaton.accepts(args.input)
            prob = automaton.acceptance_probability(args.input)
            print(f"Result: {'ACCEPTED' if result else 'REJECTED'}")
            print(f"Acceptance Probability: {prob:.4f}")
            if args.trace:
                _, trace = automaton.run_with_trace(args.input)
                print("\nQuantum State Trace:")
                for step in trace:
                    print(f"  {step}")
        else:
            result = automaton.accepts(args.input)
            print(f"Result: {'ACCEPTED' if result else 'REJECTED'}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def run_automaton(args):
    """Run automaton with detailed output"""
    test_automaton(args)  # Same as test for now

def create_automaton(automaton_type, config):
    """Create automaton instance from config"""
    if automaton_type == 'dfa':
        return DFA(**config)
    elif automaton_type == 'nfa':
        return NFA(**config)
    elif automaton_type == 'pda':
        return PDA(**config)
    elif automaton_type == 'tm':
        return TuringMachine(**config)
    elif automaton_type == 'lba':
        return LBA(**config)
    elif automaton_type == 'qfa':
        return QFA(**config)
    else:
        raise ValueError(f"Unknown automaton type: {automaton_type}")

def show_examples(automaton_type=None):
    """Show example automata configurations"""
    examples = {
        'dfa': {
            'even_zeros': {
                'states': ['q0', 'q1'],
                'alphabet': ['0', '1'],
                'transitions': {'q0,0': 'q1', 'q0,1': 'q0', 'q1,0': 'q0', 'q1,1': 'q1'},
                'start_state': 'q0',
                'accept_states': ['q0']
            }
        },
        'nfa': {
            'ends_01': {
                'states': ['q0', 'q1', 'q2'],
                'alphabet': ['0', '1'],
                'transitions': {'q0,0': ['q0', 'q1'], 'q0,1': ['q0'], 'q1,1': ['q2']},
                'start_state': 'q0',
                'accept_states': ['q2']
            }
        },
        'pda': {
            'balanced_parens': {
                'states': ['q0', 'q1'],
                'alphabet': ['(', ')'],
                'stack_alphabet': ['Z', 'P'],
                'transitions': {
                    "('q0', '(', 'Z')": [['q0', 'PZ']],
                    "('q0', '(', 'P')": [['q0', 'PP']],
                    "('q0', ')', 'P')": [['q0', '']],
                    "('q0', '', 'Z')": [['q1', 'Z']]
                },
                'start_state': 'q0',
                'accept_states': ['q1']
            }
        },
        'qfa': {
            'hadamard': {
                'description': 'Simple quantum automaton with Hadamard-like transitions',
                'example': 'Use create_hadamard_qfa() function'
            }
        }
    }
    
    if automaton_type:
        if automaton_type in examples:
            print(f"Examples for {automaton_type.upper()}:")
            for name, config in examples[automaton_type].items():
                print(f"\n{name}:")
                print(json.dumps(config, indent=2))
        else:
            print(f"No examples available for {automaton_type}")
    else:
        print("Available example types:")
        for atype in examples.keys():
            print(f"  {atype}")
        print("\nUse 'automata examples <type>' to see specific examples")

def interactive_mode():
    """Start interactive mode"""
    print("🤖 Automata Theory Interactive Mode")
    print("Type 'help' for commands, 'quit' to exit")
    
    while True:
        try:
            command = input("\nautomata> ").strip()
            if command == 'quit':
                break
            elif command == 'help':
                print("Commands:")
                print("  test <type> <config.json> <input>")
                print("  examples [type]")
                print("  quit")
            elif command.startswith('test'):
                parts = command.split()
                if len(parts) >= 4:
                    # Simulate args
                    class Args:
                        def __init__(self):
                            self.type = parts[1]
                            self.config = parts[2]
                            self.input = parts[3]
                            self.trace = False
                            self.max_steps = 1000
                    test_automaton(Args())
                else:
                    print("Usage: test <type> <config.json> <input>")
            elif command.startswith('examples'):
                parts = command.split()
                atype = parts[1] if len(parts) > 1 else None
                show_examples(atype)
            else:
                print("Unknown command. Type 'help' for available commands.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

def start_web_playground(port):
    """Start web playground"""
    try:
        from web_playground import app
        print(f"🌐 Starting web playground on http://localhost:{port}")
        app.run(debug=True, port=port)
    except ImportError:
        print("Web playground requires Flask. Install with: pip install flask")
    except Exception as e:
        print(f"Error starting web playground: {e}")

if __name__ == '__main__':
    main()
