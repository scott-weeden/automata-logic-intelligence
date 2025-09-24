"""Interactive web playground for automata theory"""

from flask import Flask, render_template, request, jsonify
import json
import sys
import os

# Add automata module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'automata'))

from automata.dfa import DFA
from automata.nfa import NFA
from automata.pda import PDA
from automata.tm import TuringMachine
from automata.lba import LBA
from automata.qfa import QFA, create_hadamard_qfa, create_deutsch_qfa

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/test_automaton', methods=['POST'])
def test_automaton():
    try:
        data = request.json
        automaton_type = data['type']
        config = data['config']
        test_string = data['test_string']
        
        result = None
        trace = None
        
        if automaton_type == 'DFA':
            dfa = DFA(**config)
            result = dfa.accepts(test_string)
            
        elif automaton_type == 'NFA':
            nfa = NFA(**config)
            result = nfa.accepts(test_string)
            
        elif automaton_type == 'PDA':
            pda = PDA(**config)
            result = pda.accepts(test_string)
            
        elif automaton_type == 'TM':
            tm = TuringMachine(**config)
            result, steps, trace_data = tm.run(test_string, max_steps=1000, trace=True)
            trace = trace_data
            
        elif automaton_type == 'LBA':
            lba = LBA(**config)
            result, steps, trace_data = lba.run(test_string, max_steps=1000, trace=True)
            trace = trace_data
            
        elif automaton_type == 'QFA':
            qfa = QFA(**config)
            result = qfa.accepts(test_string)
            prob, trace_data = qfa.run_with_trace(test_string)
            trace = trace_data
            
        return jsonify({
            'success': True,
            'result': result,
            'trace': trace
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/examples')
def get_examples():
    examples = {
        'DFA': {
            'even_zeros': {
                'states': ['q0', 'q1'],
                'alphabet': ['0', '1'],
                'transitions': {
                    'q0,0': 'q1',
                    'q0,1': 'q0',
                    'q1,0': 'q0',
                    'q1,1': 'q1'
                },
                'start_state': 'q0',
                'accept_states': ['q0']
            }
        },
        'NFA': {
            'ends_01': {
                'states': ['q0', 'q1', 'q2'],
                'alphabet': ['0', '1'],
                'transitions': {
                    'q0,0': ['q0', 'q1'],
                    'q0,1': ['q0'],
                    'q1,1': ['q2']
                },
                'start_state': 'q0',
                'accept_states': ['q2']
            }
        },
        'PDA': {
            'balanced_parens': {
                'states': ['q0', 'q1'],
                'alphabet': ['(', ')'],
                'stack_alphabet': ['Z', 'P'],
                'transitions': {
                    ('q0', '(', 'Z'): [('q0', 'PZ')],
                    ('q0', '(', 'P'): [('q0', 'PP')],
                    ('q0', ')', 'P'): [('q0', '')],
                    ('q0', '', 'Z'): [('q1', 'Z')]
                },
                'start_state': 'q0',
                'accept_states': ['q1']
            }
        },
        'QFA': {
            'hadamard': create_hadamard_qfa().__dict__
        }
    }
    
    return jsonify(examples)

@app.route('/api/visualize', methods=['POST'])
def visualize_automaton():
    try:
        data = request.json
        automaton_type = data['type']
        config = data['config']
        
        # Generate DOT notation for GraphViz
        dot = generate_dot(automaton_type, config)
        
        return jsonify({
            'success': True,
            'dot': dot
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_dot(automaton_type, config):
    """Generate GraphViz DOT notation for automaton visualization"""
    dot = "digraph automaton {\n"
    dot += "  rankdir=LR;\n"
    dot += "  node [shape=circle];\n"
    
    if automaton_type in ['DFA', 'NFA']:
        # Add states
        for state in config['accept_states']:
            dot += f'  {state} [shape=doublecircle];\n'
        
        # Add start state arrow
        dot += f'  start [shape=point];\n'
        dot += f'  start -> {config["start_state"]};\n'
        
        # Add transitions
        transitions = config['transitions']
        if automaton_type == 'DFA':
            for key, target in transitions.items():
                state, symbol = key.split(',')
                dot += f'  {state} -> {target} [label="{symbol}"];\n'
        else:  # NFA
            for key, targets in transitions.items():
                state, symbol = key.split(',')
                for target in targets:
                    dot += f'  {state} -> {target} [label="{symbol}"];\n'
    
    elif automaton_type == 'PDA':
        # Add states
        for state in config['accept_states']:
            dot += f'  {state} [shape=doublecircle];\n'
        
        # Add start state arrow
        dot += f'  start [shape=point];\n'
        dot += f'  start -> {config["start_state"]};\n'
        
        # Add transitions (simplified)
        for (state, symbol, stack_top), transitions in config['transitions'].items():
            for next_state, stack_op in transitions:
                label = f"{symbol},{stack_top}/{stack_op}"
                dot += f'  {state} -> {next_state} [label="{label}"];\n'
    
    dot += "}\n"
    return dot

if __name__ == '__main__':
    app.run(debug=True, port=5000)
