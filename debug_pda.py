"""Debug version of PDA to understand the issue"""

class DebugPDA:
    def __init__(self, states, alphabet, stack_alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.stack_alphabet = stack_alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
    
    def accepts(self, string, debug=False):
        """Check if string is accepted by PDA"""
        def explore(state, input_pos, stack, depth=0):
            indent = "  " * depth
            if debug:
                print(f"{indent}explore(state={state}, pos={input_pos}, stack={stack})")
            
            # Base case: end of input
            if input_pos == len(string):
                if debug:
                    print(f"{indent}End of input. State {state} in accept_states? {state in self.accept_states}")
                
                # Try epsilon transitions at end
                stack_top = stack[-1] if stack else None
                if stack_top and (state, '', stack_top) in self.transitions:
                    if debug:
                        print(f"{indent}Trying epsilon transition: ({state}, '', {stack_top})")
                    for next_state, stack_op in self.transitions[(state, '', stack_top)]:
                        new_stack = apply_stack_op(stack, stack_op)
                        if debug:
                            print(f"{indent}  -> ({next_state}, {stack_op}) new_stack={new_stack}")
                        if new_stack is not None and explore(next_state, input_pos, new_stack, depth+1):
                            return True
                return state in self.accept_states
            
            # Try epsilon transitions first
            stack_top = stack[-1] if stack else None
            if stack_top and (state, '', stack_top) in self.transitions:
                if debug:
                    print(f"{indent}Trying epsilon transition: ({state}, '', {stack_top})")
                for next_state, stack_op in self.transitions[(state, '', stack_top)]:
                    new_stack = apply_stack_op(stack, stack_op)
                    if debug:
                        print(f"{indent}  -> ({next_state}, {stack_op}) new_stack={new_stack}")
                    if new_stack is not None and explore(next_state, input_pos, new_stack, depth+1):
                        return True
            
            # Try input symbol transitions
            symbol = string[input_pos]
            if debug:
                print(f"{indent}Trying symbol '{symbol}' with stack_top={stack_top}")
            if stack_top and (state, symbol, stack_top) in self.transitions:
                if debug:
                    print(f"{indent}Found transition: ({state}, {symbol}, {stack_top})")
                for next_state, stack_op in self.transitions[(state, symbol, stack_top)]:
                    new_stack = apply_stack_op(stack, stack_op)
                    if debug:
                        print(f"{indent}  -> ({next_state}, {stack_op}) new_stack={new_stack}")
                    if new_stack is not None and explore(next_state, input_pos + 1, new_stack, depth+1):
                        return True
            
            if debug:
                print(f"{indent}No valid transitions, returning False")
            return False
        
        def apply_stack_op(stack, stack_op):
            """Apply stack operation and return new stack"""
            if not stack:
                return None
            
            new_stack = stack[:]
            new_stack.pop()  # Remove top symbol
            
            if stack_op:  # Push new symbols
                new_stack.extend(list(stack_op))
            
            return new_stack
        
        # Start with Z as initial stack symbol
        return explore(self.start_state, 0, ['Z'])

# Test
pda = DebugPDA(
    states={'q0', 'q1'},
    alphabet={'(', ')'},
    stack_alphabet={'Z', 'P'},
    transitions={
        ('q0', '(', 'Z'): {('q0', 'PZ')},
        ('q0', ')', 'P'): {('q0', '')},
        ('q0', '', 'Z'): {('q1', 'Z')}
    },
    start_state='q0',
    accept_states={'q1'}
)

print("Testing '()' with debug:")
result = pda.accepts('()', debug=True)
print(f"Final result: {result}")
