"""Pushdown Automaton implementation"""

class PDA:
    def __init__(self, states, alphabet, stack_alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.stack_alphabet = stack_alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
    
    def accepts(self, string):
        """Check if string is accepted by PDA"""
        def explore(state, input_pos, stack):
            # Base case: end of input
            if input_pos == len(string):
                # Try epsilon transitions at end
                stack_top = stack[-1] if stack else None
                if stack_top and (state, '', stack_top) in self.transitions:
                    for next_state, stack_op in self.transitions[(state, '', stack_top)]:
                        new_stack = apply_stack_op(stack, stack_op)
                        if new_stack is not None and explore(next_state, input_pos, new_stack):
                            return True
                return state in self.accept_states
            
            # Try epsilon transitions first
            stack_top = stack[-1] if stack else None
            if stack_top and (state, '', stack_top) in self.transitions:
                for next_state, stack_op in self.transitions[(state, '', stack_top)]:
                    new_stack = apply_stack_op(stack, stack_op)
                    if new_stack is not None and explore(next_state, input_pos, new_stack):
                        return True
            
            # Try input symbol transitions
            symbol = string[input_pos]
            if stack_top and (state, symbol, stack_top) in self.transitions:
                for next_state, stack_op in self.transitions[(state, symbol, stack_top)]:
                    new_stack = apply_stack_op(stack, stack_op)
                    if new_stack is not None and explore(next_state, input_pos + 1, new_stack):
                        return True
            
            return False
        
        def apply_stack_op(stack, stack_op):
            """Apply stack operation and return new stack"""
            if not stack:
                return None
            
            new_stack = stack[:-1]  # Remove top symbol (last element)
            
            if stack_op:  # Push new symbols (in reverse order to maintain stack semantics)
                new_stack.extend(reversed(list(stack_op)))
            
            return new_stack
        
        # Start with Z as initial stack symbol
        return explore(self.start_state, 0, ['Z'])
