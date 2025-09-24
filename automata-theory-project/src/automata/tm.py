"""Turing Machine implementation"""

class TuringMachine:
    def __init__(self, states, alphabet, tape_alphabet, transitions, start_state, accept_states, reject_states=None):
        self.states = states
        self.alphabet = alphabet
        self.tape_alphabet = tape_alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
        self.reject_states = reject_states or set()
        self.blank = '_'
    
    def accepts(self, input_string, max_steps=1000):
        """Check if string is accepted"""
        result, _, _ = self.run(input_string, max_steps)
        return result
    
    def run(self, input_string, max_steps=1000, trace=False):
        """Run Turing Machine on input string"""
        # Initialize tape
        tape = list(input_string) + [self.blank]
        head = 0
        state = self.start_state
        steps = 0
        trace_steps = []
        visited = set()
        
        if trace:
            print(f"Initial: state={state}, head={head}, tape={''.join(tape)}")
        
        while steps < max_steps:
            # Check for acceptance/rejection
            if state in self.accept_states:
                if trace:
                    print(f"ACCEPTED in state {state}")
                return True, steps, trace_steps
            if state in self.reject_states:
                if trace:
                    print(f"REJECTED in state {state}")
                return False, steps, trace_steps
            
            # Check for infinite loop
            config = (state, head, tuple(tape))
            if config in visited:
                if trace:
                    print("INFINITE LOOP detected")
                return False, steps, trace_steps
            visited.add(config)
            
            # Extend tape if needed
            while head >= len(tape):
                tape.append(self.blank)
            while head < 0:
                tape.insert(0, self.blank)
                head = 0
            
            # Get current symbol
            current_symbol = tape[head]
            
            # Look for transition
            if (state, current_symbol) in self.transitions:
                next_state, write_symbol, direction = self.transitions[(state, current_symbol)]
                
                if trace:
                    tape_display = ''.join(tape[:head]) + f'[{tape[head]}]' + ''.join(tape[head+1:])
                    print(f"Step {steps}: {state} reading {current_symbol} -> {next_state}, write {write_symbol}, move {direction}")
                    print(f"  Tape: {tape_display}")
                
                # Write symbol
                tape[head] = write_symbol
                
                # Move head
                if direction == 'L':
                    head -= 1
                elif direction == 'R':
                    head += 1
                
                state = next_state
                steps += 1
                
                trace_steps.append({
                    'step': steps - 1,
                    'old_state': state,
                    'symbol': current_symbol,
                    'new_state': next_state,
                    'write': write_symbol,
                    'direction': direction,
                    'head': head,
                    'tape': tape[:]
                })
                
                if trace:
                    tape_display = ''.join(tape[:head]) + f'[{tape[head]}]' + ''.join(tape[head+1:])
                    print(f"  After: {tape_display}")
            else:
                # No transition available - implicit rejection
                if trace:
                    print(f"No transition from state {state} on symbol {current_symbol}")
                return False, steps, trace_steps
        
        # Max steps exceeded
        if trace:
            print(f"MAX STEPS ({max_steps}) exceeded")
        return False, steps, trace_steps
