"""Linear Bounded Automaton implementation"""

class LBA:
    def __init__(self, states, alphabet, tape_alphabet, transitions, start_state, accept_states, reject_states=None):
        self.states = states
        self.alphabet = alphabet
        self.tape_alphabet = tape_alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
        self.reject_states = reject_states or set()
        self.left_marker = '⊢'
        self.right_marker = '⊣'
    
    def accepts(self, string, max_steps=10000):
        """Check if string is accepted by LBA"""
        if not string:
            return self.start_state in self.accept_states
        
        # Initialize tape with markers
        tape = [self.left_marker] + list(string) + [self.right_marker]
        head = 1  # Start at first input symbol
        state = self.start_state
        steps = 0
        
        visited = set()
        
        while steps < max_steps:
            # Check for acceptance/rejection
            if state in self.accept_states:
                return True
            if state in self.reject_states:
                return False
            
            # Check for infinite loop
            config = (state, head, tuple(tape))
            if config in visited:
                return False  # Infinite loop detected
            visited.add(config)
            
            # Get current symbol
            current_symbol = tape[head]
            
            # Look for transition
            if (state, current_symbol) in self.transitions:
                next_state, write_symbol, direction = self.transitions[(state, current_symbol)]
                
                # Write symbol
                tape[head] = write_symbol
                
                # Move head
                if direction == 'L':
                    if head > 1:  # Can't move past left marker
                        head -= 1
                elif direction == 'R':
                    if head < len(tape) - 1:  # Can't move past right marker
                        head += 1
                
                state = next_state
                steps += 1
            else:
                # No transition available
                return False
        
        # Max steps exceeded
        return False
    
    def run(self, string, max_steps=10000, trace=False):
        """Run LBA on string with optional trace"""
        if not string:
            return self.start_state in self.accept_states, 0, []
        
        # Initialize tape with markers
        tape = [self.left_marker] + list(string) + [self.right_marker]
        head = 1  # Start at first input symbol
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
                old_head = head
                if direction == 'L':
                    if head > 1:  # Can't move past left marker
                        head -= 1
                elif direction == 'R':
                    if head < len(tape) - 1:  # Can't move past right marker
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
                    'old_head': old_head,
                    'new_head': head,
                    'tape': tape[:]
                })
                
                if trace:
                    tape_display = ''.join(tape[:head]) + f'[{tape[head]}]' + ''.join(tape[head+1:])
                    print(f"  After: {tape_display}")
            else:
                # No transition available
                if trace:
                    print(f"No transition from state {state} on symbol {current_symbol}")
                return False, steps, trace_steps
        
        # Max steps exceeded
        if trace:
            print(f"MAX STEPS ({max_steps}) exceeded")
        return False, steps, trace_steps
