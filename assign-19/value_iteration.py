import numpy as np
np.set_printoptions(precision=8, suppress=True)  # For cleaner debug output

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = 1.0  # discount factor
        self.theta = 1e-4  # convergence threshold
        self.action_prob = 0.25  # equal probability for each action
        
        # Initialize value function
        self.V = np.zeros(self.n_states)
        
        # Terminal state (bottom-right corner)
        self.terminal_state = self.n_states - 1
        
    def get_next_state(self, state, action):
        """Get next state given current state and action"""
        row = state // self.size
        col = state % self.size
        
        if action == 'up':
            next_row = max(0, row - 1)
            next_col = col
        elif action == 'down':
            next_row = min(self.size - 1, row + 1)
            next_col = col
        elif action == 'left':
            next_row = row
            next_col = max(0, col - 1)
        else:  # right
            next_row = row
            next_col = min(self.size - 1, col + 1)
            
        return next_row * self.size + next_col
    
    def get_reward(self, state):
        """Get reward for being in state"""
        if state == self.terminal_state:
            return 0
        return -1
    
    def value_iteration(self):
        """Perform value iteration until convergence"""
        iteration = 0
        while True:
            delta = 0
            V_new = self.V.copy()
            
            # Update each state
            for state in range(self.n_states):
                if state == self.terminal_state:
                    continue
                
                # Calculate expected value over all actions (equal probability)
                expected_value = 0
                print(f"\nState {state} (row={state//self.size}, col={state%self.size}):")
                
                for action in self.actions:
                    next_state = self.get_next_state(state, action)
                    value_contribution = self.action_prob * (self.get_reward(state) + self.gamma * V_new[next_state])
                    expected_value += value_contribution
                    print(f"  Action {action}: next_state={next_state}, reward={self.get_reward(state)}, "
                          f"contribution={value_contribution:.4f}")
                
                V_new[state] = expected_value
                print(f"  Final value for state {state}: {V_new[state]:.4f}")
                
                # Track maximum change
                delta = max(delta, abs(V_new[state] - self.V[state]))
            
            # Update value function
            self.V = V_new
            iteration += 1
            
            print(f"\nIteration {iteration}:")
            print("Current Value Function:")
            print_value_grid(self.V, self.size)
            print(f"Max change (delta): {delta:.6f}")
            
            # Check for convergence
            if delta < self.theta:
                break
                
        return iteration

def print_value_grid(V, size):
    """Print value function as a grid"""
    V_grid = V.reshape((size, size))
    print(V_grid)

if __name__ == "__main__":
    # Create and solve GridWorld
    env = GridWorld(size=4)
    iterations = env.value_iteration()
    
    print(f"\nConverged after {iterations} iterations\n")
    print("Final Value Function:")
    print_value_grid(env.V, env.size) 