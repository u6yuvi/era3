# Value Iteration in GridWorld

This implementation demonstrates the Value Iteration algorithm in a simple GridWorld environment. The agent learns to find optimal values for each state in a grid-based world where the goal is to reach the terminal state (bottom-right corner) with minimum steps.

## Overview

The implementation includes:
- A `GridWorld` class that creates an n×n grid environment
- Value Iteration algorithm to compute optimal state values
- Configurable debug output for learning visualization

## Features

- Customizable grid size
- Configurable verbosity for debugging and visualization
- Deterministic transitions
- Discount factor (γ) support
- Convergence threshold (θ) for iteration stopping
- Equal probability action selection

## Usage

```python
from value_iteration import GridWorld

# Create a 4x4 GridWorld with verbose output
env = GridWorld(size=4, verbose=True)

# Run value iteration
iterations = env.value_iteration()

# The optimal values will be stored in env.V
```

### Parameters

- `size`: Size of the grid (default: 4)
- `verbose`: Enable/disable detailed output during iteration (default: False)

### Environment Details

- **States**: Each cell in the grid represents a state
- **Actions**: Up, Down, Left, Right
- **Rewards**: -1 for each step, 0 at terminal state
- **Terminal State**: Bottom-right corner of the grid
- **Transition Model**: Deterministic (actions always succeed)

### Output

When `verbose=True`, the algorithm will print:
- State transitions and their values
- Action contributions to value updates
- Value function grid after each iteration
- Convergence information

## Requirements

- Python 3.x
- NumPy

## Example Output

```
State 0 (row=0, col=0):
  Action up: next_state=0, reward=-1, contribution=-0.2500
  Action down: next_state=4, reward=-1, contribution=-0.2500
  ...

Final Value Function:
[[-59.42553561 -57.4257263  -54.28309529 -51.71183027]
 [-57.4257263  -54.56881007 -49.71196896 -45.14079147]
 [-54.28309529 -49.71196896 -40.85530555 -29.99868479]
 [-51.71183027 -45.14079147 -29.99868479   0.        ]]
...
``` 