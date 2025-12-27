avg_reward, costs, trajectory, median_idx = python_serial.run_sa(
avg_reward, costs, trajectory, median_idx = python_parallel.run_sa(
avg_reward, costs, trajectory, median_idx = rust_serial.run_sa(
avg_reward, costs, trajectory, median_idx = rust_parallel.run_sa(

# Simulated Annealing Unified Interface

This module provides a single interface to run different Simulated Annealing (SA) algorithm implementations.

## Usage

```python
from sa_algorithms import run_sa

# Choose one of: 'python_serial', 'python_parallel', 'rust_serial', 'rust_parallel'
result = run_sa(
    algorithm='python_serial',
    init_temp=10.0,
    cooling_rate=0.95,
    step_size=1.0,
    num_steps=1000,
    bounds=(-5.12, 5.12),
    seed=42,
    num_runs=10,
    num_threads=None  # Only for parallel versions
)
```

See the docstring in `__init__.py` for details on arguments and return values.
