# SA Algorithms - Clean Implementations

This directory contains 4 clean, standalone implementations of the Simulated Annealing algorithm, all with the same interface.

## Files

### 1. `python_serial.py`
Pure Python implementation with serial execution.
- **Use case**: Reference implementation, debugging, understanding the algorithm
- **Performance**: Baseline performance
- **Dependencies**: numpy only

### 2. `python_parallel.py`
Python implementation with multiprocessing parallelization.
- **Use case**: When you have multiple CPU cores and want moderate speedup
- **Performance**: ~2x speedup over serial Python
- **Dependencies**: numpy, multiprocessing

### 3. `rust_serial.py`
Rust implementation via PyO3 with serial execution.
- **Use case**: Maximum single-threaded performance
- **Performance**: ~90x speedup over serial Python
- **Dependencies**: numpy, sa_rust (Rust extension module)

### 4. `rust_parallel.py`
Rust implementation via PyO3 with parallel execution using Rust threads.
- **Use case**: Maximum performance with multiple cores
- **Performance**: ~170x speedup over serial Python
- **Dependencies**: numpy, sa_rust (Rust extension module)

## Interface

All implementations provide the same function signature:

```python
run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10, num_threads=None)
```

### Parameters

- `init_temp` (float): Initial temperature for SA
- `cooling_rate` (float): Temperature decay rate per step (0 < rate < 1)
- `step_size` (float): Standard deviation for random walk
- `num_steps` (int): Number of SA iterations per run
- `bounds` (tuple): (min, max) bounds for search space
- `seed` (int, optional): Random seed for reproducibility
- `num_runs` (int, default=10): Number of independent SA runs
- `num_threads` (int, optional): Number of parallel threads (parallel versions only)

### Returns

Tuple of `(avg_reward, costs, trajectory, median_idx)`:
- `avg_reward`: Average reward across all runs with penalty
- `costs`: List of final costs for each run
- `trajectory`: Trajectory of the median cost run as list of (x, y, cost) tuples
- `median_idx`: Index of the run with median cost

## Usage Examples

### Python Serial
```python
from sa_algorithms import python_serial

avg_reward, costs, trajectory, median_idx = python_serial.run_sa(
    init_temp=10.0,
    cooling_rate=0.95,
    step_size=1.0,
    num_steps=1000,
    bounds=(-5.12, 5.12),
    seed=42,
    num_runs=10
)
```

### Python Parallel
```python
from sa_algorithms import python_parallel

avg_reward, costs, trajectory, median_idx = python_parallel.run_sa(
    init_temp=10.0,
    cooling_rate=0.95,
    step_size=1.0,
    num_steps=1000,
    bounds=(-5.12, 5.12),
    seed=42,
    num_runs=100,
    num_threads=4  # Use 4 CPU cores
)
```

### Rust Serial
```python
from sa_algorithms import rust_serial

avg_reward, costs, trajectory, median_idx = rust_serial.run_sa(
    init_temp=10.0,
    cooling_rate=0.95,
    step_size=1.0,
    num_steps=1000,
    bounds=(-5.12, 5.12),
    seed=42,
    num_runs=10
)
```

### Rust Parallel
```python
from sa_algorithms import rust_parallel

avg_reward, costs, trajectory, median_idx = rust_parallel.run_sa(
    init_temp=10.0,
    cooling_rate=0.95,
    step_size=1.0,
    num_steps=1000,
    bounds=(-5.12, 5.12),
    seed=42,
    num_runs=100,
    num_threads=4  # Use 4 Rust threads
)
```

## Performance Comparison

Based on comprehensive benchmarks with 100 runs per test:

| Steps | Python Serial | Python Parallel | Rust Serial | Rust Parallel |
|-------|--------------|-----------------|-------------|---------------|
| 100   | 0.0999s      | 0.0680s         | 0.0011s     | 0.0006s       |
| 1000  | 0.9784s      | 0.4955s         | 0.0106s     | 0.0046s       |
| 5000  | 4.7767s      | 2.3546s         | 0.0482s     | 0.0217s       |
| 10000 | 9.5679s      | 4.7192s         | 0.0956s     | 0.0436s       |

**Speedup vs Python Serial:**
- Python Parallel: ~1.7x
- Rust Serial: ~91x
- Rust Parallel: ~171x

## Rastrigin Function

All implementations optimize the 2D Rastrigin function:

```
f(x, y) = 20 + (x/1.5)² - 10*cos(2π*x/1.5) + (y/1.5)² - 10*cos(2π*y/1.5)
```

- Global minimum: f(0, 0) = 0
- Search space: [-5.12, 5.12]²
- Many local minima (challenging for optimization)

## Building Rust Extensions

To use Rust implementations, build the extension module:

```bash
# Activate virtual environment
source .venv/bin/activate

# Build Rust extension (release mode for best performance)
maturin develop --release
```

See `RUST_SETUP.md` in the parent directory for detailed setup instructions.

## Running Benchmarks

Compare all implementations:

```bash
source .venv/bin/activate
python comprehensive_benchmark.py
```

This will test all 4 implementations across different step counts and produce detailed performance tables.
