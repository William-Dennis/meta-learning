# Switching SA Algorithm Implementations

This guide shows how to easily switch between the 4 SA algorithm implementations in this repository.

## Quick Switch

**All you need to do is edit one line in `sa_config.py`:**

```python
# In sa_config.py, line 15:
ALGORITHM = 'rust_parallel'  # Change this line!
```

Available options:
- `'python_serial'` - Pure Python baseline (slowest but no dependencies)
- `'python_parallel'` - Python with multiprocessing (~1.7x faster)
- `'rust_serial'` - Rust serial (~91x faster, requires Rust build)
- `'rust_parallel'` - Rust parallel (~171x faster, requires Rust build) - **RECOMMENDED**

## Examples

### Use Pure Python (No Rust Required)

Edit `sa_config.py`:
```python
ALGORITHM = 'python_serial'
```

Then run any script:
```bash
python run_experiment.py
python run_grid_search.py
```

### Use Python with Parallelization

Edit `sa_config.py`:
```python
ALGORITHM = 'python_parallel'
```

Then run:
```bash
python run_experiment.py
```

### Use Rust Serial (Fast!)

1. Build Rust extension (one time):
   ```bash
   maturin develop --release
   ```

2. Edit `sa_config.py`:
   ```python
   ALGORITHM = 'rust_serial'
   ```

3. Run:
   ```bash
   python run_experiment.py
   ```

### Use Rust Parallel (Fastest!)

1. Build Rust extension (one time):
   ```bash
   maturin develop --release
   ```

2. Edit `sa_config.py`:
   ```python
   ALGORITHM = 'rust_parallel'
   ```

3. Run:
   ```bash
   python run_experiment.py
   python run_grid_search.py
   ```

## Which Scripts Use sa_config?

**ALL** the main scripts in the repository use `sa_config.py`:

- ✅ `run_experiment.py` - PPO training
- ✅ `run_grid_search.py` - Grid search analysis
- ✅ `tuning_env.py` - SA environment
- ✅ `grid_search_analysis.py` - Grid search implementation

When you change `ALGORITHM` in `sa_config.py`, **all** these scripts automatically use the new implementation!

## Verification

To verify which algorithm is currently configured, you can:

```bash
python -c "from sa_config import get_algorithm_name; print(f'Current algorithm: {get_algorithm_name()}')"
```

Or simply run any script - it will print at startup:
```
[SA Config] Using algorithm: rust_parallel
```

## Performance Comparison

| Algorithm        | Speedup | Best For |
|------------------|---------|----------|
| python_serial    | 1x      | Debugging, no extra dependencies |
| python_parallel  | 1.7x    | Multi-core Python without Rust |
| rust_serial      | 91x     | Single-threaded high performance |
| rust_parallel    | 171x    | Maximum performance (RECOMMENDED) |

## Troubleshooting

### "ModuleNotFoundError: No module named 'sa_rust'"

You're trying to use a Rust implementation but haven't built it yet.

**Solution:**
```bash
# Install maturin if needed
pip install maturin

# Build Rust extension
maturin develop --release
```

### "ImportError: cannot import name 'rust_parallel'"

The Rust module isn't available. Either:
1. Build it with `maturin develop --release`, OR
2. Switch to Python implementation in `sa_config.py`

### Want to use different algorithms in different scripts?

Instead of using `sa_config.py`, import directly:

```python
# In your script
from sa_algorithms import python_serial as sa
# or
from sa_algorithms import rust_parallel as sa

# Then use it
avg_reward, costs, trajectory, median_idx = sa.run_sa(...)
```

## Advanced: Using Multiple Algorithms

If you want to compare algorithms in the same script:

```python
from sa_algorithms import python_serial, python_parallel, rust_serial, rust_parallel

# Compare them
for name, algo in [('Python Serial', python_serial), 
                    ('Rust Parallel', rust_parallel)]:
    result = algo.run_sa(init_temp, cooling_rate, step_size, 
                         num_steps, bounds, seed, num_runs)
    print(f"{name}: {result[0]}")
```

See `comprehensive_benchmark.py` for a complete example.
