"""
Benchmark script comparing Python and Rust implementations of SA algorithm.
Tests performance across different numbers of steps.
"""

import time
import numpy as np
import sa_algorithm as sa_python
import sa_rust
from tabulate import tabulate

# Test parameters
INIT_TEMP = 10.0
COOLING_RATE = 0.95
STEP_SIZE = 1.0
BOUNDS = (-5.12, 5.12)  # Tuple for Rust compatibility
SEED = 42
NUM_RUNS = 10

# Different step counts to test
STEP_COUNTS = [10, 50, 100, 200, 500, 1000, 2000, 5000]


def benchmark_python(num_steps):
    """Benchmark Python implementation"""
    start = time.perf_counter()
    avg_reward, costs, trajectory, median_idx = sa_python.run_sa(
        INIT_TEMP, COOLING_RATE, STEP_SIZE, num_steps, BOUNDS, SEED, NUM_RUNS
    )
    elapsed = time.perf_counter() - start
    return elapsed, avg_reward, np.mean(costs)


def benchmark_rust(num_steps):
    """Benchmark Rust implementation"""
    start = time.perf_counter()
    avg_reward, costs, trajectory, median_idx = sa_rust.run_sa(
        INIT_TEMP, COOLING_RATE, STEP_SIZE, num_steps, BOUNDS, SEED, NUM_RUNS
    )
    elapsed = time.perf_counter() - start
    return elapsed, avg_reward, np.mean(costs)


def main():
    print("=" * 80)
    print("BENCHMARK: Python vs Rust SA Implementation")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Init Temp: {INIT_TEMP}")
    print(f"  Cooling Rate: {COOLING_RATE}")
    print(f"  Step Size: {STEP_SIZE}")
    print(f"  Bounds: {BOUNDS}")
    print(f"  Seed: {SEED}")
    print(f"  Runs per test: {NUM_RUNS}")
    print()
    
    results = []
    
    for num_steps in STEP_COUNTS:
        print(f"Testing {num_steps} steps...")
        
        # Warmup
        benchmark_python(10)
        benchmark_rust(10)
        
        # Benchmark Python
        py_time, py_reward, py_cost = benchmark_python(num_steps)
        
        # Benchmark Rust
        rust_time, rust_reward, rust_cost = benchmark_rust(num_steps)
        
        # Calculate speedup
        speedup = py_time / rust_time if rust_time > 0 else float('inf')
        
        results.append({
            'steps': num_steps,
            'py_time': py_time,
            'rust_time': rust_time,
            'speedup': speedup,
            'py_cost': py_cost,
            'rust_cost': rust_cost,
            'cost_diff': abs(py_cost - rust_cost)
        })
        
        print(f"  Python: {py_time:.4f}s | Rust: {rust_time:.4f}s | Speedup: {speedup:.2f}x")
    
    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    
    table_data = []
    for r in results:
        table_data.append([
            r['steps'],
            f"{r['py_time']:.4f}",
            f"{r['rust_time']:.4f}",
            f"{r['speedup']:.2f}x",
            f"{r['py_cost']:.4f}",
            f"{r['rust_cost']:.4f}",
            f"{r['cost_diff']:.6f}"
        ])
    
    headers = ['Steps', 'Python (s)', 'Rust (s)', 'Speedup', 'Py Cost', 'Rust Cost', 'Cost Diff']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    avg_speedup = np.mean([r['speedup'] for r in results])
    max_speedup = max([r['speedup'] for r in results])
    min_speedup = min([r['speedup'] for r in results])
    
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Max Speedup: {max_speedup:.2f}x")
    print(f"Min Speedup: {min_speedup:.2f}x")
    print(f"\nRust implementation is {avg_speedup:.1f}x faster on average!")
    
    # Check correctness
    max_cost_diff = max([r['cost_diff'] for r in results])
    print(f"\nMax cost difference: {max_cost_diff:.6f}")
    if max_cost_diff < 0.1:
        print("✓ Results match between Python and Rust implementations!")
    else:
        print("⚠ Significant difference detected - check implementations")


if __name__ == '__main__':
    main()
