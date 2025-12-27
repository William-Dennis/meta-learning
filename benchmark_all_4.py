#!/usr/bin/env python3
"""
Comprehensive benchmark comparing all 4 SA implementations.
Generates performance table with speedup analysis.
"""

import time
import numpy as np
import sa_algorithms.python_serial as python_serial
import sa_algorithms.python_parallel as python_parallel
import sa_algorithms.rust_serial as rust_serial
import sa_algorithms.rust_parallel as rust_parallel

# Test configurations
TEST_CONFIGS = [
    {"steps": 10, "runs": 100},
    {"steps": 50, "runs": 100},
    {"steps": 100, "runs": 100},
    {"steps": 500, "runs": 100},
    {"steps": 1000, "runs": 100},
    {"steps": 2000, "runs": 100},
    {"steps": 5000, "runs": 50},
]

# SA parameters
INIT_TEMP = 10.0
COOLING_RATE = 0.95
STEP_SIZE = 1.0
BOUNDS = (-5.12, 5.12)
SEED = 42

def benchmark_implementation(impl_name, run_func, num_steps, num_runs):
    """Benchmark a single implementation."""
    start_time = time.time()
    
    avg_reward, costs, trajectory, median_idx = run_func(
        init_temp=INIT_TEMP,
        cooling_rate=COOLING_RATE,
        step_size=STEP_SIZE,
        num_steps=num_steps,
        bounds=BOUNDS,
        seed=SEED,
        num_runs=num_runs
    )
    
    elapsed = time.time() - start_time
    
    return {
        'time': elapsed,
        'avg_reward': avg_reward,
        'mean_cost': np.mean(costs),
        'std_cost': np.std(costs)
    }

def main():
    print("=" * 100)
    print("COMPREHENSIVE BENCHMARK: All 4 SA Implementations")
    print("=" * 100)
    print()
    
    implementations = [
        ("Python Serial", python_serial.run_sa),
        ("Python Parallel", python_parallel.run_sa),
        ("Rust Serial", rust_serial.run_sa),
        ("Rust Parallel", rust_parallel.run_sa),
    ]
    
    all_results = []
    
    for config in TEST_CONFIGS:
        num_steps = config["steps"]
        num_runs = config["runs"]
        
        print(f"Testing with {num_steps} steps, {num_runs} runs...")
        
        row = {"steps": num_steps, "runs": num_runs}
        
        for impl_name, impl_func in implementations:
            print(f"  Running {impl_name}...", end=" ", flush=True)
            result = benchmark_implementation(impl_name, impl_func, num_steps, num_runs)
            row[impl_name] = result
            print(f"{result['time']:.4f}s (cost: {result['mean_cost']:.4f})")
        
        all_results.append(row)
        print()
    
    # Print performance table
    print("\n" + "=" * 100)
    print("PERFORMANCE TABLE")
    print("=" * 100)
    print()
    print(f"{'Steps':<8} {'Runs':<6} {'Python Serial':<16} {'Python Parallel':<16} {'Rust Serial':<16} {'Rust Parallel':<16}")
    print("-" * 100)
    
    for row in all_results:
        steps = row["steps"]
        runs = row["runs"]
        py_serial_time = row["Python Serial"]["time"]
        py_parallel_time = row["Python Parallel"]["time"]
        rust_serial_time = row["Rust Serial"]["time"]
        rust_parallel_time = row["Rust Parallel"]["time"]
        
        print(f"{steps:<8} {runs:<6} {py_serial_time:>12.4f}s   {py_parallel_time:>12.4f}s   {rust_serial_time:>12.4f}s   {rust_parallel_time:>12.4f}s")
    
    # Print speedup table
    print("\n" + "=" * 100)
    print("SPEEDUP TABLE (vs Python Serial)")
    print("=" * 100)
    print()
    print(f"{'Steps':<8} {'Runs':<6} {'Python Parallel':<18} {'Rust Serial':<18} {'Rust Parallel':<18}")
    print("-" * 100)
    
    for row in all_results:
        steps = row["steps"]
        runs = row["runs"]
        py_serial_time = row["Python Serial"]["time"]
        py_parallel_time = row["Python Parallel"]["time"]
        rust_serial_time = row["Rust Serial"]["time"]
        rust_parallel_time = row["Rust Parallel"]["time"]
        
        speedup_py_parallel = py_serial_time / py_parallel_time if py_parallel_time > 0 else 0
        speedup_rust_serial = py_serial_time / rust_serial_time if rust_serial_time > 0 else 0
        speedup_rust_parallel = py_serial_time / rust_parallel_time if rust_parallel_time > 0 else 0
        
        print(f"{steps:<8} {runs:<6} {speedup_py_parallel:>14.2f}x     {speedup_rust_serial:>14.2f}x     {speedup_rust_parallel:>14.2f}x")
    
    # Calculate and print average speedups
    print("\n" + "=" * 100)
    print("AVERAGE SPEEDUPS (excluding 10-step outlier)")
    print("=" * 100)
    print()
    
    # Exclude first row (10 steps) as it's an outlier due to overhead
    relevant_results = all_results[1:]
    
    avg_speedup_py_parallel = np.mean([
        row["Python Serial"]["time"] / row["Python Parallel"]["time"]
        for row in relevant_results
    ])
    
    avg_speedup_rust_serial = np.mean([
        row["Python Serial"]["time"] / row["Rust Serial"]["time"]
        for row in relevant_results
    ])
    
    avg_speedup_rust_parallel = np.mean([
        row["Python Serial"]["time"] / row["Rust Parallel"]["time"]
        for row in relevant_results
    ])
    
    print(f"Python Parallel:  {avg_speedup_py_parallel:.2f}x")
    print(f"Rust Serial:      {avg_speedup_rust_serial:.2f}x")
    print(f"Rust Parallel:    {avg_speedup_rust_parallel:.2f}x")
    print()
    
    # Verify determinism
    print("=" * 100)
    print("DETERMINISM VERIFICATION (100 steps, 10 runs)")
    print("=" * 100)
    print()
    
    test_params = {
        "init_temp": INIT_TEMP,
        "cooling_rate": COOLING_RATE,
        "step_size": STEP_SIZE,
        "num_steps": 100,
        "bounds": BOUNDS,
        "seed": SEED,
        "num_runs": 10
    }
    
    results = []
    for impl_name, impl_func in implementations:
        avg_reward, costs, _, _ = impl_func(**test_params)
        results.append((impl_name, avg_reward, costs))
        print(f"{impl_name:20s}: avg_reward={avg_reward:.6f}, mean_cost={np.mean(costs):.6f}")
    
    # Check if all implementations produce identical results
    base_costs = results[0][2]
    all_identical = all(np.allclose(costs, base_costs, rtol=1e-9, atol=1e-9) for _, _, costs in results)
    
    if all_identical:
        print("\n✅ All implementations produce IDENTICAL results (fully deterministic)")
    else:
        print("\n⚠️  Warning: Implementations produce slightly different results")
        print("   This may be due to floating-point precision differences")
    
    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
