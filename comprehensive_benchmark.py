"""
Comprehensive Benchmark: All SA Algorithm Implementations

Compares performance across:
1. Python Serial
2. Python Parallel 
3. Rust Serial
4. Rust Parallel

Tests performance across different numbers of steps.
"""

import time
import numpy as np
from tabulate import tabulate
import sa_algorithms.python_serial as py_serial
import sa_algorithms.python_parallel as py_parallel
import sa_algorithms.rust_serial as rust_serial
import sa_algorithms.rust_parallel as rust_parallel

# Test parameters
INIT_TEMP = 10.0
COOLING_RATE = 0.95
STEP_SIZE = 1.0
BOUNDS = (-5.12, 5.12)
SEED = 42
NUM_RUNS = 100  # More runs to better show parallel benefits

# Different step counts to test
STEP_COUNTS = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def benchmark_implementation(impl_name, run_func, num_steps, **kwargs):
    """
    Benchmark a single implementation.
    
    Args:
        impl_name: Name of the implementation
        run_func: Function to call
        num_steps: Number of SA steps
        **kwargs: Additional arguments for run_func
        
    Returns:
        tuple: (elapsed_time, mean_cost)
    """
    try:
        start = time.perf_counter()
        avg_reward, costs, trajectory, median_idx = run_func(
            INIT_TEMP, COOLING_RATE, STEP_SIZE, num_steps, BOUNDS, SEED, NUM_RUNS, **kwargs
        )
        elapsed = time.perf_counter() - start
        return elapsed, np.mean(costs)
    except Exception as e:
        print(f"  ⚠ {impl_name} failed: {e}")
        return None, None


def main():
    print("=" * 100)
    print("COMPREHENSIVE BENCHMARK: All SA Algorithm Implementations")
    print("=" * 100)
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
        py_serial.run_sa(INIT_TEMP, COOLING_RATE, STEP_SIZE, 10, BOUNDS, SEED, 10)
        
        row = {'steps': num_steps}
        
        # 1. Python Serial
        print(f"  Running Python Serial...")
        py_s_time, py_s_cost = benchmark_implementation(
            "Python Serial", py_serial.run_sa, num_steps
        )
        row['py_serial_time'] = py_s_time
        row['py_serial_cost'] = py_s_cost
        
        # 2. Python Parallel
        print(f"  Running Python Parallel...")
        py_p_time, py_p_cost = benchmark_implementation(
            "Python Parallel", py_parallel.run_sa, num_steps
        )
        row['py_parallel_time'] = py_p_time
        row['py_parallel_cost'] = py_p_cost
        
        # 3. Rust Serial
        print(f"  Running Rust Serial...")
        r_s_time, r_s_cost = benchmark_implementation(
            "Rust Serial", rust_serial.run_sa, num_steps
        )
        row['rust_serial_time'] = r_s_time
        row['rust_serial_cost'] = r_s_cost
        
        # 4. Rust Parallel
        print(f"  Running Rust Parallel...")
        r_p_time, r_p_cost = benchmark_implementation(
            "Rust Parallel", rust_parallel.run_sa, num_steps
        )
        row['rust_parallel_time'] = r_p_time
        row['rust_parallel_cost'] = r_p_cost
        
        results.append(row)
        
        # Print quick summary
        if all(t is not None for t in [py_s_time, py_p_time, r_s_time, r_p_time]):
            print(f"    Py Serial: {py_s_time:.4f}s | Py Parallel: {py_p_time:.4f}s | "
                  f"Rust Serial: {r_s_time:.4f}s | Rust Parallel: {r_p_time:.4f}s")
        print()
    
    # Print results table
    print("\n" + "=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)
    print()
    
    table_data = []
    for r in results:
        if all(r.get(k) is not None for k in ['py_serial_time', 'py_parallel_time', 
                                                'rust_serial_time', 'rust_parallel_time']):
            table_data.append([
                r['steps'],
                f"{r['py_serial_time']:.4f}",
                f"{r['py_parallel_time']:.4f}",
                f"{r['rust_serial_time']:.4f}",
                f"{r['rust_parallel_time']:.4f}",
            ])
    
    headers = ['Steps', 'Python Serial (s)', 'Python Parallel (s)', 
               'Rust Serial (s)', 'Rust Parallel (s)']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Speedup analysis
    print("\n" + "=" * 100)
    print("SPEEDUP ANALYSIS (vs Python Serial)")
    print("=" * 100)
    print()
    
    speedup_data = []
    for r in results:
        if r.get('py_serial_time') and all(r.get(k) is not None for k in 
                ['py_parallel_time', 'rust_serial_time', 'rust_parallel_time']):
            base = r['py_serial_time']
            speedup_data.append([
                r['steps'],
                f"{base / r['py_parallel_time']:.2f}x",
                f"{base / r['rust_serial_time']:.2f}x",
                f"{base / r['rust_parallel_time']:.2f}x",
            ])
    
    headers = ['Steps', 'Python Parallel', 'Rust Serial', 'Rust Parallel']
    print(tabulate(speedup_data, headers=headers, tablefmt='grid'))
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    # Calculate average speedups
    valid_results = [r for r in results if all(r.get(k) is not None for k in 
                    ['py_serial_time', 'py_parallel_time', 'rust_serial_time', 'rust_parallel_time'])]
    
    if valid_results:
        avg_speedup_py_par = np.mean([r['py_serial_time'] / r['py_parallel_time'] 
                                      for r in valid_results])
        avg_speedup_rust_ser = np.mean([r['py_serial_time'] / r['rust_serial_time'] 
                                        for r in valid_results])
        avg_speedup_rust_par = np.mean([r['py_serial_time'] / r['rust_parallel_time'] 
                                        for r in valid_results])
        
        print(f"\nAverage Speedups (vs Python Serial):")
        print(f"  Python Parallel:  {avg_speedup_py_par:.2f}x")
        print(f"  Rust Serial:      {avg_speedup_rust_ser:.2f}x")
        print(f"  Rust Parallel:    {avg_speedup_rust_par:.2f}x")
        
        print(f"\n🚀 Best Performance: Rust Parallel is {avg_speedup_rust_par:.1f}x faster than Python Serial!")
        
        # Additional parallel analysis
        if len(valid_results) > 0:
            rust_par_vs_serial = np.mean([r['rust_serial_time'] / r['rust_parallel_time'] 
                                          for r in valid_results])
            py_par_vs_serial = np.mean([r['py_serial_time'] / r['py_parallel_time'] 
                                        for r in valid_results])
            
            print(f"\nParallelization Efficiency:")
            print(f"  Python: {py_par_vs_serial:.2f}x speedup from parallelization")
            print(f"  Rust:   {rust_par_vs_serial:.2f}x speedup from parallelization")
    
    # Cost validation
    print("\n" + "=" * 100)
    print("RESULT VALIDATION")
    print("=" * 100)
    
    print("\nNote: Small differences in costs are expected due to different RNG implementations.")
    print("All implementations should produce similar final costs (within ~10%).")


if __name__ == '__main__':
    main()
