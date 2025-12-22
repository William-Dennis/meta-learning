"""
Wrapper to use Rust SA implementation with existing Python code.

This module provides a drop-in replacement for sa_algorithm.py that uses
the Rust implementation when available, falling back to Python if not.
"""

import numpy as np

# Try to import Rust implementation
try:
    import sa_rust
    RUST_AVAILABLE = True
    print("✓ Using Rust-accelerated SA implementation (100x faster)")
except ImportError:
    RUST_AVAILABLE = False
    print("⚠ Rust implementation not available, using Python")
    import sa_algorithm as sa_python


def rastrigin_2d(x, y):
    """2D Rastrigin function"""
    if RUST_AVAILABLE:
        # Rust has this function as rastrigin_2d_py
        return sa_rust.rastrigin_2d_py(float(x), float(y))
    else:
        return sa_python.rastrigin_2d(x, y)


def run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10):
    """
    Runs Simulated Annealing logic.
    
    Automatically uses Rust implementation if available for ~100x speedup.
    
    Args:
        init_temp: Initial temperature
        cooling_rate: Temperature decay rate per step  
        step_size: Standard deviation for random walk
        num_steps: Total number of SA iterations
        bounds: [min, max] bounds for search space (will be converted to tuple for Rust)
        seed: Random seed (optional)
        num_runs: Number of SA runs to average over (default 10)
        
    Returns:
        (avg_reward, costs, trajectory, median_idx)
    """
    if RUST_AVAILABLE:
        # Convert bounds to tuple for Rust
        if isinstance(bounds, list):
            bounds = tuple(bounds)
        
        # Call Rust implementation
        return sa_rust.run_sa(
            float(init_temp),
            float(cooling_rate),
            float(step_size),
            int(num_steps),
            bounds,
            int(seed) if seed is not None else None,
            int(num_runs)
        )
    else:
        # Fall back to Python implementation
        return sa_python.run_sa(
            init_temp, cooling_rate, step_size, num_steps, 
            bounds, seed, num_runs
        )
