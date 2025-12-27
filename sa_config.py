"""
SA Algorithm Configuration Module

This module provides a simple way to configure which SA algorithm implementation
to use across all scripts in the project. Simply change the ALGORITHM variable
to switch between implementations.

Available algorithms:
- 'python_serial': Pure Python serial implementation (baseline)
- 'python_parallel': Python with multiprocessing (~1.7x speedup)
- 'rust_serial': Rust serial implementation (~91x speedup) 
- 'rust_parallel': Rust parallel implementation (~171x speedup) - RECOMMENDED

Usage:
    from sa_config import get_sa_algorithm
    
    # Get the configured algorithm module
    sa = get_sa_algorithm()
    
    # Use it
    avg_reward, costs, trajectory, median_idx = sa.run_sa(
        init_temp, cooling_rate, step_size, num_steps, bounds, seed, num_runs
    )
"""

# =============================================================================
# CONFIGURATION: Change this to switch SA algorithm implementation
# =============================================================================
ALGORITHM = 'rust_parallel'  # Options: 'python_serial', 'python_parallel', 'rust_serial', 'rust_parallel'
# =============================================================================


def get_sa_algorithm():
    """
    Get the configured SA algorithm module.
    
    Returns:
        module: The SA algorithm module with run_sa() and rastrigin_2d() functions
    """
    if ALGORITHM == 'python_serial':
        from sa_algorithms import python_serial
        return python_serial
    elif ALGORITHM == 'python_parallel':
        from sa_algorithms import python_parallel
        return python_parallel
    elif ALGORITHM == 'rust_serial':
        from sa_algorithms import rust_serial
        return rust_serial
    elif ALGORITHM == 'rust_parallel':
        from sa_algorithms import rust_parallel
        return rust_parallel
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}. "
                        f"Must be one of: 'python_serial', 'python_parallel', 'rust_serial', 'rust_parallel'")


def get_algorithm_name():
    """Get the name of the currently configured algorithm."""
    return ALGORITHM


# Convenience exports - can be imported directly
sa = get_sa_algorithm()
run_sa = sa.run_sa
rastrigin_2d = sa.rastrigin_2d

# Print which algorithm is being used when module is imported
print(f"[SA Config] Using algorithm: {ALGORITHM}")
