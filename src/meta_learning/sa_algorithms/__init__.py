
"""
Unified Simulated Annealing Interface

This module provides a single interface to run different Simulated Annealing (SA) algorithm implementations.
Select the implementation by specifying the algorithm name.
"""

import importlib

_ALGORITHMS = {
    'python_serial': 'python_serial',
    'python_parallel': 'python_parallel',
    'rust_serial': 'rust_serial',
    'rust_parallel': 'rust_parallel',
}

def run_sa(
    algorithm: str,
    init_temp,
    cooling_rate,
    step_size,
    num_steps,
    bounds,
    seed=None,
    num_runs=10,
    num_threads=None
):
    """
    Run Simulated Annealing using the specified implementation.

    Args:
        algorithm (str): One of 'python_serial', 'python_parallel', 'rust_serial', 'rust_parallel'
        init_temp (float): Initial temperature
        cooling_rate (float): Temperature decay rate per step (0 < rate < 1)
        step_size (float): Standard deviation for random walk
        num_steps (int): Number of SA iterations per run
        bounds (tuple): (min, max) bounds for search space
        seed (int, optional): Random seed for reproducibility
        num_runs (int): Number of independent SA runs
        num_threads (int, optional): Number of parallel threads (parallel versions only)

    Returns:
        tuple: (avg_reward, costs, trajectory, median_idx)
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(_ALGORITHMS.keys())}")
    module = importlib.import_module(f".{{}}".format(_ALGORITHMS[algorithm]), __name__)
    if algorithm in ('python_parallel', 'rust_parallel'):
        return module.run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed, num_runs, num_threads)
    else:
        return module.run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed, num_runs)

__all__ = ['run_sa']
