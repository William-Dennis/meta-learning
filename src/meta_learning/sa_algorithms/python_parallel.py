"""
Python Parallel Implementation of Simulated Annealing

Uses multiprocessing to run multiple SA iterations in parallel.
"""

from multiprocessing import Pool, cpu_count
from ._common import (
    rastrigin_2d, 
    with_samples, 
    run_single_sa_iteration,
    compute_result
)


def _run_single_sa(args):
    """Helper function for parallel execution."""
    (init_temp, cooling_rate, step_size, num_steps, bounds, run_idx, 
     starting_points, random_steps, acceptance_probs) = args
    
    return run_single_sa_iteration(
        run_idx, init_temp, cooling_rate, step_size, num_steps, bounds,
        starting_points, random_steps, acceptance_probs
    )


@with_samples
def run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, 
           num_runs=10, num_threads=None, starting_points=None, random_steps=None, 
           acceptance_probs=None):
    """
    Run Simulated Annealing algorithm (parallel version).
    
    Args:
        init_temp (float): Initial temperature
        cooling_rate (float): Temperature decay rate per step (0 < rate < 1)
        step_size (float): Standard deviation for random walk
        num_steps (int): Total number of SA iterations per run
        bounds (tuple): (min, max) bounds for search space
        seed (int, optional): Random seed (ignored - uses pre-computed samples)
        num_runs (int): Number of independent SA runs to average over
        num_threads (int, optional): Number of parallel processes
        starting_points (ndarray, optional): Pre-computed starting points
        random_steps (ndarray, optional): Pre-computed random steps
        acceptance_probs (ndarray, optional): Pre-computed acceptance probs
        
    Returns:
        tuple: (avg_reward, costs, trajectory, median_idx)
    """
    if num_threads is None:
        num_threads = cpu_count()
    
    args_list = [
        (init_temp, cooling_rate, step_size, num_steps, bounds, i, 
         starting_points, random_steps, acceptance_probs)
        for i in range(num_runs)
    ]
    
    with Pool(processes=num_threads) as pool:
        results = pool.map(_run_single_sa, args_list)
    
    costs = [r[0] for r in results]
    trajectories = [r[1] for r in results]
    
    return compute_result(costs, trajectories, num_runs)
