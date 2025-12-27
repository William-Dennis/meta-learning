"""
Python Serial Implementation of Simulated Annealing

This is a clean, standalone implementation that can be used as a reference.
"""

from ._common import (
    rastrigin_2d, 
    with_samples, 
    run_single_sa_iteration,
    compute_result
)


@with_samples
def run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10, 
           starting_points=None, random_steps=None, acceptance_probs=None):
    """
    Run Simulated Annealing algorithm (serial version).
    
    Args:
        init_temp (float): Initial temperature
        cooling_rate (float): Temperature decay rate per step (0 < rate < 1)
        step_size (float): Standard deviation for random walk
        num_steps (int): Total number of SA iterations per run
        bounds (tuple): (min, max) bounds for search space
        seed (int, optional): Random seed (ignored - uses pre-computed samples)
        num_runs (int): Number of independent SA runs to average over
        starting_points (ndarray, optional): Pre-computed starting points
        random_steps (ndarray, optional): Pre-computed random steps
        acceptance_probs (ndarray, optional): Pre-computed acceptance probs
        
    Returns:
        tuple: (avg_reward, costs, trajectory, median_idx)
    """
    costs = []
    trajectories = []
    
    for run_idx in range(num_runs):
        cost, trajectory = run_single_sa_iteration(
            run_idx, init_temp, cooling_rate, step_size, num_steps, bounds,
            starting_points, random_steps, acceptance_probs
        )
        costs.append(cost)
        trajectories.append(trajectory)
    
    return compute_result(costs, trajectories, num_runs)
