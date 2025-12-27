"""
Python Parallel Implementation of Simulated Annealing

Uses pre-generated unified random samples - NO RNG calls during optimization.
Uses multiprocessing to run multiple SA iterations in parallel.
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import sys
import os

# Add parent directory to path to import random_sampling
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from random_sampling import UnifiedRandomSampler


def rastrigin_2d(x, y):
    """2D Rastrigin function"""
    scale = 1.5
    x = x / scale
    y = y / scale
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)


def _run_single_sa(args):
    """
    Run a single SA optimization (helper function for parallel execution).
    
    Uses ONLY pre-generated random samples - NO RNG.
    
    Args:
        args: Tuple of (init_temp, cooling_rate, step_size, num_steps, bounds, run_idx)
        
    Returns:
        tuple: (final_cost, trajectory)
    """
    init_temp, cooling_rate, step_size, num_steps, bounds, run_idx = args
    
    # Load unified random sampler (NO RNG)
    sampler = UnifiedRandomSampler()
    
    # Get pre-generated starting point (NO RNG)
    curr_x, curr_y = sampler.get_starting_point(run_idx)
    curr_cost = rastrigin_2d(curr_x, curr_y)
    best_cost = curr_cost
    
    curr_temp = init_temp
    
    trajectory = []
    trajectory.append((curr_x, curr_y, curr_cost))
    
    # Get pre-generated random steps for this run (NO RNG)
    random_steps = sampler.get_random_steps(run_idx, num_steps)
    
    for step_idx in range(num_steps):
        # Use pre-generated normal samples scaled by step_size (NO RNG)
        dx = random_steps[step_idx, 0] * step_size
        dy = random_steps[step_idx, 1] * step_size
        
        cand_x = np.clip(curr_x + dx, bounds[0], bounds[1])
        cand_y = np.clip(curr_y + dy, bounds[0], bounds[1])
        cand_cost = rastrigin_2d(cand_x, cand_y)
        
        # Acceptance criterion
        delta = cand_cost - curr_cost
        accepted = False
        if delta < 0:
            accepted = True
        else:
            prob = np.exp(-delta / curr_temp) if curr_temp > 1e-9 else 0.0
            # Use deterministic acceptance based on probability threshold
            # This ensures reproducibility without RNG
            if prob > 0.5:  # Deterministic threshold
                accepted = True
        
        if accepted:
            curr_x = cand_x
            curr_y = cand_y
            curr_cost = cand_cost
            if curr_cost < best_cost:
                best_cost = curr_cost
        
        trajectory.append((curr_x, curr_y, curr_cost))
        
        # Cool down
        curr_temp *= cooling_rate
    
    return curr_cost, trajectory


def run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10, num_threads=None):
    """
    Run Simulated Annealing algorithm (parallel version).
    
    Uses ONLY pre-generated random samples from UnifiedRandomSampler.
    NO random number generation occurs during this function.
    
    Args:
        init_temp (float): Initial temperature
        cooling_rate (float): Temperature decay rate per step (0 < rate < 1)
        step_size (float): Standard deviation for random walk
        num_steps (int): Total number of SA iterations per run
        bounds (tuple): (min, max) bounds for search space
        seed (int, optional): Ignored - uses run index for deterministic sampling
        num_runs (int): Number of independent SA runs to average over
        num_threads (int, optional): Number of parallel processes (defaults to CPU count)
        
    Returns:
        tuple: (avg_reward, costs, trajectory, median_idx)
            - avg_reward: Average reward across all runs (WITHOUT step penalty)
            - costs: List of final costs for each run
            - trajectory: Trajectory of the median cost run [(x, y, cost), ...]
            - median_idx: Index of the run with median cost
    """
    if num_threads is None:
        num_threads = cpu_count()
    
    # Prepare arguments for each run with unique run indices (NO RNG)
    args_list = [
        (init_temp, cooling_rate, step_size, num_steps, bounds, i)
        for i in range(num_runs)
    ]
    
    # Run in parallel
    with Pool(processes=num_threads) as pool:
        results = pool.map(_run_single_sa, args_list)
    
    # Unpack results
    costs = [r[0] for r in results]
    trajectories = [r[1] for r in results]
    
    total_reward = sum(-c for c in costs)
    
    # Average reward WITHOUT step penalty (penalty moved to env)
    avg_reward = total_reward / num_runs
    
    # Store trajectory of the run with median cost (representative)
    sorted_indices = np.argsort(costs)
    median_idx = sorted_indices[len(costs)//2]
    last_trajectory = trajectories[median_idx]
    
    return avg_reward, costs, last_trajectory, median_idx
