"""
Python Parallel Implementation of Simulated Annealing

Uses multiprocessing to run multiple SA iterations in parallel.
"""

import numpy as np
from multiprocessing import Pool, cpu_count


def rastrigin_2d(x, y):
    """2D Rastrigin function"""
    scale = 1.5
    x = x / scale
    y = y / scale
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)


def _run_single_sa(args):
    """
    Run a single SA optimization (helper function for parallel execution).
    
    Args:
        args: Tuple of (init_temp, cooling_rate, step_size, num_steps, bounds, seed)
        
    Returns:
        tuple: (final_cost, trajectory)
    """
    init_temp, cooling_rate, step_size, num_steps, bounds, seed = args
    
    np_random = np.random.default_rng(seed)
    
    # Random start position
    curr_x = np_random.uniform(bounds[0], bounds[1])
    curr_y = np_random.uniform(bounds[0], bounds[1])
    curr_cost = rastrigin_2d(curr_x, curr_y)
    best_cost = curr_cost
    
    curr_temp = init_temp
    
    trajectory = []
    trajectory.append((curr_x, curr_y, curr_cost))
    
    for _ in range(num_steps):
        # Generate neighbor using normal distribution
        dx = np_random.normal(0, step_size)
        dy = np_random.normal(0, step_size)
        
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
            if np_random.random() < prob:
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
    
    Args:
        init_temp (float): Initial temperature
        cooling_rate (float): Temperature decay rate per step (0 < rate < 1)
        step_size (float): Standard deviation for random walk
        num_steps (int): Total number of SA iterations per run
        bounds (tuple): (min, max) bounds for search space
        seed (int, optional): Random seed for reproducibility
        num_runs (int): Number of independent SA runs to average over
        num_threads (int, optional): Number of parallel processes (defaults to CPU count)
        
    Returns:
        tuple: (avg_reward, costs, trajectory, median_idx)
            - avg_reward: Average reward across all runs with penalty
            - costs: List of final costs for each run
            - trajectory: Trajectory of the median cost run [(x, y, cost), ...]
            - median_idx: Index of the run with median cost
    """
    if num_threads is None:
        num_threads = cpu_count()
    
    # Prepare arguments for each run with unique seeds
    base_seed = seed if seed is not None else np.random.randint(0, 2**31)
    args_list = [
        (init_temp, cooling_rate, step_size, num_steps, bounds, base_seed + i)
        for i in range(num_runs)
    ]
    
    # Run in parallel
    with Pool(processes=num_threads) as pool:
        results = pool.map(_run_single_sa, args_list)
    
    # Unpack results
    costs = [r[0] for r in results]
    trajectories = [r[1] for r in results]
    
    total_reward = sum(-c for c in costs)
    
    # Average reward with penalty based on number of steps
    avg_reward = (total_reward / num_runs) - (num_steps / 1000) + 10
    
    # Store trajectory of the run with median cost (representative)
    sorted_indices = np.argsort(costs)
    median_idx = sorted_indices[len(costs)//2]
    last_trajectory = trajectories[median_idx]
    
    return avg_reward, costs, last_trajectory, median_idx
