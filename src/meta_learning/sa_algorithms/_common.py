"""
Common utilities for SA algorithms.
"""

import numpy as np
from functools import wraps


def with_samples(func):
    """Decorator to load pre-computed samples if not provided."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if samples are provided in kwargs
        if (kwargs.get('starting_points') is None or 
            kwargs.get('random_steps') is None or 
            kwargs.get('acceptance_probs') is None):
            from ..random_sampling import load_random_samples
            starting_points, random_steps, acceptance_probs = load_random_samples()
            kwargs['starting_points'] = starting_points
            kwargs['random_steps'] = random_steps
            kwargs['acceptance_probs'] = acceptance_probs
        return func(*args, **kwargs)
    return wrapper


def rastrigin_2d(x, y):
    """2D Rastrigin function"""
    scale = 1.5
    x = x / scale
    y = y / scale
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)


def run_sa_step(curr_x, curr_y, curr_cost, curr_temp, step_idx, run_idx,
                step_size, bounds, random_steps, acceptance_probs):
    """
    Execute a single SA step.
    
    Returns:
        tuple: (new_x, new_y, new_cost, accepted)
    """
    # Use pre-computed random steps
    dx = random_steps[step_idx, run_idx, 0] * step_size
    dy = random_steps[step_idx, run_idx, 1] * step_size
    
    cand_x = np.clip(curr_x + dx, bounds[0], bounds[1])
    cand_y = np.clip(curr_y + dy, bounds[0], bounds[1])
    cand_cost = rastrigin_2d(cand_x, cand_y)
    
    # Acceptance criterion
    delta = cand_cost - curr_cost
    if delta < 0:
        return cand_x, cand_y, cand_cost, True
    
    prob = np.exp(-delta / curr_temp) if curr_temp > 1e-9 else 0.0
    accepted = acceptance_probs[step_idx, run_idx] < prob
    
    if accepted:
        return cand_x, cand_y, cand_cost, True
    return curr_x, curr_y, curr_cost, False


def run_single_sa_iteration(run_idx, init_temp, cooling_rate, step_size, 
                             num_steps, bounds, starting_points, random_steps, 
                             acceptance_probs):
    """Run a single SA optimization iteration."""
    curr_x, curr_y = starting_points[run_idx]
    curr_cost = rastrigin_2d(curr_x, curr_y)
    best_cost = curr_cost
    curr_temp = init_temp
    
    trajectory = [(curr_x, curr_y, curr_cost)]
    
    for step_idx in range(num_steps):
        curr_x, curr_y, curr_cost, accepted = run_sa_step(
            curr_x, curr_y, curr_cost, curr_temp, step_idx, run_idx,
            step_size, bounds, random_steps, acceptance_probs
        )
        
        if accepted and curr_cost < best_cost:
            best_cost = curr_cost
        
        trajectory.append((curr_x, curr_y, curr_cost))
        curr_temp *= cooling_rate
    
    return curr_cost, trajectory


def compute_result(costs, trajectories, num_runs):
    """Compute final SA result from costs and trajectories."""
    total_reward = sum(-c for c in costs)
    avg_reward = total_reward / num_runs
    
    sorted_indices = np.argsort(costs)
    median_idx = sorted_indices[len(costs) // 2]
    median_trajectory = trajectories[median_idx]
    
    return avg_reward, costs, median_trajectory, median_idx
