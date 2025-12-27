"""
Python Serial Implementation of Simulated Annealing

This is a clean, standalone implementation that can be used as a reference.
"""

import numpy as np
from ..random_sampling import UnifiedRandomSampler


def rastrigin_2d(x, y):
    """2D Rastrigin function"""
    scale = 1.5
    x = x / scale
    y = y / scale
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)


def run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10):
    """
    Run Simulated Annealing algorithm (serial version).
    
    Args:
        init_temp (float): Initial temperature
        cooling_rate (float): Temperature decay rate per step (0 < rate < 1)
        step_size (float): Standard deviation for random walk
        num_steps (int): Total number of SA iterations per run
        bounds (tuple): (min, max) bounds for search space
        seed (int, optional): Random seed for reproducibility (ignored - uses pre-computed samples)
        num_runs (int): Number of independent SA runs to average over
        
    Returns:
        tuple: (avg_reward, costs, trajectory, median_idx)
            - avg_reward: Average reward across all runs with penalty
            - costs: List of final costs for each run
            - trajectory: Trajectory of the median cost run [(x, y, cost), ...]
            - median_idx: Index of the run with median cost
    """
    # Use pre-computed random samples from UnifiedRandomSampler
    sampler = UnifiedRandomSampler()
    
    total_reward = 0
    costs = []
    trajectories = []
    
    for run_idx in range(num_runs):
        # Get starting position from pre-computed samples
        curr_x, curr_y = sampler.get_starting_point(run_idx)
        curr_cost = rastrigin_2d(curr_x, curr_y)
        best_cost = curr_cost
        
        curr_temp = init_temp
        
        # Get random steps and acceptance probs for this run
        random_steps = sampler.get_random_steps(run_idx, num_steps)
        acceptance_probs = sampler.get_acceptance_probs(run_idx, num_steps)
        
        trajectory = []
        trajectory.append((curr_x, curr_y, curr_cost))
        
        for step_idx in range(num_steps):
            # Use pre-computed random steps
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
                # Use pre-computed acceptance probability
                if acceptance_probs[step_idx] < prob:
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
        
        costs.append(curr_cost)
        trajectories.append(trajectory)
        total_reward += (-curr_cost)
    
    # Average reward (no penalty - penalty is handled in environment)
    avg_reward = (total_reward / num_runs)
    
    # Store trajectory of the run with median cost (representative)
    sorted_indices = np.argsort(costs)
    median_idx = sorted_indices[len(costs)//2]
    last_trajectory = trajectories[median_idx]
    
    return avg_reward, costs, last_trajectory, median_idx
