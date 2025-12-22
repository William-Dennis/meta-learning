import numpy as np
import time

def rastrigin_2d(x, y):
    # 2D Rastrigin function
    scale = 1.5
    x = x / scale
    y = y / scale
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

def run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10):
    """
    Runs Simulated Annealing logic.
    
    Args:
        num_runs: Number of SA runs to average over (default 10)
    """
    if seed is None:
        np_random = np.random
    else:
        np_random = np.random.default_rng(seed)
    total_reward = 0
    costs = []
    trajectories = []
    
    # s = time.perf_counter()
    for _ in range(num_runs):
        # Random start
        curr_x = np_random.uniform(bounds[0], bounds[1])
        curr_y = np_random.uniform(bounds[0], bounds[1])
        curr_cost = rastrigin_2d(curr_x, curr_y)
        best_cost = curr_cost
        
        curr_temp = init_temp
        
        trajectory = []
        trajectory.append((curr_x, curr_y, curr_cost))
        
        for _ in range(num_steps):
            # Neighbor
            dx = np_random.normal(0, step_size)
            dy = np_random.normal(0, step_size)
            
            cand_x = np.clip(curr_x + dx, bounds[0], bounds[1])
            cand_y = np.clip(curr_y + dy, bounds[0], bounds[1])
            cand_cost = rastrigin_2d(cand_x, cand_y)
            
            # Accept?
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
            
        costs.append(curr_cost)
        trajectories.append(trajectory)
        total_reward += (-curr_cost)
        
    # Average reward
    # avg_reward = (total_reward / num_runs) - (1 + (time.perf_counter() - s))**2
    avg_reward = (total_reward / num_runs) - (num_steps / 1000) + 10
    
    # Store trajectory of the run with median cost (representative)
    sorted_indices = np.argsort(costs)
    median_idx = sorted_indices[len(costs)//2]
    last_trajectory = trajectories[median_idx]
    
    return avg_reward, costs, last_trajectory, median_idx
