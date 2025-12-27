"""
Python Serial Implementation of Simulated Annealing

This is a clean, standalone implementation that can be used as a reference.
"""

import numpy as np


def rastrigin_2d(x, y):
    """2D Rastrigin function"""
    scale = 1.5
    x = x / scale
    y = y / scale
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)


def run_sa(
    init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10
):
    """Run Simulated Annealing algorithm (serial version)."""

    np_random = np.random.default_rng(seed)

    total_reward = 0
    costs = []
    trajectories = []

    for _ in range(num_runs):
        cost, traj = _run_single_sa(
            init_temp, cooling_rate, step_size, num_steps, bounds, np_random
        )
        costs.append(cost)
        trajectories.append(traj)
        total_reward += -cost

    avg_reward = (total_reward / num_runs) - (num_steps / 1000) + 10

    sorted_indices = np.argsort(costs)
    median_idx = sorted_indices[len(costs) // 2]
    last_trajectory = trajectories[median_idx]

    return avg_reward, costs, last_trajectory, median_idx


def _run_single_sa(init_temp, cooling_rate, step_size, num_steps, bounds, np_random):
    """Run single SA optimization."""
    curr_x = np_random.uniform(bounds[0], bounds[1])
    curr_y = np_random.uniform(bounds[0], bounds[1])
    curr_cost = rastrigin_2d(curr_x, curr_y)
    best_cost = curr_cost
    curr_temp = init_temp

    trajectory = [(curr_x, curr_y, curr_cost)]

    for _ in range(num_steps):
        curr_x, curr_y, curr_cost, curr_temp, best_cost = _sa_step(
            curr_x,
            curr_y,
            curr_cost,
            curr_temp,
            best_cost,
            step_size,
            cooling_rate,
            bounds,
            np_random,
        )
        trajectory.append((curr_x, curr_y, curr_cost))

    return curr_cost, trajectory


def _sa_step(
    curr_x,
    curr_y,
    curr_cost,
    curr_temp,
    best_cost,
    step_size,
    cooling_rate,
    bounds,
    np_random,
):
    """Execute one SA step."""
    dx = np_random.normal(0, step_size)
    dy = np_random.normal(0, step_size)

    cand_x = np.clip(curr_x + dx, bounds[0], bounds[1])
    cand_y = np.clip(curr_y + dy, bounds[0], bounds[1])
    cand_cost = rastrigin_2d(cand_x, cand_y)

    if _accept_move(curr_cost, cand_cost, curr_temp, np_random):
        curr_x, curr_y, curr_cost = cand_x, cand_y, cand_cost
        if curr_cost < best_cost:
            best_cost = curr_cost

    curr_temp *= cooling_rate
    return curr_x, curr_y, curr_cost, curr_temp, best_cost


def _accept_move(curr_cost, cand_cost, temp, np_random):
    """Determine if move should be accepted."""
    delta = cand_cost - curr_cost
    if delta < 0:
        return True
    prob = np.exp(-delta / temp) if temp > 1e-9 else 0.0
    return np_random.random() < prob
