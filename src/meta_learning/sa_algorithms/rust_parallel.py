"""
Rust Parallel Implementation of Simulated Annealing

Wrapper for the Rust parallel implementation via PyO3 bindings.
Uses Rust threads for high-performance parallel execution.
"""

try:
    import sa_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    raise ImportError(
        "Rust module 'sa_rust' not found. "
        "Please build it with: maturin develop --release"
    )


def rastrigin_2d(x, y):
    """2D Rastrigin function (uses Rust implementation)"""
    return sa_rust.rastrigin_2d_py(float(x), float(y))


def run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10, num_threads=None,
           starting_points=None, random_steps=None, acceptance_probs=None):
    """
    Run Simulated Annealing algorithm (Rust parallel version).
    
    Args:
        init_temp (float): Initial temperature
        cooling_rate (float): Temperature decay rate per step (0 < rate < 1)
        step_size (float): Standard deviation for random walk
        num_steps (int): Total number of SA iterations per run
        bounds (tuple): (min, max) bounds for search space
        seed (int, optional): Random seed for reproducibility (ignored - uses pre-computed samples)
        num_runs (int): Number of independent SA runs to average over
        num_threads (int, optional): Number of parallel threads (defaults to CPU count)
        starting_points (ndarray, optional): Pre-computed starting points, shape (num_runs, 2)
        random_steps (ndarray, optional): Pre-computed random steps, shape (num_steps, num_runs, 2)
        acceptance_probs (ndarray, optional): Pre-computed acceptance probabilities, shape (num_steps, num_runs)
        
    Returns:
        tuple: (avg_reward, costs, trajectory, median_idx)
            - avg_reward: Average reward across all runs
            - costs: List of final costs for each run
            - trajectory: Trajectory of the median cost run [(x, y, cost), ...]
            - median_idx: Index of the run with median cost
    """
    # Load samples if not provided
    if starting_points is None or random_steps is None or acceptance_probs is None:
        from ..random_sampling import load_random_samples
        starting_points, random_steps, acceptance_probs = load_random_samples()
    
    # Convert bounds to tuple if needed
    if isinstance(bounds, list):
        bounds = tuple(bounds)
    
    return sa_rust.run_sa_parallel(
        float(init_temp),
        float(cooling_rate),
        float(step_size),
        int(num_steps),
        bounds,
        int(seed) if seed is not None else None,
        int(num_runs),
        int(num_threads) if num_threads is not None else None,
        starting_points,
        random_steps,
        acceptance_probs
    )
