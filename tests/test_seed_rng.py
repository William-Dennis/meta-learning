import sys
import os
# Ensure project root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.sa_algorithms import python_serial, rust_parallel
import numpy as np

def test_wrapper():
    """Decorator with try-except for test functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
                print(f"{func.__name__}: PASSED")
            except AssertionError as e:
                print(f"AssertionError in {func.__name__}: {e}")
            except Exception as e:
                print(f"Exception in {func.__name__}: {e}")
        return wrapper
    return decorator

@test_wrapper()
def test_that_rust_seed_rng_is_deterministic():
    """Test that the RNG seed produces deterministic results."""
    bounds = [-5.12, 5.12]
    init_temp = 10.0
    cooling_rate = 0.95
    step_size = 1.0
    num_steps = 100
    seed = 12345
    num_runs = 5

    # Run SA twice with the same seed
    avg_reward1, costs1, trajectory1, median_idx1 = rust_parallel.run_sa(
        init_temp, cooling_rate, step_size, num_steps, bounds, seed, num_runs
    )
    avg_reward2, costs2, trajectory2, median_idx2 = rust_parallel.run_sa(
        init_temp, cooling_rate, step_size, num_steps, bounds, seed, num_runs
    )

    # Check that results are identical, sorting is due to possible non-deterministic ordering
    assert avg_reward1 == avg_reward2, "Average rewards differ between runs with same seed"
    assert np.array_equal(np.sort(costs1), np.sort(costs2)), "Costs differ between runs with same seed"
    assert np.array_equal(np.sort(trajectory1), np.sort(trajectory2)), "Trajectories differ between runs with same seed"
    assert costs1[median_idx1] == costs2[median_idx2], "Median values differ between runs with same seed"

@test_wrapper()
def test_that_python_seed_rng_is_deterministic():
    """Test that the RNG seed produces deterministic results."""
    bounds = [-5.12, 5.12]
    init_temp = 10.0
    cooling_rate = 0.95
    step_size = 1.0
    num_steps = 100
    seed = 12345
    num_runs = 5

    # Run SA twice with the same seed
    avg_reward1, costs1, trajectory1, median_idx1 = python_serial.run_sa(
        init_temp, cooling_rate, step_size, num_steps, bounds, seed, num_runs
    )
    avg_reward2, costs2, trajectory2, median_idx2 = python_serial.run_sa(
        init_temp, cooling_rate, step_size, num_steps, bounds, seed, num_runs
    )

    # Check that results are identical
    assert avg_reward1 == avg_reward2, "Average rewards differ between runs with same seed"
    assert np.array_equal(np.sort(costs1), np.sort(costs2)), "Costs differ between runs with same seed"
    assert np.array_equal(np.sort(trajectory1), np.sort(trajectory2)), "Trajectories differ between runs with same seed"
    assert costs1[median_idx1] == costs2[median_idx2], "Median values differ between runs with same seed"

if __name__ == "__main__":
    test_that_rust_seed_rng_is_deterministic()
    test_that_python_seed_rng_is_deterministic()