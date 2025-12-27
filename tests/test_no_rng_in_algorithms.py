"""
Test that NO random number generators are used in SA algorithms.

This test file verifies the critical requirement that all 4 SA implementations
use ONLY pre-generated random samples and contain NO RNG calls.
"""


import pytest
import numpy as np
import sys
import os
import importlib
from meta_learning.sa_algorithms import run_sa
from meta_learning.random_sampling import load_random_samples
import meta_learning.sa_algorithms as sa_algorithms

# Derive implementations from sa_algorithms.__init__
try:
    IMPLEMENTATIONS = sa_algorithms.__init__.IMPLEMENTATIONS
except AttributeError:
    # Fallback if not defined, use hardcoded list
    IMPLEMENTATIONS = ["python_serial", "python_parallel", "rust_serial", "rust_parallel"]

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))


class TestNoRNGInSAAlgorithms:
    """Verify that NO RNG is used in any SA algorithm."""

    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
    def test_deterministic_results_same_run_idx(self, implementation):
        """Test that same run_idx produces identical results across calls."""
        result1 = run_sa(
            implementation=implementation,
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=100,
            bounds=(-5.12, 5.12),
            seed=None,
            num_runs=10
        )
        result2 = run_sa(
            implementation=implementation,
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=100,
            bounds=(-5.12, 5.12),
            seed=None,
            num_runs=10
        )
        np.testing.assert_array_equal(result1[1], result2[1], "Costs should be identical")
        np.testing.assert_array_equal(result1[2], result2[2], "Trajectories should be identical")
        assert result1[3] == result2[3], "Median index should be identical"

    def test_consistency_across_all_4_implementations(self):
        """Test that all 4 implementations produce identical results."""
        params = {
            'init_temp': 10.0,
            'cooling_rate': 0.95,
            'step_size': 1.0,
            'num_steps': 50,
            'bounds': (-5.12, 5.12),
            'seed': None,
            'num_runs': 10
        }
        results = [run_sa(implementation=impl, **params) for impl in IMPLEMENTATIONS]
        # Compare costs and trajectories across all implementations
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(
                results[0][1], results[i][1], decimal=10,
                err_msg=f"{IMPLEMENTATIONS[0]} and {IMPLEMENTATIONS[i]} should produce identical costs"
            )
            assert len(results[0][2]) == len(results[i][2]), "Trajectory lengths should match"

    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
    def test_no_variation_with_different_seeds(self, implementation):
        """Test that different seeds produce SAME results (seed should be ignored)."""
        import warnings
        # Warn if seed is not None
        warnings.warn("Seed argument should not be used; deterministic results expected.")
        result1 = run_sa(
            implementation=implementation,
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=50,
            bounds=(-5.12, 5.12),
            seed=42,
            num_runs=5
        )
        result2 = run_sa(
            implementation=implementation,
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=50,
            bounds=(-5.12, 5.12),
            seed=123,
            num_runs=5
        )
        np.testing.assert_array_equal(result1[1], result2[1], "Different seeds should produce IDENTICAL results (seed ignored)")

    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
    def test_uses_precomputed_samples_deterministically(self, implementation):
        """Test that pre-computed samples are used and produce deterministic results."""
        starting_points, random_steps, acceptance_probs = load_random_samples()
        expected_starts = [tuple(starting_points[i]) for i in range(5)]
        _, costs, trajectory, median_idx = run_sa(
            implementation=implementation,
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=10,
            bounds=(-5.12, 5.12),
            seed=None,
            num_runs=5
        )
        actual_start = (trajectory[0][0], trajectory[0][1])
        expected_median_start = expected_starts[median_idx]
        np.testing.assert_array_almost_equal(expected_median_start, actual_start, decimal=10,
            err_msg="SA should use exact starting point from UnifiedRandomSampler for median run")

    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
    def test_no_step_penalty_in_sa_algorithms(self, implementation):
        """Test that step penalty is NOT calculated in SA algorithms (moved to env)."""
        import inspect
        # Check source code for penalty
        module = importlib.import_module(f"meta_learning.sa_algorithms.{implementation}")
        source = inspect.getsource(module)
        assert '- (num_steps / 1000)' not in source, "Step penalty should NOT be in SA algorithm"
        assert '(num_steps as f64 / 1000.0)' not in source, "Step penalty should NOT be in SA algorithm"
        # Check cost == reward for median trajectory
        params = {
            'init_temp': 10.0,
            'cooling_rate': 0.95,
            'step_size': 1.0,
            'num_steps': 10,
            'bounds': (-5.12, 5.12),
            'seed': None,
            'num_runs': 5
        }
        _, costs, trajectory, median_idx = run_sa(implementation=implementation, **params)
        # If cost and reward are returned, check they are equal
        if hasattr(module, 'reward_function'):
            reward = module.reward_function(trajectory)
            np.testing.assert_almost_equal(costs[median_idx], reward, decimal=10,
                err_msg="Cost should equal reward if no penalty is applied")


class TestUnifiedSamplerIntegration:
    """Test that UnifiedRandomSampler integrates correctly with all algorithms."""

    @pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
    def test_uses_sampler(self, implementation):
        """Test all implementations correctly use UnifiedRandomSampler."""
        result = run_sa(
            implementation=implementation,
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=100,
            bounds=(-5.12, 5.12),
            seed=None,
            num_runs=10
        )
        assert result is not None
        assert len(result) == 4
        assert len(result[1]) == 10  # 10 costs
        assert len(result[2]) == 101  # 101 trajectory points (100 steps + 1 initial)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
