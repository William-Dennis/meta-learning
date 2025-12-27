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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNoRNGInSAAlgorithms:
    """Verify that NO RNG is used in any SA algorithm."""
    
    def test_python_serial_no_random_import(self):
        """Test that python_serial.py does NOT import random or use np.random for generation."""
        from sa_algorithms import python_serial
        
        # Check source code for random imports
        import inspect
        source = inspect.getsource(python_serial)
        
        # Remove comments to avoid false positives
        lines = [line.split('#')[0] for line in source.split('\n')]
        code_only = '\n'.join(lines)
        
        # Should NOT have random imports for generation
        assert 'import random' not in code_only, "python_serial should NOT import random module"
        assert 'from random import' not in code_only, "python_serial should NOT import from random"
        assert 'np.random.uniform' not in code_only, "python_serial should NOT use np.random.uniform"
        assert 'np.random.normal' not in code_only, "python_serial should NOT use np.random.normal"
        assert 'np_random.uniform' not in code_only, "python_serial should NOT use RNG uniform"
        assert 'np_random.normal' not in code_only, "python_serial should NOT use RNG normal"
        assert 'np_random.random()' not in code_only, "python_serial should NOT use RNG random()"
        
        # Should USE UnifiedRandomSampler
        assert 'UnifiedRandomSampler' in source, "python_serial MUST use UnifiedRandomSampler"
    
    def test_python_parallel_no_random_import(self):
        """Test that python_parallel.py does NOT import random or use np.random for generation."""
        from sa_algorithms import python_parallel
        
        # Check source code for random imports
        import inspect
        source = inspect.getsource(python_parallel)
        
        # Remove comments to avoid false positives
        lines = [line.split('#')[0] for line in source.split('\n')]
        code_only = '\n'.join(lines)
        
        # Should NOT have random imports for generation
        assert 'import random' not in code_only, "python_parallel should NOT import random module"
        assert 'from random import' not in code_only, "python_parallel should NOT import from random"
        assert 'np.random.uniform' not in code_only, "python_parallel should NOT use np.random.uniform"
        assert 'np.random.normal' not in code_only, "python_parallel should NOT use np.random.normal"
        assert 'np_random.uniform' not in code_only, "python_parallel should NOT use RNG uniform"
        assert 'np_random.normal' not in code_only, "python_parallel should NOT use RNG normal"
        assert 'np_random.random()' not in code_only, "python_parallel should NOT use RNG random()"
        assert 'np.random.randint' not in code_only, "python_parallel should NOT use np.random.randint"
        
        # Should USE UnifiedRandomSampler
        assert 'UnifiedRandomSampler' in source, "python_parallel MUST use UnifiedRandomSampler"
    
    def test_rust_no_random_imports(self):
        """Test that Rust implementation does NOT import rand crate."""
        # Check Cargo.toml
        with open('Cargo.toml', 'r') as f:
            cargo_content = f.read()
        
        # Should NOT have rand dependencies
        assert 'rand =' not in cargo_content, "Cargo.toml should NOT have rand dependency"
        assert 'rand_chacha' not in cargo_content, "Cargo.toml should NOT have rand_chacha dependency"
        assert 'rand_distr' not in cargo_content, "Cargo.toml should NOT have rand_distr dependency"
        
        # Check Rust source code
        with open('src/lib.rs', 'r') as f:
            rust_source = f.read()
        
        # Should NOT import rand
        assert 'use rand::' not in rust_source, "Rust code should NOT use rand crate"
        assert 'use rand_chacha::' not in rust_source, "Rust code should NOT use rand_chacha"
        assert 'use rand_distr::' not in rust_source, "Rust code should NOT use rand_distr"
        assert 'rng.gen' not in rust_source, "Rust code should NOT call rng.gen"
        assert 'rng.sample' not in rust_source, "Rust code should NOT call rng.sample"
        assert 'RngCore' not in rust_source, "Rust code should NOT use RngCore trait"
        assert 'thread_rng' not in rust_source, "Rust code should NOT use thread_rng"
        assert 'ChaCha8Rng' not in rust_source, "Rust code should NOT use ChaCha8Rng"
        
        # Should USE UnifiedRandomSampler
        assert 'UnifiedRandomSampler' in rust_source, "Rust MUST use UnifiedRandomSampler"
    
    def test_deterministic_results_same_run_idx(self):
        """Test that same run_idx produces identical results across calls."""
        from sa_algorithms import python_serial
        
        # Run twice with same parameters
        result1 = python_serial.run_sa(
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=100,
            bounds=(-5.12, 5.12),
            seed=None,  # Seed should be ignored
            num_runs=10
        )
        
        result2 = python_serial.run_sa(
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=100,
            bounds=(-5.12, 5.12),
            seed=None,  # Seed should be ignored
            num_runs=10
        )
        
        # Results should be IDENTICAL (not just similar)
        np.testing.assert_array_equal(result1[1], result2[1], "Costs should be identical")
        np.testing.assert_array_equal(result1[2], result2[2], "Trajectories should be identical")
        assert result1[3] == result2[3], "Median index should be identical"
    
    def test_consistency_across_all_4_implementations(self):
        """Test that all 4 implementations produce identical results."""
        from sa_algorithms import python_serial, python_parallel
        
        params = {
            'init_temp': 10.0,
            'cooling_rate': 0.95,
            'step_size': 1.0,
            'num_steps': 50,
            'bounds': (-5.12, 5.12),
            'seed': None,
            'num_runs': 10
        }
        
        # Run Python implementations
        result_py_serial = python_serial.run_sa(**params)
        result_py_parallel = python_parallel.run_sa(**params)
        
        # Results should be IDENTICAL across implementations
        np.testing.assert_array_almost_equal(
            result_py_serial[1], result_py_parallel[1], decimal=10,
            err_msg="Python serial and parallel should produce identical costs"
        )
        
        # Trajectories should match
        assert len(result_py_serial[2]) == len(result_py_parallel[2]), \
            "Trajectory lengths should match"
    
    def test_no_variation_with_different_seeds(self):
        """Test that different seeds produce SAME results (seed should be ignored)."""
        from sa_algorithms import python_serial
        
        # Run with different seeds - should produce SAME results
        result1 = python_serial.run_sa(
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=50,
            bounds=(-5.12, 5.12),
            seed=42,
            num_runs=5
        )
        
        result2 = python_serial.run_sa(
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=50,
            bounds=(-5.12, 5.12),
            seed=123,  # Different seed
            num_runs=5
        )
        
        # Results should be IDENTICAL (seed is ignored)
        np.testing.assert_array_equal(result1[1], result2[1], 
            "Different seeds should produce IDENTICAL results (seed ignored)")
    
    def test_uses_unified_sampler_deterministically(self):
        """Test that UnifiedRandomSampler is used and produces deterministic results."""
        from sa_algorithms import python_serial
        from random_sampling import UnifiedRandomSampler
        
        # Get expected starting points directly from sampler
        sampler = UnifiedRandomSampler()
        expected_starts = [sampler.get_starting_point(i) for i in range(5)]
        
        # Run SA and check it uses exact starting points
        _, costs, trajectory, median_idx = python_serial.run_sa(
            init_temp=10.0,
            cooling_rate=0.95,
            step_size=1.0,
            num_steps=10,
            bounds=(-5.12, 5.12),
            seed=None,
            num_runs=5
        )
        
        # The returned trajectory is from the median run
        # Check that one of the expected starting points matches
        actual_start = (trajectory[0][0], trajectory[0][1])
        expected_median_start = expected_starts[median_idx]
        
        np.testing.assert_array_almost_equal(expected_median_start, actual_start, decimal=10,
            err_msg="SA should use exact starting point from UnifiedRandomSampler for median run")
    
    def test_no_step_penalty_in_sa_algorithms(self):
        """Test that step penalty is NOT calculated in SA algorithms (moved to env)."""
        from sa_algorithms import python_serial
        
        # Check source code
        import inspect
        source = inspect.getsource(python_serial)
        
        # Should NOT have step penalty calculation in SA algorithm
        # Penalty should be in environment
        assert '- (num_steps / 1000)' not in source, "Step penalty should NOT be in SA algorithm"
        assert '(num_steps as f64 / 1000.0)' not in source, "Step penalty should NOT be in SA algorithm"


class TestUnifiedSamplerIntegration:
    """Test that UnifiedRandomSampler integrates correctly with all algorithms."""
    
    def test_python_serial_uses_sampler(self):
        """Test python_serial correctly uses UnifiedRandomSampler."""
        from sa_algorithms import python_serial
        
        result = python_serial.run_sa(
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
    
    def test_python_parallel_uses_sampler(self):
        """Test python_parallel correctly uses UnifiedRandomSampler."""
        from sa_algorithms import python_parallel
        
        result = python_parallel.run_sa(
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
        assert len(result[2]) == 101  # 101 trajectory points


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
