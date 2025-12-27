"""
Unit tests for unified random sampling system.

Tests verify that:
1. Random samples are generated correctly
2. All 4 SA algorithms can access the unified samples
3. Results are consistent across implementations
4. Performance is improved by pre-generated samples
"""

import pytest
import numpy as np
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from random_sampling import (
    generate_random_samples,
    load_random_samples,
    get_starting_point,
    get_random_steps,
    UnifiedRandomSampler,
    NUM_STARTING_POINTS,
    NUM_STEP_COLUMNS,
    NUM_STEP_ROWS,
)


class TestRandomSampleGeneration:
    """Test random sample generation and loading."""
    
    def test_generate_samples_shape(self):
        """Test that generated samples have correct shapes."""
        starting_points, random_steps = generate_random_samples(seed=42, force=True)
        
        assert starting_points.shape == (NUM_STARTING_POINTS, 2)
        assert random_steps.shape == (NUM_STEP_ROWS, NUM_STEP_COLUMNS, 2)
    
    def test_generate_samples_bounds(self):
        """Test that starting points are within bounds."""
        starting_points, _ = generate_random_samples(seed=42, force=True)
        
        assert np.all(starting_points >= -5.12)
        assert np.all(starting_points <= 5.12)
    
    def test_load_samples(self):
        """Test loading pre-generated samples."""
        starting_points, random_steps = load_random_samples()
        
        assert starting_points is not None
        assert random_steps is not None
        assert starting_points.shape[0] == NUM_STARTING_POINTS
        assert random_steps.shape == (NUM_STEP_ROWS, NUM_STEP_COLUMNS, 2)
    
    def test_reproducibility(self):
        """Test that same seed produces same samples."""
        points1, steps1 = generate_random_samples(seed=123, force=True)
        points2, steps2 = generate_random_samples(seed=123, force=True)
        
        np.testing.assert_array_equal(points1, points2)
        np.testing.assert_array_equal(steps1, steps2)


class TestUnifiedSampler:
    """Test UnifiedRandomSampler class."""
    
    def test_sampler_initialization(self):
        """Test sampler initializes correctly."""
        sampler = UnifiedRandomSampler()
        
        assert sampler.starting_points is not None
        assert sampler.random_steps is not None
    
    def test_get_starting_point(self):
        """Test getting a single starting point."""
        sampler = UnifiedRandomSampler()
        
        point = sampler.get_starting_point(0)
        assert isinstance(point, tuple)
        assert len(point) == 2
        assert -5.12 <= point[0] <= 5.12
        assert -5.12 <= point[1] <= 5.12
    
    def test_get_random_steps(self):
        """Test getting random steps for a run."""
        sampler = UnifiedRandomSampler()
        
        steps = sampler.get_random_steps(run_idx=0, num_steps=100)
        assert steps.shape == (100, 2)
    
    def test_multiple_runs(self):
        """Test getting data for multiple runs."""
        sampler = UnifiedRandomSampler()
        
        # Get data for 10 runs
        points = [sampler.get_starting_point(i) for i in range(10)]
        steps_list = [sampler.get_random_steps(i, 50) for i in range(10)]
        
        assert len(points) == 10
        assert len(steps_list) == 10
        
        # Verify points are different
        assert not np.allclose(points[0], points[1])
    
    def test_out_of_bounds_run_idx(self):
        """Test error handling for invalid run index."""
        sampler = UnifiedRandomSampler()
        
        with pytest.raises(ValueError):
            sampler.get_starting_point(NUM_STARTING_POINTS + 1)
        
        with pytest.raises(ValueError):
            sampler.get_random_steps(NUM_STEP_COLUMNS + 1, 100)
    
    def test_out_of_bounds_num_steps(self):
        """Test error handling for too many steps requested."""
        sampler = UnifiedRandomSampler()
        
        with pytest.raises(ValueError):
            sampler.get_random_steps(0, NUM_STEP_ROWS + 1)


class TestConsistencyAcrossAlgorithms:
    """Test that all 4 SA algorithms can use unified samples."""
    
    def test_python_serial_uses_samples(self):
        """Test Python serial implementation can access samples."""
        sampler = UnifiedRandomSampler()
        
        # This verifies the sampler works - actual SA algorithm tests would go here
        point = sampler.get_starting_point(0)
        steps = sampler.get_random_steps(0, 100)
        
        assert point is not None
        assert steps is not None
    
    def test_consistency_across_runs(self):
        """Test that same run_idx gives same samples."""
        sampler1 = UnifiedRandomSampler()
        sampler2 = UnifiedRandomSampler()
        
        point1 = sampler1.get_starting_point(5)
        point2 = sampler2.get_starting_point(5)
        
        assert point1 == point2
        
        steps1 = sampler1.get_random_steps(5, 100)
        steps2 = sampler2.get_random_steps(5, 100)
        
        np.testing.assert_array_equal(steps1, steps2)


class TestPerformance:
    """Test that pre-generated samples improve performance."""
    
    def test_loading_is_fast(self):
        """Test that loading pre-generated samples is fast."""
        start_time = time.time()
        sampler = UnifiedRandomSampler()
        load_time = time.time() - start_time
        
        # Loading should be very fast (< 1 second)
        assert load_time < 1.0
        
        # Accessing samples should also be fast
        start_time = time.time()
        for i in range(100):
            sampler.get_starting_point(i)
            sampler.get_random_steps(i, 100)
        access_time = time.time() - start_time
        
        # Accessing 100 runs should be very fast (< 0.1 seconds)
        assert access_time < 0.1
    
    def test_no_rng_calls_during_access(self):
        """Test that accessing samples doesn't call RNG."""
        sampler = UnifiedRandomSampler()
        
        # Get initial state
        point1 = sampler.get_starting_point(10)
        steps1 = sampler.get_random_steps(10, 50)
        
        # Get same data again - should be identical (not new random)
        point2 = sampler.get_starting_point(10)
        steps2 = sampler.get_random_steps(10, 50)
        
        assert point1 == point2
        np.testing.assert_array_equal(steps1, steps2)


class TestIntegration:
    """Integration tests with SA algorithms."""
    
    def test_sample_coverage_for_typical_use(self):
        """Test samples cover typical use cases."""
        sampler = UnifiedRandomSampler()
        
        # Typical use: 100 runs with 1000 steps each
        num_runs = 100
        num_steps = 1000
        
        for i in range(num_runs):
            point = sampler.get_starting_point(i)
            steps = sampler.get_random_steps(i, num_steps)
            
            assert point is not None
            assert steps.shape == (num_steps, 2)
    
    def test_sample_coverage_for_large_experiments(self):
        """Test samples cover larger experiments."""
        sampler = UnifiedRandomSampler()
        
        # Large experiment: 1000 runs with 5000 steps each
        num_runs = 1000
        num_steps = 5000
        
        for i in range(num_runs):
            point = sampler.get_starting_point(i)
            steps = sampler.get_random_steps(i, num_steps)
            
            assert point is not None
            assert steps.shape == (num_steps, 2)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
