"""
Unified Random Sampling for SA Algorithms

This module generates and provides unified random samples for all SA algorithm
implementations to ensure they produce identical results when given the same seed.

The randomness in SA comes from two sources:
1. Starting locations (x, y coordinates)
2. Random normal variables for step generation during optimization

By pre-generating these samples, we:
- Eliminate discrepancies between Python and Rust implementations
- Speed up runtime by avoiding RNG calls during optimization
- Enable reproducible comparisons across all 4 implementations
"""

import numpy as np
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
STARTING_POINTS_FILE = os.path.join(DATA_DIR, 'starting_points.npy')
RANDOM_STEPS_FILE = os.path.join(DATA_DIR, 'random_steps.npy')

# Pre-generate sufficient samples for typical use cases
NUM_STARTING_POINTS = 10000  # Support up to 10k SA runs
NUM_STEP_COLUMNS = 10000     # Support up to 10k SA runs
NUM_STEP_ROWS = 10000        # Support up to 10k steps per run

# Default bounds for Rastrigin function
DEFAULT_BOUNDS = (-5.12, 5.12)


def generate_random_samples(seed=42, force=False):
    """
    Generate unified random samples for all SA algorithms.
    
    Args:
        seed: Random seed for reproducibility
        force: If True, regenerate even if files exist
        
    Returns:
        tuple: (starting_points, random_steps)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if files already exist
    if not force and os.path.exists(STARTING_POINTS_FILE) and os.path.exists(RANDOM_STEPS_FILE):
        print(f"[Random Sampling] Loading existing random samples from {DATA_DIR}")
        starting_points = np.load(STARTING_POINTS_FILE)
        random_steps = np.load(RANDOM_STEPS_FILE)
        return starting_points, random_steps
    
    print(f"[Random Sampling] Generating unified random samples (seed={seed})...")
    rng = np.random.default_rng(seed)
    
    # Generate starting points (x, y) within bounds
    # Shape: (NUM_STARTING_POINTS, 2)
    starting_points = rng.uniform(
        DEFAULT_BOUNDS[0], 
        DEFAULT_BOUNDS[1], 
        size=(NUM_STARTING_POINTS, 2)
    )
    
    # Generate random normal steps for optimization
    # Shape: (NUM_STEP_ROWS, NUM_STEP_COLUMNS, 2) for dx and dy
    random_steps = rng.standard_normal(size=(NUM_STEP_ROWS, NUM_STEP_COLUMNS, 2))
    
    # Save to files
    np.save(STARTING_POINTS_FILE, starting_points)
    np.save(RANDOM_STEPS_FILE, random_steps)
    
    print(f"[Random Sampling] Saved starting points: {starting_points.shape}")
    print(f"[Random Sampling] Saved random steps: {random_steps.shape}")
    print(f"[Random Sampling] Files saved to {DATA_DIR}")
    
    return starting_points, random_steps


def load_random_samples():
    """
    Load pre-generated random samples.
    
    Returns:
        tuple: (starting_points, random_steps)
    """
    if not os.path.exists(STARTING_POINTS_FILE) or not os.path.exists(RANDOM_STEPS_FILE):
        print("[Random Sampling] Random sample files not found, generating...")
        return generate_random_samples()
    
    starting_points = np.load(STARTING_POINTS_FILE)
    random_steps = np.load(RANDOM_STEPS_FILE)
    return starting_points, random_steps


def get_starting_point(run_idx):
    """
    Get a specific starting point for an SA run.
    
    Args:
        run_idx: Index of the run (0-based)
        
    Returns:
        tuple: (x, y) starting coordinates
    """
    starting_points, _ = load_random_samples()
    if run_idx >= len(starting_points):
        raise ValueError(f"Run index {run_idx} exceeds available starting points ({len(starting_points)})")
    return starting_points[run_idx]


def get_random_steps(run_idx, num_steps):
    """
    Get random steps for a specific SA run.
    
    Args:
        run_idx: Index of the run (0-based)
        num_steps: Number of steps needed
        
    Returns:
        ndarray: Shape (num_steps, 2) with dx, dy for each step
    """
    _, random_steps = load_random_samples()
    if run_idx >= random_steps.shape[1]:
        raise ValueError(f"Run index {run_idx} exceeds available columns ({random_steps.shape[1]})")
    if num_steps > random_steps.shape[0]:
        raise ValueError(f"Requested {num_steps} steps exceeds available rows ({random_steps.shape[0]})")
    
    return random_steps[:num_steps, run_idx, :]


class UnifiedRandomSampler:
    """
    Provides unified random samples for SA algorithms.
    
    Usage:
        sampler = UnifiedRandomSampler()
        
        # For multiple runs
        for run_idx in range(num_runs):
            x, y = sampler.get_starting_point(run_idx)
            steps = sampler.get_random_steps(run_idx, num_steps)
            # Use (x, y) as starting point
            # Use steps[:, 0] for dx and steps[:, 1] for dy
    """
    
    def __init__(self, ensure_loaded=True):
        """
        Initialize the sampler.
        
        Args:
            ensure_loaded: If True, load samples immediately
        """
        self.starting_points = None
        self.random_steps = None
        
        if ensure_loaded:
            self.starting_points, self.random_steps = load_random_samples()
    
    def get_starting_point(self, run_idx):
        """Get starting point for a specific run."""
        if self.starting_points is None:
            self.starting_points, self.random_steps = load_random_samples()
        
        if run_idx >= len(self.starting_points):
            raise ValueError(f"Run index {run_idx} exceeds available starting points ({len(self.starting_points)})")
        
        return tuple(self.starting_points[run_idx])
    
    def get_random_steps(self, run_idx, num_steps):
        """Get random steps for a specific run."""
        if self.random_steps is None:
            self.starting_points, self.random_steps = load_random_samples()
        
        if run_idx >= self.random_steps.shape[1]:
            raise ValueError(f"Run index {run_idx} exceeds available columns ({self.random_steps.shape[1]})")
        if num_steps > self.random_steps.shape[0]:
            raise ValueError(f"Requested {num_steps} steps exceeds available rows ({self.random_steps.shape[0]})")
        
        return self.random_steps[:num_steps, run_idx, :]


# Generate random samples when module is imported (lazy loading)
# This ensures samples are available but doesn't slow down imports
if __name__ == "__main__":
    # Command-line interface for generating samples
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate unified random samples for SA algorithms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if files exist")
    args = parser.parse_args()
    
    starting_points, random_steps = generate_random_samples(seed=args.seed, force=args.force)
    print(f"\n✓ Generated {len(starting_points)} starting points")
    print(f"✓ Generated {random_steps.shape[1]} columns × {random_steps.shape[0]} rows of random steps")
    print(f"✓ Each column supports up to {random_steps.shape[0]} optimization steps")
