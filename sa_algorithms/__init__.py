"""
SA Algorithms Package

This package contains 4 implementations of the Simulated Annealing algorithm:
1. python_serial: Pure Python serial implementation
2. python_parallel: Python with multiprocessing parallelization
3. rust_serial: Rust serial implementation via PyO3
4. rust_parallel: Rust parallel implementation via PyO3 with Rust threads

All implementations provide the same interface:
    run_sa(init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10)
"""

from . import python_serial
from . import python_parallel

# Import Rust implementations if available
try:
    from . import rust_serial
    from . import rust_parallel
    __all__ = ['python_serial', 'python_parallel', 'rust_serial', 'rust_parallel']
except ImportError:
    # Rust not available
    __all__ = ['python_serial', 'python_parallel']
