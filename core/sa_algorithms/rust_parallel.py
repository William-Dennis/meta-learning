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


import numpy as np


def _rastrigin_2d_scalar(x, y):
    return sa_rust.rastrigin_2d_py(float(x), float(y))


rastrigin_2d = np.vectorize(_rastrigin_2d_scalar)


def run_sa(
    init_temp,
    cooling_rate,
    step_size,
    num_steps,
    bounds,
    seed=None,
    num_runs=10,
    num_threads=None,
    function=None,
):
    """Run Simulated Annealing algorithm (Rust parallel version).
    
    Note: The Rust implementation currently only supports rastrigin_2d.
    If a different function is provided, this will raise an error.
    """
    if function is not None:
        # Check if it's the rastrigin function by name
        function_name = getattr(function, '__name__', None)
        if function_name != 'rastrigin_2d':
            raise NotImplementedError(
                f"Rust parallel implementation currently only supports rastrigin_2d, "
                f"but received '{function_name}'. "
                f"To use a different objective function, set ALGORITHM='python_serial' in core/sa_config.py"
            )
    
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
    )
