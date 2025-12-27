"""
SA Algorithm Configuration Module

Available algorithms:
- 'python_serial': Pure Python serial implementation (baseline)
- 'rust_parallel': Rust parallel implementation (recommended)
"""

ALGORITHM = "rust_parallel"


def get_sa_algorithm():
    """Get the configured SA algorithm module."""
    if ALGORITHM == "python_serial":
        from core.sa_algorithms import python_serial

        return python_serial
    elif ALGORITHM == "rust_parallel":
        from core.sa_algorithms import rust_parallel

        return rust_parallel
    else:
        raise ValueError(
            f"Unknown algorithm: {ALGORITHM}. "
            f"Must be 'python_serial' or 'rust_parallel'"
        )


def get_algorithm_name():
    """Get the name of the currently configured algorithm."""
    return ALGORITHM



def get_run_sa():
    sa = get_sa_algorithm()
    return sa.run_sa

def get_rastrigin_2d():
    sa = get_sa_algorithm()
    return sa.rastrigin_2d

print(f"[SA Config] Using algorithm: {ALGORITHM}")
