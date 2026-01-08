"""
Tests that rust functions and python functions return the same values
"""

import numpy as np
from helper import set_wd

set_wd()

from core.math import rastrigin_2d, quadratic_2d

# Build the Rust module first if not already built
try:
    import sa_rust

    rust_available = True
except ImportError:
    print(
        "Warning: sa_rust module not available. Run 'uv run maturin develop --release' first."
    )
    rust_available = False


def test_rastrigin_equivalence():
    """Test that Python and Rust rastrigin functions return same values"""
    if not rust_available:
        print("Skipping rastrigin test - Rust module not available")
        return

    # Test multiple points
    test_points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (2.5, -1.5),
        (-2.0, 3.0),
        (0.5, -0.5),
    ]

    for x, y in test_points:
        py_result = rastrigin_2d(x, y)
        rust_result = sa_rust.rastrigin_2d_py(x, y)

        # Check if values are very close (allow for floating point differences)
        assert np.isclose(py_result, rust_result, rtol=1e-10), (
            f"Rastrigin mismatch at ({x}, {y}): Python={py_result}, Rust={rust_result}"
        )

    print("✓ Rastrigin function equivalence test passed")


def test_quadratic_equivalence():
    """Test that Python and Rust quadratic functions return same values"""
    if not rust_available:
        print("Skipping quadratic test - Rust module not available")
        return

    # Test multiple points
    test_points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (2.5, -1.5),
        (-2.0, 3.0),
        (0.5, -0.5),
        (10.0, -10.0),
    ]

    for x, y in test_points:
        py_result = quadratic_2d(x, y)
        rust_result = sa_rust.quadratic_2d_py(x, y)

        # Check if values are very close (allow for floating point differences)
        assert np.isclose(py_result, rust_result, rtol=1e-10), (
            f"Quadratic mismatch at ({x}, {y}): Python={py_result}, Rust={rust_result}"
        )

    print("✓ Quadratic function equivalence test passed")


if __name__ == "__main__":
    test_rastrigin_equivalence()
    test_quadratic_equivalence()
    print("\nAll function equivalence tests passed!")
