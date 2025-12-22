# Rust Implementation Setup Guide

This guide explains how to set up and use the Rust-accelerated Simulated Annealing implementation.

## Prerequisites

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:
```bash
rustc --version
cargo --version
```

### 2. Install UV Package Manager

UV is a fast Python package manager written in Rust.

```bash
pip install uv
```

Or via curl:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Project Setup

### 1. Create Virtual Environment

```bash
cd /path/to/meta-learning
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
uv pip install numpy matplotlib torch maturin tabulate
```

### 3. Build the Rust Extension

The Rust code is compiled into a Python extension module using [Maturin](https://github.com/PyO3/maturin).

```bash
# Development build (with debug symbols)
maturin develop

# Release build (optimized, ~100x faster)
maturin develop --release
```

## Project Structure

```
meta-learning/
├── Cargo.toml          # Rust package configuration
├── pyproject.toml      # Python package configuration
├── src/
│   └── lib.rs         # Rust implementation of SA algorithm
├── sa_algorithm.py     # Original Python implementation
├── benchmark_rust.py   # Performance comparison script
└── .venv/             # Virtual environment (excluded from git)
```

## Usage

### Using the Rust Implementation

```python
import sa_rust

# Run SA with Rust implementation
avg_reward, costs, trajectory, median_idx = sa_rust.run_sa(
    init_temp=10.0,
    cooling_rate=0.95,
    step_size=1.0,
    num_steps=1000,
    bounds=(-5.12, 5.12),  # Must be a tuple
    seed=42,
    num_runs=10
)

print(f"Average reward: {avg_reward}")
print(f"Mean cost: {np.mean(costs)}")
```

### Running the Benchmark

```bash
source .venv/bin/activate
python benchmark_rust.py
```

This will compare Python vs Rust performance across different step counts.

## Performance Results

The Rust implementation provides significant speedup:

| Steps | Python (s) | Rust (s) | Speedup |
|-------|-----------|----------|---------|
| 10    | 0.0017    | 0.00002  | 107x    |
| 100   | 0.0119    | 0.0001   | 110x    |
| 1000  | 0.0995    | 0.0010   | 95x     |
| 5000  | 0.4920    | 0.0048   | 104x    |

**Average speedup: ~100x faster!**

## Development Workflow

### 1. Modify Rust Code

Edit `src/lib.rs` to make changes to the SA algorithm.

### 2. Rebuild

```bash
maturin develop --release
```

### 3. Test

```bash
python benchmark_rust.py
```

### 4. Run Full Experiments

```bash
# Use Rust implementation in tuning environment
python run_experiment.py
```

## Troubleshooting

### "No module named 'sa_rust'"

Make sure you've built the extension:
```bash
maturin develop --release
```

### Build Errors

If you get compilation errors, ensure:
1. Rust is installed: `rustc --version`
2. Python dev headers are available: `python3-dev` (Linux) or install from python.org (Windows/Mac)
3. Clean and rebuild:
   ```bash
   cargo clean
   maturin develop --release
   ```

### UV Not Found

```bash
pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
```

## PyO3 and Maturin

This project uses:
- **PyO3**: Rust bindings for Python, allowing seamless Rust-Python interop
- **Maturin**: Build tool that compiles Rust code into Python extension modules

Benefits:
- ✅ Native performance (100x speedup)
- ✅ Memory safety (Rust's guarantees)
- ✅ Easy Python integration
- ✅ No runtime overhead

## Further Reading

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Guide](https://www.maturin.rs/)
- [UV Documentation](https://github.com/astral-sh/uv)
- [Rust-Python Interop Exercises](https://rust-exercises.com/rust-python-interop/)
