# Meta-Learning using Simulated Annealing

This repository demonstrates using **Proximal Policy Optimization (PPO)** to automatically tune **Simulated Annealing (SA)** hyperparameters for minimizing the 2D Rastrigin function.

## 🚀 Rust Performance Boost

The SA algorithm has been reimplemented in **Rust** for dramatic performance improvements:

- **Python Serial:** Baseline performance
- **Python Parallel:** 1.7x speedup (multiprocessing)
- **Rust Serial:** 91x speedup ⚡
- **Rust Parallel:** **171x speedup** 🚀

Example: 10,000 SA steps
- Python: 9.6 seconds
- **Rust Parallel: 0.04 seconds** (220x faster!)

See [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md) for complete performance analysis.

## 🎲 Unified Random Sampling

All 4 implementations now use **unified pre-generated random samples** to ensure:
- ✅ Identical results across Python and Rust implementations
- ✅ Faster runtime (no RNG calls during optimization)
- ✅ Reproducible experiments

The random samples are pre-generated and stored in `data/`:
- `starting_points.npy`: 10,000 starting locations
- `random_steps.npy`: 10,000 × 10,000 random normal samples

## 🔧 Switching SA Algorithm Implementations

**All code in this repository uses a single configuration file to select the SA algorithm.**

To switch implementations, simply edit `sa_config.py`:

```python
# In sa_config.py, change this line:
ALGORITHM = 'rust_parallel'  # Options: 'python_serial', 'python_parallel', 'rust_serial', 'rust_parallel'
```

That's it! All scripts (`run_experiment.py`, `run_grid_search.py`, etc.) will automatically use the selected algorithm.

## Quick Start

### Option 1: Use Python Implementation (Easy)
```bash
pip install numpy matplotlib torch pytest

# Edit sa_config.py and set: ALGORITHM = 'python_serial'
python run_experiment.py
```

### Option 2: Use Rust Implementation (Fast)
```bash
# Install UV and dependencies
pip install uv
uv venv
source .venv/bin/activate
uv pip install numpy matplotlib torch maturin pytest

# Build Rust extension
maturin develop --release

# Edit sa_config.py and set: ALGORITHM = 'rust_parallel'
python run_experiment.py

# Run benchmarks
python comprehensive_benchmark.py
```

See [docs/RUST_SETUP.md](docs/RUST_SETUP.md) for detailed setup instructions.

## Repository Structure

```
meta-learning/
├── sa_algorithms/           # 4 clean SA implementations
│   ├── python_serial.py    # Pure Python (baseline)
│   ├── python_parallel.py  # Python + multiprocessing
│   ├── rust_serial.py      # Rust serial (~91x faster)
│   └── rust_parallel.py    # Rust parallel (~171x faster)
├── sa_config.py            # Configuration: Switch algorithms here!
├── random_sampling.py      # Unified random sampling system
├── data/                   # Pre-generated random samples
│   ├── starting_points.npy
│   └── random_steps.npy
├── tests/                  # Pytest unit tests
│   └── test_random_sampling.py
├── docs/                   # Documentation and images
│   ├── ALGORITHM_SWITCHING.md
│   ├── RUST_SETUP.md
│   ├── RESULTS_SUMMARY.md
│   └── images/
├── src/lib.rs              # Rust implementation
├── run_experiment.py       # PPO training (uses sa_config)
├── run_grid_search.py      # Grid search (uses sa_config)
└── comprehensive_benchmark.py  # Performance comparison
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_random_sampling.py -v
```

## Performance Results

| Steps | Python Serial | Python Parallel | Rust Serial | Rust Parallel |
|-------|--------------|-----------------|-------------|---------------|
| 100   | 0.0999s      | 0.0680s         | 0.0011s     | 0.0006s       |
| 1,000 | 0.9784s      | 0.4955s         | 0.0106s     | 0.0046s       |
| 5,000 | 4.7767s      | 2.3546s         | 0.0482s     | 0.0217s       |
| 10,000| 9.5679s      | 4.7192s         | 0.0956s     | 0.0436s       |

![Performance Comparison](docs/images/performance_comparison.png)

## Key Features

- ✅ **Easy algorithm switching** via single config file
- ✅ **Unified random sampling** for consistent results
- ✅ 4 interchangeable SA implementations (Python serial/parallel, Rust serial/parallel)
- ✅ PPO-based hyperparameter tuning
- ✅ 91-171x performance improvement with Rust
- ✅ Comprehensive benchmarking suite
- ✅ **Full test coverage** with pytest
- ✅ UV package manager integration
- ✅ Complete documentation

## Documentation

- [docs/RUST_SETUP.md](docs/RUST_SETUP.md) - Rust setup guide with UV
- [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md) - Performance analysis
- [docs/ALGORITHM_SWITCHING.md](docs/ALGORITHM_SWITCHING.md) - Algorithm switching guide
- [docs/results_report.md](docs/results_report.md) - Original PPO tuning results
- [sa_algorithms/README.md](sa_algorithms/README.md) - Algorithm usage guide

## License

See [LICENSE](LICENSE) file.

