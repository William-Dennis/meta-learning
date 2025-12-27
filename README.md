# Meta-Learning using Simulated Annealing

This repository demonstrates using **Proximal Policy Optimization (PPO)** to automatically tune **Simulated Annealing (SA)** hyperparameters for minimizing the 2D Rastrigin function.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager

### Installation

```bash
# Install UV
pip install uv

# Install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install numpy matplotlib torch maturin

# Build Rust extension (optional, for better performance)
maturin develop --release
```

### Running Experiments

```bash
# Run PPO training to tune SA hyperparameters
python run_experiment.py

# Run grid search over SA hyperparameters
python run_grid_search.py
```

## 📁 Repository Structure

```
meta-learning/
├── run_experiment.py      # PPO training runner
├── run_grid_search.py     # Grid search runner
├── core/                  # Core implementation modules
│   ├── sa_algorithms/     # SA algorithm implementations
│   │   ├── python_serial.py   # Python serial SA
│   │   └── rust_parallel.py   # Rust parallel SA (fast)
│   ├── sa_config.py       # SA algorithm configuration
│   ├── tuning_env.py      # PPO training environment
│   └── ppo_agent.py       # PPO agent implementation
├── outputs/               # Generated plots and results
├── src/                   # Rust source code
│   └── lib.rs            # Rust SA implementation
└── README.md             # This file
```

## 🎯 Algorithm Selection

The repository includes two SA implementations:
- **python_serial**: Pure Python implementation (baseline)
- **rust_parallel**: Rust parallel implementation (recommended for speed)

To switch algorithms, edit `core/sa_config.py`:

```python
ALGORITHM = 'rust_parallel'  # or 'python_serial'
```

## 🔑 Key Features

- ✅ Simple, clean codebase
- ✅ PPO-based hyperparameter tuning
- ✅ Rust acceleration for performance
- ✅ Automatic output organization
- ✅ Seed-based reproducibility
- ✅ All functions ≤ 30 lines

## 📊 Outputs

All generated outputs are saved in the `outputs/` directory:
- Training curves
- Parameter evolution plots
- Trajectory visualizations
- Performance metrics

## 🛠️ Development

**Always use UV for dependency management:**

```bash
# Add a new dependency
uv pip install <package>

# Never use pip directly except to install UV itself
```

## 📝 Notes

- All randomness is controlled via the `seed` parameter
- Functions are kept simple (≤ 30 lines)
- Code follows clean, minimal design principles

## 📄 License

See [LICENSE](LICENSE) file.
