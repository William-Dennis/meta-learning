import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sqlite3
import os
from sa_config import run_sa
from param_scaling import (
    get_nn_grid, get_param_grid, nn_output_to_param, 
    PARAM_RANGES, NN_MIN, NN_MAX
)


# SQLite database for caching results (faster than CSV for incremental writes)
CACHE_DB = "grid_search_cache.db"

def init_cache_db():
    """Initialize SQLite database with proper schema."""
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    # Create table if not exists with index on lookup columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nn_init_temp REAL NOT NULL,
            nn_cooling_rate REAL NOT NULL,
            nn_step_size REAL NOT NULL,
            nn_num_steps REAL NOT NULL,
            init_temp REAL NOT NULL,
            cooling_rate REAL NOT NULL,
            step_size REAL NOT NULL,
            num_steps INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            num_sa_runs INTEGER NOT NULL,
            mean_cost REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for fast lookups
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_nn_params ON results (
            nn_init_temp, nn_cooling_rate, nn_step_size, nn_num_steps, seed, num_sa_runs
        )
    ''')
    
    conn.commit()
    return conn

conn = init_cache_db()

# Get all costs from DB to compute median
cursor = conn.cursor()
cursor.execute('SELECT mean_cost FROM results')
costs_all = [row[0] for row in cursor.fetchall()]

if len(costs_all) < 100:
    upper_bound = 20
else:
    # median_cost = 2 * np.median(costs_all)
    upper_bound = np.percentile(costs_all, 95)

# Fixed color scale bounds for Rastrigin 2D function
# Global minimum: 0, Maximum in bounds [-5.12, 5.12]: ~80
COST_VMIN = 0
COST_VMAX = upper_bound  # Upper bound for Rastrigin function


# SA run parameters
SA_SEED = 42
SA_NUM_RUNS = 100  # Number of SA runs to average for each configuration

# SA bounds for Rastrigin function
SA_BOUNDS = [-5.12, 5.12]



def get_cached_result(conn, nn_params, seed, num_sa_runs):
    """
    Check if a result is already cached based on NN-space parameters.
    Returns mean_cost if found, None otherwise.
    """
    cursor = conn.cursor()
    cursor.execute('''
        SELECT mean_cost FROM results 
        WHERE nn_init_temp = ?
          AND nn_cooling_rate = ?
          AND nn_step_size = ?
          AND nn_num_steps = ?
          AND seed = ?
          AND num_sa_runs = ?
        LIMIT 1
    ''', (nn_params['init_temp'], nn_params['cooling_rate'], 
          nn_params['step_size'], nn_params['num_steps'],
          seed, num_sa_runs))
    
    result = cursor.fetchone()
    return result[0] if result else None

# def get_cached_result(conn, nn_params, seed, num_sa_runs):
#     """
#     Check if a result is already cached based on NN-space parameters.
#     Returns mean_cost if found, None otherwise.
#     """
#     cursor = conn.cursor()
#     cursor.execute('''
#         SELECT mean_cost FROM results 
#         WHERE abs(nn_init_temp - ?) < 1e-9
#           AND abs(nn_cooling_rate - ?) < 1e-9
#           AND abs(nn_step_size - ?) < 1e-9
#           AND abs(nn_num_steps - ?) < 1e-9
#           AND seed = ?
#           AND num_sa_runs = ?
#         LIMIT 1
#     ''', (nn_params['init_temp'], nn_params['cooling_rate'], 
#           nn_params['step_size'], nn_params['num_steps'],
#           seed, num_sa_runs))
    
#     result = cursor.fetchone()
#     return result[0] if result else None


def save_result_immediately(conn, nn_params, actual_params, seed, num_sa_runs, mean_cost):
    """Save a single result immediately to the database."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO results (
            nn_init_temp, nn_cooling_rate, nn_step_size, nn_num_steps,
            init_temp, cooling_rate, step_size, num_steps,
            seed, num_sa_runs, mean_cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        nn_params['init_temp'], nn_params['cooling_rate'],
        nn_params['step_size'], nn_params['num_steps'],
        actual_params['init_temp'], actual_params['cooling_rate'],
        actual_params['step_size'], actual_params['num_steps'],
        seed, num_sa_runs, mean_cost
    ))
    conn.commit()  # Commit immediately for durability


def get_cache_stats(conn):
    """Get statistics about the cache."""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM results')
    count = cursor.fetchone()[0]
    return count


def run_grid_search(nn_step=1.0, verbose=True):
    """
    Run grid search over SA hyperparameters using normalized NN space.
    
    Args:
        nn_step: Step size in NN output space [-5, 5]. 
                 Step=1.0 gives 11 points per axis.
                 Step=0.5 gives 21 points per axis.
        verbose: Print detailed progress
    """
    nn_grid = get_nn_grid(nn_step)
    grid_size = len(nn_grid)
    
    print(f"{'='*60}")
    print(f"Starting Grid Search Analysis")
    print(f"  NN step size: {nn_step}")
    print(f"  Grid points per axis: {grid_size}")
    print(f"  NN values: {nn_grid}")
    print(f"{'='*60}")
    
    # Initialize database connection
    # conn = init_cache_db()
    initial_cache_size = get_cache_stats(conn)
    print(f"[CACHE] SQLite database: {CACHE_DB}")
    print(f"[CACHE] Initial cache size: {initial_cache_size} results")
    
    # Get default NN value (middle of range = 0)
    default_nn = 0.0
    
    # Pre-compute parameter grids
    param_grids = {
        name: get_param_grid(name, nn_step) 
        for name in PARAM_RANGES.keys()
    }
    
    # Print parameter mappings
    print("\nParameter mappings:")
    for name, values in param_grids.items():
        print(f"  {name}: [{values[0]:.4g}, ..., {values[-1]:.4g}]")
    print()
    
    pairs = [
        ('init_temp', 'cooling_rate'),
        ('init_temp', 'step_size'),
        ('init_temp', 'num_steps'),
        ('cooling_rate', 'step_size'),
        ('cooling_rate', 'num_steps'),
        ('step_size', 'num_steps')
    ]
    
    total_pairs = len(pairs)
    for pair_idx, (p1_name, p2_name) in enumerate(pairs, 1):
        p1_nn_values = nn_grid
        p2_nn_values = nn_grid
        p1_param_values = param_grids[p1_name]
        p2_param_values = param_grids[p2_name]
        
        total_configs = len(p1_nn_values) * len(p2_nn_values)
        print(f"\n[PAIR {pair_idx}/{total_pairs}] {p1_name} vs {p2_name} ({total_configs} configurations)")
        print(f"-" * 50)
        
        grid_cost = np.zeros((len(p2_nn_values), len(p1_nn_values)))
        cache_hits = 0
        cache_misses = 0
        config_count = 0
        
        for i, (nn1, param1) in enumerate(zip(p1_nn_values, p1_param_values)):
            for j, (nn2, param2) in enumerate(zip(p2_nn_values, p2_param_values)):
                config_count += 1
                
                # NN-space parameters (for cache key)
                nn_params = {
                    'init_temp': default_nn,
                    'cooling_rate': default_nn,
                    'step_size': default_nn,
                    'num_steps': default_nn
                }
                nn_params[p1_name] = nn1
                nn_params[p2_name] = nn2
                
                # Actual SA parameters
                actual_params = {
                    'init_temp': nn_output_to_param(nn_params['init_temp'], 'init_temp'),
                    'cooling_rate': nn_output_to_param(nn_params['cooling_rate'], 'cooling_rate'),
                    'step_size': nn_output_to_param(nn_params['step_size'], 'step_size'),
                    'num_steps': nn_output_to_param(nn_params['num_steps'], 'num_steps')
                }
                
                # Check cache using NN-space keys
                cached = get_cached_result(conn, nn_params, SA_SEED, SA_NUM_RUNS)
                
                if cached is not None:
                    grid_cost[j, i] = cached
                    cache_hits += 1
                    if verbose and config_count % 50 == 0:
                        print(f"  [{config_count}/{total_configs}] Cache hit (hits: {cache_hits}, new: {cache_misses})")
                else:
                    # Run SA with configured number of runs
                    _, costs, _, _ = run_sa(
                        actual_params['init_temp'],
                        actual_params['cooling_rate'],
                        actual_params['step_size'],
                        actual_params['num_steps'],
                        SA_BOUNDS,
                        seed=SA_SEED,
                        num_runs=SA_NUM_RUNS
                    )
                    mean_cost = np.mean(costs)
                    grid_cost[j, i] = mean_cost
                    cache_misses += 1
                    
                    # IMMEDIATE SAVE: Write to database right after computation
                    save_result_immediately(conn, nn_params, actual_params, SA_SEED, SA_NUM_RUNS, mean_cost)
                    
                    if verbose:
                        print(f"  [{config_count}/{total_configs}] NEW: nn=({nn1:.1f},{nn2:.1f}) -> {p1_name}={param1:.4g}, {p2_name}={param2:.4g} -> cost={mean_cost:.4f}")
        
        print(f"  SUMMARY: Cache hits: {cache_hits}, New runs: {cache_misses}")
        
        # Plot with improved styling and power-law color scale
        # Using PowerNorm with gamma<1 expands the low range where most values fall
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Power-law normalization: gamma=0.4 expands 0-10 range significantly
        norm = mcolors.PowerNorm(gamma=1, vmin=COST_VMIN, vmax=COST_VMAX)
        
        im = ax.imshow(grid_cost, origin='lower', aspect='auto', cmap='RdYlGn_r',
                       extent=[NN_MIN, NN_MAX, NN_MIN, NN_MAX],
                       norm=norm)
        
        # Tick labels showing actual parameter values
        tick_indices = [0, len(nn_grid)//4, len(nn_grid)//2, 3*len(nn_grid)//4, len(nn_grid)-1]
        x_tick_nn = [nn_grid[i] for i in tick_indices]
        x_tick_labels = [f"{p1_param_values[i]:.2g}" for i in tick_indices]
        y_tick_nn = [nn_grid[i] for i in tick_indices]
        y_tick_labels = [f"{p2_param_values[i]:.2g}" for i in tick_indices]
        
        ax.set_xticks(x_tick_nn)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticks(y_tick_nn)
        ax.set_yticklabels(y_tick_labels)
        
        # Labels using actual parameter names
        low1, high1, scale1 = PARAM_RANGES[p1_name]
        low2, high2, scale2 = PARAM_RANGES[p2_name]
        ax.set_xlabel(f"{p1_name} [{low1:.2g} - {high1:.2g}]", fontsize=12)
        ax.set_ylabel(f"{p2_name} [{low2:.2g} - {high2:.2g}]", fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'Mean Cost (0 = Best, {COST_VMAX} = Worst)', fontsize=10)
        
        ax.set_title(f"SA Performance: {p1_name} vs {p2_name}", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fname = f"heatmap_{p1_name}_vs_{p2_name}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  [PLOT] Saved {fname}")
    
    # Final cache stats
    final_cache_size = get_cache_stats(conn)
    print(f"\n[CACHE] Final cache size: {final_cache_size} results")
    print(f"[CACHE] New results added: {final_cache_size - initial_cache_size}")
    
    conn.close()
    
    print(f"\n{'='*60}")
    print("Grid Search Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_grid_search(nn_step=1.0)
