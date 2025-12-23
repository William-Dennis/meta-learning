"""
Runner script for grid search analysis.
Uses a single NN_STEP to control grid resolution.
"""
from grid_search_analysis import run_grid_search
import time
from datetime import datetime

# =============================================================================
# SETTINGS
# =============================================================================
NN_STEP = 0.1  # Controls grid resolution: 1.0 = 11 points per axis, 0.5 = 21 points, etc.
# =============================================================================

def main(nn_step = NN_STEP):
    print("=" * 60)
    print("GRID SEARCH RUNNER - Normalized NN Space")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"NN_STEP: {nn_step}")
    print("=" * 60)
    
    start = time.perf_counter()
    run_grid_search(nn_step=nn_step, verbose=True)
    duration = time.perf_counter() - start
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Complete! Duration: {duration:.2f}s")
    print(f"Analysis used NN_STEP={nn_step}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main(0.05)
