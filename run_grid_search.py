"""Simple grid search runner for SA hyperparameter analysis."""

import numpy as np
from datetime import datetime
from core.sa_config import get_run_sa


def run_single_config(init_temp, cooling_rate, step_size, num_steps):
    """Run SA with specific hyperparameters."""
    bounds = [-5.12, 5.12]
    seed = 42
    num_runs = 10

    run_sa = get_run_sa()
    avg_reward, costs, _, _ = run_sa(
        init_temp, cooling_rate, step_size, num_steps, bounds, seed, num_runs
    )

    return np.mean(costs), np.std(costs)


def process_config(temp, cool, step, count, total):
    """Process a single configuration."""
    mean_cost, std_cost = run_single_config(temp, cool, step, 100) # fix num_steps to 100

    result = {
        "temp": temp,
        "cool": cool,
        "step": step,
        "mean": mean_cost,
        "std": std_cost,
    }

    print(
        f"[{count}/{total}] T={temp:.1f}, "
        f"α={cool:.2f}, S={step:.1f} -> "
        f"{mean_cost:.4f} ± {std_cost:.4f}"
    )

    return result, mean_cost


def grid_search():
    """Run grid search over SA hyperparameters."""
    print("Running grid search...")

    temp_values = [1.0, 10.0, 50.0]
    cooling_values = [0.90, 0.95, 0.99]
    step_values = [0.5, 1.0, 2.0]

    best_cost = float("inf")
    best_params = None
    results = []

    total = len(temp_values) * len(cooling_values) * len(step_values)
    count = 0

    for temp in temp_values:
        for cool in cooling_values:
            for step in step_values:
                count += 1
                result, mean_cost = process_config(temp, cool, step, count, total)

                results.append(result)

                if mean_cost < best_cost:
                    best_cost = mean_cost
                    best_params = (temp, cool, step)

    return best_params, best_cost, results


def save_results(best_params, best_cost, results):
    """Save grid search results."""
    with open("outputs/grid_search_results.txt", "w", encoding="utf-8") as f:
        f.write("Grid Search Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("Best Parameters:\n")
        temp, cool, step = best_params
        f.write(f"  Init Temp: {temp:.2f}\n")
        f.write(f"  Cooling Rate: {cool:.3f}\n")
        f.write(f"  Step Size: {step:.2f}\n")
        f.write(f"  Best Cost: {best_cost:.4f}\n\n")

        f.write("All Results:\n")
        for r in results:
            f.write(
                f"T={r['temp']:.1f}, α={r['cool']:.2f}, "
                f"S={r['step']:.1f}: {r['mean']:.4f} "
                f"± {r['std']:.4f}\n"
            )

    print("\nSaved outputs/grid_search_results.txt")


def main():
    """Run grid search with default settings."""
    print("=" * 60)
    print("GRID SEARCH RUNNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    best_params, best_cost, results = grid_search()

    print("\n" + "=" * 60)
    print("Best Configuration Found:")
    temp, cool, step = best_params
    print(f"  Init Temp: {temp:.2f}")
    print(f"  Cooling Rate: {cool:.3f}")
    print(f"  Step Size: {step:.2f}")
    print(f"  Mean Cost: {best_cost:.4f}")
    print("=" * 60)

    save_results(best_params, best_cost, results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
