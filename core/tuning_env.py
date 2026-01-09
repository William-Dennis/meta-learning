import numpy as np
from core.sa_config import get_run_sa, get_objective_function


class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


def map_action_to_param(value, param_name):
    """Map action [-5, 5] to parameter range."""
    ranges = {
        "init_temp": (0.1, 100.0, "log"),
        "cooling_rate": (0.01, 0.99, "linear"),
        "step_size": (0.1, 5.0, "log"),
        "num_steps": (10, 1000, "log_int"),
    }
    low, high, scale_type = ranges[param_name]
    t = (value + 5.0) / 10.0

    if scale_type in ("log", "log_int"):
        result = low * (high / low) ** t
        if scale_type == "log_int":
            result = int(round(result))
    else:
        result = low + t * (high - low)

    return result


class TuningEnv:
    """Environment for tuning SA hyperparameters."""

    def __init__(self, seed=42, objective_function=None):
        self.seed = seed
        self.action_space = Box(low=-5.0, high=5.0, shape=(4,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.bounds = [-5.12, 5.12]
        self.np_random = np.random.default_rng(self.seed)
        self.last_trajectory = []
        self.optim = get_run_sa()
        self.objective_function = (
            objective_function
            if objective_function is not None
            else get_objective_function("rastrigin")
        )

    def reset(self, seed=None):
        self.np_random = np.random.default_rng(seed if seed is not None else self.seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        """Execute one SA run with the given action.
        TODO: eventually we want this to be optimiser agnostic. Where _get_params directly feeds into self.optim
        """
        action = np.clip(action, -5.0, 5.0)
        params = self._get_params(action)

        run_seed = self.np_random.integers(0, 2**32)
        # run_seed = self.seed # fixed seed for more stable training

        avg_reward, costs, trajectory, median_idx = self.optim(
            params["init_temp"],
            params["cooling_rate"],
            params["step_size"],
            params["num_steps"],
            self.bounds,
            seed=run_seed,
            num_runs=100,
            function=self.objective_function,
        )

        # Apply penalties here
        avg_reward -= params["num_steps"] * 1e-3
        avg_reward -= np.std(costs) * 1e-3

        self.last_trajectory = trajectory

        info = self._build_info(params, action, costs, median_idx)

        return (np.array([0.0], dtype=np.float32), avg_reward, True, False, info)

    def _get_params(self, action):
        """Convert action to SA parameters."""
        return {
            "init_temp": map_action_to_param(action[0], "init_temp"),
            "cooling_rate": map_action_to_param(action[1], "cooling_rate"),
            "step_size": map_action_to_param(action[2], "step_size"),
            "num_steps": map_action_to_param(action[3], "num_steps"),
        }

    def _build_info(self, params, action, costs, median_idx):
        """Build info dictionary."""
        return {
            "init_temp": params["init_temp"],
            "cooling_rate": params["cooling_rate"],
            "step_size": params["step_size"],
            "num_steps": params["num_steps"],
            "nn_action": action.tolist(),
            "final_cost": costs[median_idx],
            "mean_cost": np.mean(costs),
        }
