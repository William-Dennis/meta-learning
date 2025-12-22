
import numpy as np

from param_scaling import nn_output_to_param, NN_MIN, NN_MAX


class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class TuningEnv:
    """
    Environment for tuning SA hyperparameters.
    One step = One full SA run.
    
    Actions are in NN output space [-5, 5] and are converted to
    actual hyperparameters using the shared param_scaling module.
    """
    def __init__(self, seed=42):
        self.seed = seed
        
        # Action: 4D vector in [-5, 5] range for each parameter
        # [init_temp_nn, cooling_rate_nn, step_size_nn, num_steps_nn]
        self.action_space = Box(low=NN_MIN, high=NN_MAX, shape=(4,), dtype=np.float32)
        
        # State: Dummy constant state since the problem is static
        self.observation_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # SA bounds for Rastrigin function
        self.bounds = [-5.12, 5.12]
        self.np_random = np.random.default_rng(self.seed)
        
        # Store last run for visualization
        self.last_trajectory = []

    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng(self.seed)

        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        """
        Execute one SA run with the given action (NN output space).
        
        Args:
            action: numpy array of shape (4,) with values in [-5, 5]
                    [init_temp_nn, cooling_rate_nn, step_size_nn, num_steps_nn]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action to valid range
        action = np.clip(action, NN_MIN, NN_MAX)
        
        # Convert NN outputs to actual hyperparameters using shared scaling
        init_temp = nn_output_to_param(action[0], 'init_temp')
        cooling_rate = nn_output_to_param(action[1], 'cooling_rate')
        step_size = nn_output_to_param(action[2], 'step_size')
        num_steps = nn_output_to_param(action[3], 'num_steps')
        
        from sa_algorithm import run_sa
        
        # Generate a seed for this run to ensure reproducibility if needed 
        # but continuing the stream of the env's RNG.
        run_seed = self.np_random.integers(0, 2**32)
        
        # Run SA
        avg_reward, costs, last_trajectory, median_idx = run_sa(
            init_temp, cooling_rate, step_size, num_steps, self.bounds, seed=run_seed
        )

        self.last_trajectory = last_trajectory
        
        # Info includes the decoded params for logging
        info = {
            'init_temp': init_temp,
            'cooling_rate': cooling_rate,
            'step_size': step_size,
            'num_steps': num_steps,
            'nn_action': action.tolist(),  # Store the NN action for reference
            'final_cost': costs[median_idx],  # Log median cost
            'mean_cost': np.mean(costs)
        }
        
        return np.array([0.0], dtype=np.float32), avg_reward, True, False, info
