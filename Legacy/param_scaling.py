"""
Parameter scaling module for mapping NN outputs [-5, 5] to SA hyperparameters.
This provides a single source of truth for parameter scaling used by both
grid search analysis and PPO training.
"""
import numpy as np

# Define parameter ranges and their scaling type
# Format: (min_value, max_value, scale_type)
PARAM_RANGES = {
    'init_temp': (0.1, 100.0, 'log'),      # T0: log scale from 0.1 to 100
    'cooling_rate': (0.5, 0.99, 'linear'), # α: linear scale from 0.5 to 0.99
    'step_size': (0.1, 5.0, 'log'),        # step: log scale from 0.1 to 5.0
    'num_steps': (10, 1000, 'log_int'),     # N: log scale, integer, 20 to 300
}

# NN output bounds
NN_MIN = -5.0
NN_MAX = 5.0


def nn_to_normalized(nn_value):
    """
    Convert NN output [-5, 5] to normalized [0, 1] range.
    
    Args:
        nn_value: Value in [-5, 5] range
        
    Returns:
        Value in [0, 1] range
    """
    return (nn_value - NN_MIN) / (NN_MAX - NN_MIN)


def normalized_to_nn(t):
    """
    Convert normalized [0, 1] to NN output [-5, 5] range.
    
    Args:
        t: Value in [0, 1] range
        
    Returns:
        Value in [-5, 5] range
    """
    return NN_MIN + t * (NN_MAX - NN_MIN)


def nn_output_to_param(nn_value, param_name):
    """
    Map NN output [-5, 5] to actual parameter value.
    
    Args:
        nn_value: NN output in [-5, 5] range
        param_name: One of 'init_temp', 'cooling_rate', 'step_size', 'num_steps'
        
    Returns:
        Actual parameter value in the appropriate range
    """
    low, high, scale_type = PARAM_RANGES[param_name]
    t = nn_to_normalized(nn_value)  # Convert to [0, 1]
    
    if scale_type in ('log', 'log_int'):
        # Logarithmic interpolation
        value = low * (high / low) ** t
        if scale_type == 'log_int':
            value = int(round(value))
    else:  # linear
        value = low + t * (high - low)
    
    return value


def param_to_nn_output(param_value, param_name):
    """
    Map actual parameter value to NN output [-5, 5].
    Inverse of nn_output_to_param.
    
    Args:
        param_value: Actual parameter value
        param_name: One of 'init_temp', 'cooling_rate', 'step_size', 'num_steps'
        
    Returns:
        NN output in [-5, 5] range
    """
    low, high, scale_type = PARAM_RANGES[param_name]
    
    if scale_type in ('log', 'log_int'):
        # Inverse logarithmic interpolation
        t = np.log(param_value / low) / np.log(high / low)
    else:  # linear
        t = (param_value - low) / (high - low)
    
    return normalized_to_nn(t)


def get_nn_grid(step=1.0):
    """
    Generate a grid of NN output values with the specified step size.
    
    Args:
        step: Step size in NN space (default 1.0 gives [-5,-4,...,4,5])
        
    Returns:
        numpy array of NN values from -5 to 5 with given step
    """
    return np.arange(NN_MIN, NN_MAX + step/2, step)  # +step/2 to include endpoint


def get_param_grid(param_name, step=1.0):
    """
    Generate a grid of actual parameter values corresponding to NN grid.
    
    Args:
        param_name: Parameter name
        step: Step size in NN space
        
    Returns:
        numpy array of actual parameter values
    """
    nn_values = get_nn_grid(step)
    return np.array([nn_output_to_param(v, param_name) for v in nn_values])


def get_default_value(param_name):
    """
    Get the default/middle value for a parameter (corresponds to NN output = 0).
    
    Args:
        param_name: Parameter name
        
    Returns:
        Default parameter value
    """
    return nn_output_to_param(0.0, param_name)


# Print parameter ranges when imported for reference
if __name__ == "__main__":
    print("Parameter Scaling Reference")
    print("=" * 60)
    print(f"NN output range: [{NN_MIN}, {NN_MAX}]")
    print()
    
    for param_name, (low, high, scale_type) in PARAM_RANGES.items():
        print(f"{param_name} ({scale_type}):")
        print(f"  Range: [{low}, {high}]")
        print(f"  NN=-5 -> {nn_output_to_param(-5, param_name)}")
        print(f"  NN=0  -> {nn_output_to_param(0, param_name)}")
        print(f"  NN=+5 -> {nn_output_to_param(5, param_name)}")
        print()
    
    print("Grid with step=1.0:")
    nn_grid = get_nn_grid(1.0)
    print(f"  NN values: {nn_grid}")
    print(f"  Grid size: {len(nn_grid)} points")
