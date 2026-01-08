import numpy as np

def rastrigin_2d(x, y):
    """2D Rastrigin function"""
    scale = 1.5
    x = x / scale
    y = y / scale
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

def quadratic_2d(x, y):
    """Simple 2D quadratic function: z = x^2 + y^2"""
    return x**2 + y**2
