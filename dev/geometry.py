import numpy as np

def cartesian_to_polar(x, y):
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return r, theta

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y