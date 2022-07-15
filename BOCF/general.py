import numpy as np

def unif_2d(n):
    """
    Function that generates n uniformly spaced points on the positive sector of the unit circle.

    param n: number of points.
    param objective_name: name of the objective function.
    """
    output = [None]*n
    for i in range(n):
        arg = 0.5*np.pi*(i)/(n-1)
        output[i] = [np.cos(arg), np.sin(arg)]
    return output
        
