# Copyright (c) 2018, Raul Astudillo Marban

import time
import numpy as np


class Utility(object):
    """
    Class to handle a continuously differentiable utility function.

    param func: utility function.
    param dfunc: gradient of the utility function (if available).
    param parameter_space: space of parameters (Theta) of the utility function.
    param parameter_dist: distribution over the spaceof parameters.
    param linear: whether utility function is linear or not (this is used to save computations later; default, False)

    .. Note:: .
    """


    def __init__(self, func, dfunc=None, parameter_dist=None, linear=False):
        self.func  = func
        self.dfunc  = dfunc
        self.parameter_dist = parameter_dist
        self.linear = linear

    
    def evaluate_w_gradient(self, parameter, y):
        """
        Samples random parameter from parameter distribution and evaluates the utility function and its gradient at y given this parameter.
        """
        U_eval = self.eval_func(parameter,y)
        dU_eval = self._eval_gradient(parameter,y)
        return U_eval, dU_eval
    
    
    def eval_func(self, parameter, y):
        """
        Evaluates the utility function at y given a fixed parameter.
        """
        return self.func(parameter,y)
    
    
    def eval_gradient(self, parameter, y):
        """
        Evaluates the gradient f the utility function at y given a fixed parameter.
        """
        return self.dfunc(parameter,y)
