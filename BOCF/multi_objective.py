# Copyright (c) 2018, Raul Astudillo Marban

#import time
import numpy as np
#from ...util.general import spawn
#from ...util.general import get_d_moments
#import GPy
import GPyOpt
from GPyOpt.core.task.objective import Objective


class MultiObjective(Objective):
    """
    Class to handle problems with multiple objective functions.

    param func: objective function.
    param objective_name: name of the objective function.


    .. Note:: every objective should take 2-dimensional numpy arrays as input and outputs. Each row should
    contain a location (in the case of the inputs) or a function evaluation (in the case of the outputs).
    """


    def __init__(self, func, noise_var=None, objective_name=None, as_list=True, output_dim=None):     
        self.func = func
        if as_list:
            self.output_dim  = len(self.func)
        else:
            self.output_dim = output_dim
        self.noise_var = noise_var
        if objective_name is None:
            self.objective_name = ['no_name']*self.output_dim
        else:
            self.objective_name = objective_name
        if as_list:   
            self.objective = [None]*self.output_dim
            for j in range(self.output_dim):
                self.objective[j] = GPyOpt.core.task.SingleObjective(func=self.func[j],objective_name=self.objective_name[j])
        else:
            self.objective = func
        self.as_list = as_list


    def evaluate(self, X):
        """
        Performs the evaluation of the objectives at x.
        """
        f_eval = [None]*self.output_dim #np.zeros(self.output_dim)
        cost_eval = 0
        if self.as_list:
            for j in range(self.output_dim):
                f_eval[j] = self.objective[j].evaluate(X)[0]
        else:
            fX = self.objective(X)
            for j in range(self.output_dim):
                f_eval[j] = np.reshape(fX[j,:], (X.shape[0],1))
        return f_eval, cost_eval


    def evaluate_as_array(self, X):
        """
        Performs the evaluation of the objectives at x.
        """
        f_eval = np.empty((self.output_dim, X.shape[0]))
        for j in range(self.output_dim):
            f_eval[j, :] = self.objective[j].evaluate(X)[0][:,0]
        return f_eval

    
    def evaluate_w_noise(self, X):
        """
        Performs the evaluation of the objectives at x.
        """
        f_noisy_eval, cost_eval = self.evaluate(X)
        if self.noise_var  is not None:           
            for j in range(self.output_dim):
                f_noisy_eval[j] += np.random.normal(scale = np.sqrt(self.noise_var[j]))
            
        return f_noisy_eval, cost_eval

    def get_output_dim(self):
        return self.output_dim