# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .optimizer import OptLbfgs, OptSGD, OptDirect, OptCma, apply_optimizer, choose_optimizer, apply_optimizer_inner
from .anchor_points_generator import ObjectiveAnchorPointsGenerator, ThompsonSamplingAnchorPointsGenerator
from ..core.task.space import Design_space
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np


max_objective_anchor_points_logic = "max_objective"
thompson_sampling_anchor_points_logic = "thompsom_sampling"
sobol_design_type = "sobol"
random_design_type = "random"
latin_design_type = "latin"


class GeneralOptimizer(object):
    """
    General class for acquisition optimizers defined in domains with mix of discrete, continuous, bandit variables

    :param space: design space class from GPyOpt.
    :param optimizer: optimizer to use. Can be selected among:
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    """

    def __init__(self, space, optimizer='lbfgs', inner_optimizer='lbfgs', **kwargs):

        self.space              = space
        self.optimizer_name     = optimizer
        self.inner_optimizer_name     = inner_optimizer
        self.kwargs             = kwargs

        ## -- save extra options than can be passed to the optimizer
        if 'model' in self.kwargs:
            self.model = self.kwargs['model']

        if 'anchor_points_logic' in self.kwargs:
            self.type_anchor_points_logic = self.kwargs['type_anchor_points_logic']
        else:
            self.type_anchor_points_logic = max_objective_anchor_points_logic

        ## -- Context handler: takes
        self.context_manager = ContextManager(space)
        ## -- Set optimizer and inner optimizer (WARNING: this won't update context)
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)
        self.inner_optimizer = choose_optimizer(self.inner_optimizer_name, self.context_manager.noncontext_bounds)
    
    
    def optimize(self, f=None, df=None, f_df=None, parallel=False, duplicate_manager=None, n_starting=200, n_anchor=24):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df
        

        ## --- Update the optimizer, in case context has beee passed.
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f, n_starting)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)
           
        ## -- Select the anchor points (with context)
        anchor_points, anchor_points_values = anchor_points_generator.get(num_anchor=n_anchor, duplicate_manager=duplicate_manager, context_manager=self.context_manager, get_scores=True)
        #print('anchor points')
        #print(anchor_points)
        #print(anchor_points_values)
        if parallel:
            pool = Pool(4)
            optimized_points = pool.map(self._parallel_optimization_wrapper, anchor_points)
            #print('optimized points')
            #:print(optimized_points)
        else:
            optimized_points = [apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]                 
                        
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])
        if np.asscalar(anchor_points_values[0]) < np.asscalar(fx_min):
            print('anchor_point was best found in evaluation')
            fx_min = np.atleast_2d(anchor_points_values[0])
            x_min = np.atleast_2d(anchor_points[0])
        return x_min, fx_min
    

    def optimize2(self, f=None, df=None, f_df=None, duplicate_manager=None):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df
        

        ## --- Update the optimizer, in case context has beee passed.
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, latin_design_type, f)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)
           
        ## -- Select the anchor points (with context)
        anchor_points = anchor_points_generator.get(duplicate_manager=duplicate_manager, context_manager=self.context_manager)
        print('anchor_points ready')
        print(anchor_points)
        pool = Pool(4)
        optimized_points = pool.map(self._parallel_optimization_wrapper, anchor_points)
        print('parallel')
        print(optimized_points)
        optimized_points2 = [apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]          
        print('sequential')
        print(optimized_points2)
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])
        return x_min, fx_min
    
    
    def optimize_inner_func(self, f=None, df=None, f_df=None, duplicate_manager=None):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df

        ## --- Update the optimizer, in case context has beee passed.
        self.inner_optimizer = choose_optimizer(self.inner_optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f, 64)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)
           
        ## -- Select the anchor points (with context)
        anchor_points, anchor_points_values = anchor_points_generator.get(num_anchor=8, duplicate_manager=duplicate_manager, context_manager=self.context_manager, get_scores=True)
        #print(anchor_points)
        
        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)
        optimized_points = [apply_optimizer_inner(self.inner_optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]
        #print('inner optimized points')
        #print(optimized_points)
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])
        #x_min = np.atleast_2d(anchor_points[0])
        #fx_min = np.atleast_2d(anchor_points_values[0])       
        return x_min, fx_min
    
    #def anchor_points_generator_wrapper(self, f)
    
    def _parallel_optimization_wrapper(self, x0):
        #print(x0)
        return apply_optimizer(self.optimizer, x0, self.f, None, self.f_df)


class ContextManager(object):
    """
    class to handle the context variable in the optimizer
    :param space: design space class from GPyOpt.
    :param context: dictionary of variables and their contex values
    """

    def __init__ (self, space, context = None):
        self.space              = space
        self.all_index          = list(range(space.model_dimensionality))
        self.all_index_obj      = list(range(len(self.space.config_space_expanded)))
        self.context_index      = []
        self.context_value      = []
        self.context_index_obj  = []
        self.nocontext_index_obj= self.all_index_obj
        self.noncontext_bounds  = self.space.get_bounds()[:]
        self.noncontext_index   = self.all_index[:]

        if context is not None:
            #print('context')

            ## -- Update new context
            for context_variable in context.keys():
                variable = self.space.find_variable(context_variable)
                self.context_index += variable.index_in_model
                self.context_index_obj += variable.index_in_objective
                self.context_value += variable.objective_to_model(context[context_variable])

            ## --- Get bounds and index for non context
            self.noncontext_index = [idx for idx in self.all_index if idx not in self.context_index]
            self.noncontext_bounds = [self.noncontext_bounds[idx] for idx in  self.noncontext_index]

            ## update non context index in objective
            self.nocontext_index_obj = [idx for idx in self.all_index_obj if idx not in self.context_index_obj]



    def _expand_vector(self,x):
        '''
        Takes a value x in the subspace of not fixed dimensions and expands it with the values of the fixed ones.
        :param x: input vector to be expanded by adding the context values
        '''
        x = np.atleast_2d(x)
        x_expanded = np.zeros((x.shape[0],self.space.model_dimensionality))
        x_expanded[:,np.array(self.noncontext_index).astype(int)]  = x
        x_expanded[:,np.array(self.context_index).astype(int)]  = self.context_value
        return x_expanded
