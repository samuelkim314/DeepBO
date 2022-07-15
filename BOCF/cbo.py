# Copyright (c) 2018, Raul Astudillo
import GPyOpt
import collections
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from GPyOpt.experiment_design import initial_design
from GPyOpt.util.general import best_value
from GPyOpt.util.duplicate_manager import DuplicateManager
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.cost import CostModel
from GPyOpt.optimization.acquisition_optimizer import ContextManager
from plotting_services import plot, plot_convergence, plot_acquisition, integrated_plot
from pathos.multiprocessing import ProcessingPool as Pool
from copy import deepcopy


class CBO(object):
    """
    Runner of the multi-attribute Bayesian optimization loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: GPyOpt DuplicateManager class. Avoids re-evaluating the objective at previous, pending or infeasible locations (default, False).
    """


    def __init__(self, model, space, objective, acquisition, evaluator, X_init, Y_init=None, cost=None, normalize_Y = False, model_update_interval = 1, expectation_utility=None):
        self.model = model
        self.space = space
        self.objective = objective
        self.acquisition = acquisition
        self.utility = acquisition.utility
        self.expectation_utility = expectation_utility
        self.evaluator = evaluator
        self.X = X_init
        self.Y = Y_init
        self.normalize_Y = normalize_Y
        self.cost = CostModel(cost)
        self.model_update_interval = model_update_interval
        self.historical_optimal_values = []
        self.historical_time = []
        self.n_attributes = self.model.output_dim
        self.n_hyps_samples = min(1, self.model.number_of_hyps_samples())
        self.n_parameter_samples = 10
        self.full_parameter_support = self.utility.parameter_dist.use_full_support
        self.evaluation_optimizer = GPyOpt.optimization.GeneralOptimizer(optimizer='lbfgs', space=space)
        #
        self.context = None
        self.current_argmax = np.atleast_2d(X_init[0,:])


    def _current_max_value(self):
        """
        Computes E_n[U(f(x_max))|f], where U is the utility function, f is the true underlying ojective function and x_max = argmax E_n[U(f(x))|U]. See
        function _marginal_max_value_so_far below.
        """
        val = 0
        if self.full_parameter_support:
            utility_param_support = self.utility.parameter_dist.support
            utility_param_dist = self.utility.parameter_dist.prob_dist
            for i in range(len(utility_param_support)):
                marginal_argmax = self._current_marginal_argmax(utility_param_support[i])
                marginal_max_val = np.reshape(self.objective.evaluate(marginal_argmax)[0],(self.n_attributes,))
                val += self.utility.eval_func(utility_param_support[i],marginal_max_val)*utility_param_dist[i]
        else:
            utility_param_samples = self.utility.parameter_dist.sample(self.n_parameter_samples)
            for i in range(len(utility_param_samples)):
                marginal_argmax = self._current_marginal_argmax(utility_param_samples[i])
                marginal_max_val = np.reshape(self.objective.evaluate(marginal_argmax)[0],(self.n_attributes,))
                val += self.utility.eval_func(utility_param_samples[i],marginal_max_val)
            val /= len(utility_param_samples)
        print('Current optimal value: {}'.format(val))
        return val


    def _current_max_value_and_var(self):
        val = 0
        var = 0
        support = self.utility.parameter_dist.support
        utility_dist = self.utility.parameter_dist.prob_dist
        for i in range(len(support)):
            marginal_argmax = self._current_marginal_argmax(support[i])
            marginal_max_val = np.reshape(self.objective.evaluate(marginal_argmax)[0],(self.n_attributes,))
            var_marginal_argmax = np.reshape(self.model.posterior_variance_noiseless(marginal_argmax),(self.n_attributes,))
            var += self.utility.eval_func(support[i],var_marginal_argmax)*utility_dist[i]
            val += self.utility.eval_func(support[i],marginal_max_val)*utility_dist[i]
        print('Current optimal value: {}'.format(val))
        return val, var


    def _current_max_value_parallel(self):
        """
        Computes E_n[U(f(x_max))|f], where U is the utility function, f is the true underlying ojective function and x_max = argmax E_n[U(f(x))|U]. See
        function _marginal_max_value_so_far below.
        """
        pool = Pool(4)
        utility_param_samples = self.utility.parameter_dist.sample(self.n_parameter_samples)
        val = sum(pool.map(self._current_marginal_max_value, utility_param_samples))/self.n_parameter_samples
        print('Current optimal value')
        print(val)
        return val


    def _current_marginal_max_value(self, parameter):
        marginal_argmax = self._current_marginal_argmax(parameter)
        marginal_max_val = np.reshape(self.objective.evaluate(marginal_argmax)[0], (self.objective.output_dim,))
        return self.utility.eval_func(parameter, marginal_max_val)



    def _current_marginal_argmax(self, parameter):
        """
        Computes argmax E_n[U(f(x))|U] (The abuse of notation can be misleading; note that the expectation is with
        respect to the posterior distribution on f after n evaluations)
        """
        if self.utility.linear:
            def val_func(X):
                X = np.atleast_2d(X)
                muX = self.model.posterior_mean(X)
                valX = np.reshape(np.matmul(parameter, muX), (X.shape[0], 1))
                return -valX

            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                muX = self.model.posterior_mean(X)
                dmu_dX = self.model.posterior_mean_gradient(X)
                valX = np.reshape(np.matmul(parameter, muX), (X.shape[0], 1))
                dval_dX = np.tensordot(parameter, dmu_dX, axes=1)
                return -valX, -dval_dX

        elif self.expectation_utility is not None:
            # Note: the value of these functions is not normalized, i.e. it is not divided by the number of Z samples and GP hyps.

            def val_func(X):
                X = np.atleast_2d(X)
                func_val = np.empty((X.shape[0], 1))
                for h in range(self.n_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    for i in range(X.shape[0]):
                        func_val[i,0] = self.expectation_utility.func(parameter, mean[:,i], var[:,i])
                return -func_val

            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                func_val = np.zeros((X.shape[0], 1))
                func_gradient = np.zeros(X.shape)
                for h in range(self.n_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    dmean_dX = self.model.posterior_mean_gradient(X)
                    dvar_dX = self.model.posterior_variance_gradient(X)
                    aux = np.concatenate((dmean_dX,dvar_dX))
                    for i in range(X.shape[0]):
                        func_val[i,0] = self.expectation_utility.func(parameter, mean[:,i], var[:,i])
                        func_gradient[i,:] = np.matmul(self.expectation_utility.gradient(parameter,mean[:,i],var[:,i]),aux[:,i])
                return -func_val, -func_gradient

        else:
            Z_samples = np.random.normal(size=(50, self.n_attributes))
            def val_func(X):
                X = np.atleast_2d(X)
                func_val = np.zeros((X.shape[0], 1))
                for h in range(self.n_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    std = np.sqrt(var)
                    for i in range(X.shape[0]):
                        for Z in Z_samples:
                            func_val[i,0] += self.utility.eval_func(parameter,mean[:,i] + np.multiply(std[:,i],Z))
                return -func_val

            def val_func_with_gradient(X):
                X = np.atleast_2d(X)
                func_val = np.zeros((X.shape[0], 1))
                func_gradient = np.zeros(X.shape)
                for h in range(self.n_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(X)
                    std = np.sqrt(var)
                    dmean_dX = self.model.posterior_mean_gradient(X)
                    dstd_dX = self.model.posterior_variance_gradient(X)
                    for i in range(X.shape[0]):
                        for j in range(self.n_attributes):
                            dstd_dX[j,i,:] /= (2*std[j,i])
                        for Z in Z_samples:
                            aux1 = mean[:,i] + np.multiply(Z, std[:,i])
                            func_val[i,0] += self.utility.eval_func(parameter, aux1)
                            aux2 = dmean_dX[:,i,:] + np.multiply(dstd_dX[:,i,:].T, Z).T
                            func_gradient[i,:] += np.matmul(self.utility.eval_gradient(parameter, aux1), aux2)
                return -func_val, -func_gradient

        argmax = self.evaluation_optimizer.optimize(f=val_func, f_df=val_func_with_gradient, parallel=False)[0]
        self.current_argmax = argmax
        return argmax


    def run_optimization(self, max_iter=1, parallel=False, plot=False, results_file=None,  max_time=np.inf,  eps=1e-8, context=None, verbosity=False):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param evaluations_file: filename of the file where the evaluated points and corresponding evaluations are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.results_file = results_file
        self.context = context


        # --- Setting up stop conditions
        self.eps = eps
        if (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)
        # --- Initialize model
        self.model.updateModel(self.X,self.Y)

        # --- Initialize iterations and running time
        self.time_zero = time.clock()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y


        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time) and (self.num_acquisitions < self.max_iter):

            #if not ((self.num_acquisitions < self.max_iter) and (self._distance_last_evaluations() > self.eps)):

            tmp = self.suggested_sample
            self.suggested_sample = self.compute_next_evaluations()
            if np.all(self.suggested_sample == tmp):
                self.suggested_sample = self._perturb(self.suggested_sample)
            try:
                self.acquisition.update_Z_samples()
            except:
                pass
            # --- Augment X
            self.X = np.vstack((self.X, self.suggested_sample))

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            print('Acquisition {}'.format(self.num_acquisitions+1))
            self.evaluate_objective()
            # --- Update model
            if (self.num_acquisitions%self.model_update_interval)==0:
                self._update_model()
            self.model.get_model_parameters_names()
            self.model.get_model_parameters()
            if parallel and (not self.full_parameter_support):
                current_max_val = self._current_max_value_parallel()
            else:    
                current_max_val = self._current_max_value()
            self.historical_optimal_values.append(current_max_val)

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.clock() - self.time_zero
            self.historical_time.append(self.cum_time)
            self.num_acquisitions += 1

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))
        if results_file is not None:
            self.save_results(results_file)
        if plot:
            self.plot_convergence(confidence_interval=True)
            #self._plot_pareto_front()

        # --- Print the desired result in files
        #if self.evaluations_file is not None:
            #self.save_evaluations(self.evaluations_file)

        #file = open('test_file.txt','w')
        #plt.plot(range(self.num_acquisitions),value_so_far)
        #plt.show()
        #np.savetxt('test_file.txt',value_so_far)


    def best_evaluated(self):
        if self.n_attributes > 1:
            raise InvalidConfigError("This option is not avialable with multiple objectives")
        else:
            scores = self.Y[0].flatten()
            x_best = self.X[np.argsort(-scores)[0],:]
            fx_best = -np.sort(-scores)
            return x_best, fx_best


    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        print('Suggested point to evaluate: {}'.format(self.suggested_sample))
        self.Y_new, cost_new = self.objective.evaluate_w_noise(self.suggested_sample)
        self.cost.update_cost_model(self.suggested_sample, cost_new)
        for j in range(self.n_attributes):
            #print(self.Y_new[j])
            self.Y[j] = np.vstack((self.Y[j],self.Y_new[j]))


    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        return np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))

    def _perturb(self, x):
        perturbed_x = np.copy(x)
        while np.all(perturbed_x == x):
            perturbed_x = x + np.random.normal(size=x.shape, scale=1e-2)
            perturbed_x = self.space.round_optimum(perturbed_x)

        return perturbed_x


    def compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """
        # --- Initial function evaluation
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)
        # --- Initialize model
        self.model.updateModel(self.X, self.Y)

        ## --- Update the context if any
        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)

        ### We zip the value in case there are categorical variables
        x_baseline = self.current_argmax + np.random.normal(loc=0.0, scale=0.1, size=self.current_argmax.shape)
        return self.space.zip_inputs(self.evaluator.compute_batch(duplicate_manager=None, x_baseline=self.current_argmax))
        #return self.space.zip_inputs(self.evaluator.compute_batch(duplicate_manager=None))
        #return initial_design('random', self.space, 1)

    def _update_model(self):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """

        ### --- input that goes into the model (is unziped in case there are categorical variables)
        X_inmodel = self.space.unzip_inputs(self.X)
        Y_inmodel = list(self.Y)
        #print(X_inmodel)
        #print(Y_inmodel)
        self.model.updateModel(X_inmodel, Y_inmodel)

        ### --- Save parameters of the model
        #self._save_model_parameter_values()

    def convergence_assesment(self, n_iter=10, attribute=0, context=None):
        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")
        #self.model_parameters_iterations = None
        self.context = context
        # --- Initial function evaluation
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)
            self._update_model()
        for i in range(n_iter):
            self.suggested_sample = self.compute_next_evaluations()
            filename = './experiments/1d' + str(i) + '.eps'
            model_to_plot = deepcopy(self.model)
            integrated_plot(self.acquisition.space.get_bounds(),
                                    self.X.shape[1],
                                    model_to_plot,
                                    self.X,
                                    self.Y,
                                    self.acquisition.acquisition_function,
                                    self.suggested_sample,
                                    attribute,
                                    filename)

            self.X = np.vstack((self.X,self.suggested_sample))
            self.evaluate_objective()
            self._update_model()
            #self.model.get_model_parameters_names()
            #self.model.get_model_parameters()
            #print('Acquisition value at previously evaluated points:')
            #print(self.acquisition.acquisition_function(self.X))
            #print('Posterior mean and variance')
            #print(self.model.predict(self.X))
            #print(self.Y)
            self.historical_optimal_values.append(self._current_max_value())



    def one_step_assesment(self, attribute=0, context=None):
        """
        """
        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")
        #self.model_parameters_iterations = None
        self.context = context
        # --- Initial function evaluation
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)
            self._update_model()

        self.suggested_sample = self.compute_next_evaluations()

        model_to_plot = deepcopy(self.model)

        integrated_plot(self.acquisition.space.get_bounds(),
                                self.X.shape[1],
                                model_to_plot,
                                self.X,
                                self.Y,
                                self.acquisition.acquisition_function,
                                self.suggested_sample,
                                attribute,
                                None)

        self.X = np.vstack((self.X,self.suggested_sample))
        self.evaluate_objective()
        self._update_model()
        self.historical_optimal_values.append(self._current_max_value())

    def integrated_plot(self, attribute=0, filename=None):
        """
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
        :param filename: name of the file where the plot is saved
        """
        from copy import deepcopy
        model_to_plot = deepcopy(self.model)

        integrated_plot(self.acquisition.space.get_bounds(),
                                self.X.shape[1],
                                model_to_plot,
                                self.X,
                                self.Y,
                                self.acquisition.acquisition_function,
                                self.suggest_next_locations(),
                                attribute,
                                filename)

    def plot_acquisition(self,filename=None):
        """
        Plots the acquisition function.
        """

        return plot_acquisition(self.acquisition.space.get_bounds(),
                                self.X.shape[1],
                                self.acquisition.acquisition_function,
                                filename)


    def plot_convergence(self, confidence_interval=False, filename=None):
        """
        Makes twp plots to evaluate the convergence of the model:
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
        :param filename: name of the file where the plot is saved
        """
        return plot_convergence(self.historical_optimal_values, self.var_at_historical_optima, confidence_interval, filename)


    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()


    def save_results(self, filename):

        results = np.zeros((len(self.historical_optimal_values),2))
        results[:, 0] = np.atleast_1d(self.historical_optimal_values)
        results[:, 1] = np.atleast_1d(self.historical_time)

        np.savetxt(filename,results)