# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy
from .base import BOModel


class GPModelFixedHyps(BOModel):
    """
    General class for handling a Gaussian Process in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param sparse: whether to use a sparse GP (default, False). This is useful when many observations are available.
    :param num_inducing: number of inducing points if a sparse GP is used.
    :param verbose: print out the model messages (default, False).
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note::

    """
    analytical_gradient_prediction = True  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, kernel=None, noise_var=None, ARD=False, exact_feval=False, n_samples=10, kernel_hyps=None, verbose=False):
        self.kernel = kernel
        self.noise_var = noise_var
        self.ARD = ARD
        self.kernel_hyps = kernel_hyps
        self.verbose = verbose
        self.model = None
        self.current_model = None

    @staticmethod
    def fromConfig(config):
        return GPModelFixedHyps(**config)

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        print('create model')
        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.SE(input_dim=self.input_dim, variance=2., lengthscale=0.3)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        noise_var = 1e-10 if self.noise_var is None else self.noise_var
        self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        self.current_model = self.model


    def updateModel(self, X_all, Y_all, X_new=None, Y_new=None):
        """
        Updates the model with new observations.
        """

        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

    def number_of_hyps_samples(self):
        """
        """
        return self.n_samples

    def set_hyperparameters(self, i):
        pass

    def predict(self, X, full_cov=False):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """
        if X.ndim == 1: X = X[None, :]
        m, v = self.current_model.predict(X, full_cov)
        # m += self.constant_mean
        v = np.clip(v, 1e-10, np.inf)
        return m, v

    def predict_noiseless(self, X, full_cov=False):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """
        if X.ndim == 1: X = X[None, :]
        m = self.posterior_mean(X)
        v = self.posterior_variance_noiseless(X)

        return m, v

    def posterior_mean(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """
        if X.ndim == 1: X = X[None, :]
        return self.current_model.posterior_mean(X)

    def posterior_variance(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """
        if X.ndim == 1: X = X[None, :]
        v = np.clip(self.current_model.posterior_variance(X), 1e-10, np.inf)
        return v

    def posterior_variance_noiseless(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """
        if X.ndim == 1: X = X[None, :]
        v = np.clip(self.current_model.posterior_variance_noiseless(X), 1e-10, np.inf)
        return v

    def partial_precomputation_for_covariance(self, X):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        self.current_model.partial_precomputation_for_covariance(X)

    def partial_precomputation_for_covariance_gradient(self, x):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        self.current_model.partial_precomputation_for_covariance_gradient(x)

    def partial_precomputation_for_variance_conditioned_on_next_point(self, next_point):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        self.current_model.partial_precomputation_for_variance_conditioned_on_next_point(next_point)

    def posterior_variance_conditioned_on_next_point(self, X):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        return self.current_model.posterior_variance_conditioned_on_next_point(X)

    def posterior_variance_gradient_conditioned_on_next_point(self, X):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        return self.current_model.posterior_variance_gradient_conditioned_on_next_point(X)

    def posterior_covariance_between_points(self, X1, X2):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        return self.current_model.posterior_covariance_between_points(X1, X2)

    def posterior_covariance_between_points_partially_precomputed(self, X1, X2):
        """
        Computes the posterior covariance between points.

        :param kern: GP kernel
        :param X: current input observations
        :param X1: some input observations
        :param X2: other input observations
        """
        return self.current_model.posterior_covariance_between_points_partially_precomputed(X1, X2)

    def get_fmin(self):
        """
        Returns the location where the posterior mean takes its minimal value.
        """
        return self.current_model.predict(self.model.X)[0].min()

    def posterior_mean_gradient(self, X):
        """
        Computes the gradient of the posterior mean at X.
        :param X:  input observations
        """
        return self.current_model.posterior_mean_gradient(X)

    def posterior_variance_gradient(self, X):
        """
        Computes the gradient of the posterior variance at X.
        :param X:  input observations
        """
        return self.current_model.posterior_variance_gradient(X)

    def posterior_covariance_gradient(self, X, X2):
        """
        Computes dK/dx(X,X2).
        :param x: input obersevation.
        :param X2:  input observations
        """
        return self.current_model.posterior_covariance_gradient(X, X2)

    def posterior_covariance_gradient_partially_precomputed(self, X, x2):
        """
        Compute the derivatives of the posterior covariance, K^(n)(X,x2), with respect to X.
        """
        return self.current_model.posterior_covariance_gradient_partially_precomputed(X, x2)

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim == 1: X = X[None, :]
        m, v = self.current_model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.current_model.predictive_gradients(X)
        dmdx = dmdx[:, :, 0]
        dsdx = dvdx / (2 * np.sqrt(v))

        return m, np.sqrt(v), dmdx, dsdx

    def copy(self):
        """
        Makes a safe copy of the model.
        """
        copied_model = GPModel(kernel=self.model.kern.copy(),
                               noise_var=self.noise_var,
                               exact_feval=self.exact_feval,
                               optimizer=self.optimizer,
                               max_iters=self.max_iters,
                               optimize_restarts=self.optimize_restarts,
                               verbose=self.verbose,
                               ARD=self.ARD)

        copied_model._create_model(self.model.X, self.model.Y)
        copied_model.updateModel(self.model.X, self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        print(np.atleast_2d(self.model[:]))
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        print(self.model.parameter_names_flat().tolist())
        return self.model.parameter_names_flat().tolist()
