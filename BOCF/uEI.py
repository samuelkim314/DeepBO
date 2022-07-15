# Copyright (c) 2018, Raul Astudillo

import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from pathos.multiprocessing import ProcessingPool as Pool


class uEI(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details.
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, utility=None, noiseless=True):
        self.optimizer = optimizer
        self.utility = utility
        super(uEI, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients
        self.n_attributes = self.model.output_dim
        self.W_samples = np.random.normal(size=(1, self.n_attributes))
        self.Z_samples = np.random.normal(size=(1, self.n_attributes))
        self.n_hyps_samples = min(1, self.model.number_of_hyps_samples())
        self.use_full_support = self.utility.parameter_dist.use_full_support  # If true, the full support of the utility function distribution will be used when computing the acquisition function value.
        if self.use_full_support:
            self.utility_params_samples = self.utility.parameter_dist.support
            self.utility_prob_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        else:
            self.utility_params_samples = self.utility.parameter_dist.sample(10)
        self.noiseless = noiseless

    def _compute_acq(self, X, parallel=False):
        """
        Computes the aquisition function

        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        if parallel and len(X) > 1:
            marginal_acqX = self._marginal_acq_parallel(X)
        else:
            marginal_acqX = self._marginal_acq(X, self.utility_params_samples)
            # print('parallel')
            # print(marginal_acqX)
        # marginal_acqX = self._marginal_acq(X, utility_params_samples)
        # print('sequential')
        # print(marginal_acqX)
        if self.use_full_support:
            acqX = np.matmul(marginal_acqX, self.utility_prob_dist)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(self.utility_params_samples)
        acqX = np.reshape(acqX, (X.shape[0], 1))
        #print(acqX)
        return acqX

    def _marginal_acq(self, X, utility_params_samples):
        """
        """
        if self.noiseless:
            fX_evaluated = self.model.posterior_mean_at_evaluated_points()
        else:
            X_evaluated = self.model.get_evaluated_points()
        n_w = self.W_samples.shape[0]
        n_z = self.Z_samples.shape[0]
        L = len(utility_params_samples)
        marginal_acqX = np.zeros((X.shape[0], L))

        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            inv_sqrt_varX = (self.model.posterior_variance(X)) ** (-0.5)
            for i in range(len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                self.model.partial_precomputation_for_variance_conditioned_on_next_point(x)
                for l in range(L):
                    for W in self.W_samples:
                        aux = np.multiply(inv_sqrt_varX[:, i], W)

                        def inner_func(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            func_val = np.zeros((X_inner.shape[0], 1))
                            cross_cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner,
                                                                                                             x)[:, :, 0]
                            posterior_std_conditioned_on_next_point = np.sqrt(
                                self.model.posterior_variance_conditioned_on_next_point(X_inner))
                            a = self.model.posterior_mean(X_inner)
                            a += np.multiply(cross_cov.T, aux).T
                            for Z in self.Z_samples:
                                b = np.multiply(posterior_std_conditioned_on_next_point.T, Z).T
                                func_val[:, 0] += self.utility.eval_func(utility_params_samples[l], a + b)
                            return func_val

                        inner_val_x = inner_func(x)
                        if self.noiseless:
                            inner_val_X_evaluated = self.Z_samples.shape[0] * self.utility.eval_func(utility_params_samples[l], fX_evaluated)
                        else:
                            inner_val_X_evaluated = inner_func(X_evaluated)

                        marginal_acqX[i, l] += max(inner_val_x - np.max(inner_val_X_evaluated), 0)

        marginal_acqX /= (self.n_hyps_samples * n_w * n_z)
        return marginal_acqX

    def _marginal_acq_parallel(self, X):
        """
        """
        n_x = X.shape[0]
        marginal_acqX = np.zeros((X.shape[0], len(self.utility_params_samples)))
        n_w = self.W_samples.shape[0]
        n_z = self.Z_samples.shape[0]
        args = [[0 for i in range(2)] for j in range(n_x)]
        for i in range(n_x):
            args[i][0] = np.atleast_2d(X[i])
        pool = Pool(4)
        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            inv_sqrt_varX = (self.model.posterior_variance(X)) ** (-0.5)
            for i in range(n_x):
                args[i][1] = inv_sqrt_varX[:, i]
            marginal_acqX += np.atleast_2d(pool.map(self._parallel_acq_helper, args))

        marginal_acqX /= (self.n_hyps_samples * n_w * n_z)
        return marginal_acqX

    def _parallel_acq_helper(self, args):
        """
        """
        if self.noiseless:
            fX_evaluated = self.model.posterior_mean_at_evaluated_points()
        else:
            X_evaluated = self.model.get_evaluated_points()
        utility_params_samples = self.utility_params_samples
        L = len(utility_params_samples)
        marginal_acqx = np.zeros(L)
        x = args[0]
        inv_sqrt_varx = args[1]
        self.model.partial_precomputation_for_covariance(x)
        self.model.partial_precomputation_for_covariance_gradient(x)
        self.model.partial_precomputation_for_variance_conditioned_on_next_point(x)
        for l in range(L):
            for W in self.W_samples:
                aux = np.multiply(inv_sqrt_varx, W)

                def inner_func(X_inner):
                    X_inner = np.atleast_2d(X_inner)
                    func_val = np.zeros((X_inner.shape[0], 1))
                    cross_cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner,
                                                                                                     x)[:, :, 0]
                    posterior_std_conditioned_on_next_point = np.sqrt(
                        self.model.posterior_variance_conditioned_on_next_point(X_inner))
                    a = self.model.posterior_mean(X_inner)
                    a += np.multiply(cross_cov.T, aux).T
                    for Z in self.Z_samples:
                        b = np.multiply(posterior_std_conditioned_on_next_point.T, Z).T
                        func_val[:, 0] += self.utility.eval_func(utility_params_samples[l], a + b)
                    return func_val

                inner_val_x = inner_func(x)
                if self.noiseless:
                    inner_val_X_evaluated = self.Z_samples.shape[0] * self.utility.eval_func(utility_params_samples[l], fX_evaluated)
                else:
                    inner_val_X_evaluated = inner_func(X_evaluated)

                marginal_acqx[l] += max(inner_val_x - np.max(inner_val_X_evaluated), 0)

        return marginal_acqx

    def _compute_acq_withGradients(self, X):
        """
        """
        X = np.atleast_2d(X)
        # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples
        if self.use_full_support:
            utility_params_samples2 = self.utility.parameter_dist.support
        else:
            utility_params_samples2 = self.utility.parameter_dist.sample(1)
        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X, utility_params_samples2)
        if self.use_full_support:
            acqX = np.matmul(marginal_acqX, self.utility_prob_dist)
            dacq_dX = np.tensordot(marginal_dacq_dX, self.utility_prob_dist, 1)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(utility_params_samples2)
            dacq_dX = np.sum(marginal_dacq_dX, axis=2) / len(utility_params_samples2)
        acqX = np.reshape(acqX, (X.shape[0], 1))
        dacq_dX = np.reshape(dacq_dX, X.shape)
        return acqX, dacq_dX

    def _marginal_acq_with_gradient(self, X, utility_params_samples):
        """
        """
        if self.noiseless:
            fX_evaluated = self.model.posterior_mean_at_evaluated_points()
            #X_evaluated = self.model.get_evaluated_points()
        else:
            X_evaluated = self.model.get_evaluated_points()
        X = np.atleast_2d(X)
        marginal_acqX = np.zeros((X.shape[0], len(utility_params_samples)))
        marginal_dacq_dX = np.zeros((X.shape[0], X.shape[1], len(utility_params_samples)))
        W_samples2 = self.W_samples#np.random.normal(size=(1, self.n_attributes))
        Z_samples2 = self.Z_samples#np.random.normal(size=(1, self.n_attributes))

        n_w = W_samples2.shape[0]
        n_z = Z_samples2.shape[0]
        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            inv_sqrt_varX = (self.model.posterior_variance(X)) ** (-0.5)
            inv_varX_noiseless = np.reciprocal(self.model.posterior_variance_noiseless(X))
            dvar_dX = self.model.posterior_variance_gradient(X)
            for i in range(len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                self.model.partial_precomputation_for_variance_conditioned_on_next_point(x)
                for l in range(len(utility_params_samples)):
                    for W in W_samples2:
                        aux = np.multiply(inv_sqrt_varX[:, i], W)

                        def inner_func(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            func_val = np.zeros((X_inner.shape[0], 1))
                            cross_cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner,
                                                                                                             x)[:, :, 0]
                            posterior_std_conditioned_on_next_point = np.sqrt(
                                self.model.posterior_variance_conditioned_on_next_point(X_inner))
                            a = self.model.posterior_mean(X_inner)
                            a += np.multiply(cross_cov.T, aux).T
                            for Z in self.Z_samples:
                                b = np.multiply(posterior_std_conditioned_on_next_point.T, Z).T
                                func_val[:, 0] += self.utility.eval_func(utility_params_samples[l], a + b)
                            return func_val

                        inner_val_x = inner_func(x)
                        if self.noiseless:
                            inner_val_X_evaluated = self.Z_samples.shape[0] * self.utility.eval_func(utility_params_samples[l], fX_evaluated)
                        else:
                            inner_val_X_evaluated = inner_func(X_evaluated)

                        max_inner_val_X_evaluated = np.max(inner_val_X_evaluated)

                        marginal_acqX[i, l] += max(inner_val_x - max_inner_val_X_evaluated, 0)
                        if inner_val_x > max_inner_val_X_evaluated:
                            x_opt = x
                            cross_cov = self.model.posterior_covariance_between_points_partially_precomputed(x_opt, x)[
                                        :, :,
                                        0]
                            dcross_cov_dx = self.model.posterior_covariance_gradient(x, x_opt)[:, 0, :]
                            posterior_std_conditioned_on_x = np.sqrt(
                                self.model.posterior_variance_conditioned_on_next_point(x_opt))
                            a = self.model.posterior_mean(x_opt)
                            a += np.multiply(cross_cov.T, aux).T
                            aux2 = -0.5 * np.multiply(cross_cov.T, (inv_sqrt_varX[:, i] ** 3) * W)
                            c = np.multiply(dcross_cov_dx.T, aux).T + np.multiply(dvar_dX[:, i, :].T, aux2).T
                            tmp = cross_cov[:, 0] * inv_varX_noiseless[:, i]
                            dposterior_std_conditioned_on_x_dx = (-np.multiply(dcross_cov_dx.T,
                                                                               tmp).T + 0.5 * np.multiply(
                                dvar_dX[:, i, :].T, np.square(tmp)).T) / posterior_std_conditioned_on_x

                            for Z in Z_samples2:
                                b = np.multiply(posterior_std_conditioned_on_x.T, Z).T
                                d = np.multiply(dposterior_std_conditioned_on_x_dx.T, Z).T
                                marginal_dacq_dX[i, :, l] += np.matmul(
                                    self.utility.eval_gradient(utility_params_samples[l], a + b), c + d)


        marginal_acqX /= (self.n_hyps_samples * n_w * self.Z_samples.shape[0])
        marginal_dacq_dX /= (self.n_hyps_samples * n_w * n_z)
        return marginal_acqX, marginal_dacq_dX

    def update_Z_samples(self, n_samples):
        print('Update utility parameter W and Z samples')
        self.W_samples = np.random.normal(size=self.W_samples.shape)
        #self.Z_samples = np.random.normal(size=self.Z_samples.shape)
        #if not self.use_full_support:
            #self.utility_params_samples = self.utility.parameter_dist.sample(len(self.utility_params_samples))