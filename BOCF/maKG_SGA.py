# Copyright (c) 2018, Raul Astudillo

import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from pathos.multiprocessing import ProcessingPool as Pool


class maKG_SGA(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details.
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, utility=None, normalize=False):
        self.optimizer = optimizer
        self.utility = utility
        self.normalize = normalize
        super(maKG_SGA, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

        self.Z_samples = np.random.normal(size=15)
        self.n_hyps_samples = min(1, self.model.number_of_hyps_samples())
        self.use_full_support = self.utility.parameter_dist.use_full_support  # If true, the full support of the utility function distribution will be used when computing the acquisition function value.
        self.acq_mean = 0.
        self.acq_std = 1.
        if self.use_full_support:
            self.utility_params_samples = self.utility.parameter_dist.support
            self.utility_prob_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        else:
            self.utility_params_samples = self.utility.parameter_dist.sample(1)

    def _compute_acq(self, X, parallel=True):
        """
        Computes the aquisition function

        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        # X =np.atleast_2d(X)
        if parallel and X.shape[0] > 1:
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
        if self.normalize and X.shape[0] > 1:
            sorted_acqX = np.sort(acqX, axis=None)
            self.acq_mean = np.mean(sorted_acqX[-10:])
            self.acq_std = 1e-1 * np.std(sorted_acqX[-10:])
            print('acq mean and std changed')
            print(self.acq_std)
        acqX = (acqX - self.acq_mean) / self.acq_std
        return acqX

    def _marginal_acq(self, X, utility_params_samples):
        """
        """
        marginal_acqX = np.zeros((X.shape[0], len(utility_params_samples)))
        n_z = len(self.Z_samples)

        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            varX = self.model.posterior_variance(X)
            for i in range(0, len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                for l in range(0, len(utility_params_samples)):
                    aux = np.multiply(np.square(utility_params_samples[l]), np.reciprocal(
                        varX[:, i]))  # Precompute this quantity for computational efficiency.
                    for Z in self.Z_samples:
                        # inner function of maKG acquisition function.
                        def inner_func(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            func_val = np.reshape(a + b * Z, (len(X_inner), 1))
                            return -func_val

                        # inner function of maKG acquisition function with its gradient.
                        def inner_func_with_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner, x)
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            da_dX_inner = np.tensordot(utility_params_samples[l], dmu_dX_inner, axes=1)
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            for k in range(X_inner.shape[1]):
                                dcov_dX_inner[:, :, k] = np.multiply(cov, dcov_dX_inner[:, :, k])
                            db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                            func_val = np.reshape(a + b * Z, (len(X_inner), 1))
                            func_gradient = np.reshape(da_dX_inner + db_dX_inner * Z, X_inner.shape)
                            return -func_val, -func_gradient

                        marginal_acqX[i, l] -= \
                        self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)[1]

        marginal_acqX = marginal_acqX / (self.n_hyps_samples * n_z)
        return marginal_acqX

    def _marginal_acq_parallel(self, X):
        """
        """
        n_x = len(X)
        marginal_acqX = np.zeros((n_x, len(self.utility_params_samples)))
        n_z = len(self.Z_samples)
        args = [[0 for i in range(2)] for j in range(n_x)]
        for i in range(n_x):
            args[i][0] = np.atleast_2d(X[i])
        pool = Pool(4)
        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            varX = self.model.posterior_variance(X)
            for i in range(n_x):
                args[i][1] = varX[:, i]
            marginal_acqX += np.atleast_2d(pool.map(self._parallel_acq_helper, args))

        marginal_acqX = marginal_acqX / (self.n_hyps_samples * n_z)
        return marginal_acqX

    def _parallel_acq_helper(self, args):
        """
        """
        #
        x = args[0]
        varx = args[1]
        utility_params_samples = self.utility_params_samples
        #
        L = len(utility_params_samples)
        marginal_acqx = np.zeros(L)
        self.model.partial_precomputation_for_covariance(x)
        self.model.partial_precomputation_for_covariance_gradient(x)
        for l in range(L):
            aux = np.multiply(np.square(utility_params_samples[l]),
                              np.reciprocal(varx))  # Precompute this quantity for computational efficiency.
            for Z in self.Z_samples:
                # inner function of maKG acquisition function.
                def inner_func(X_inner):
                    X_inner = np.atleast_2d(X_inner)
                    muX_inner = self.model.posterior_mean(X_inner)
                    cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :, 0]
                    a = np.matmul(utility_params_samples[l], muX_inner)
                    # a = support[t]*muX_inner
                    b = np.sqrt(np.matmul(aux, np.square(cov)))
                    func_val = np.reshape(a + b * Z, (len(X_inner), 1))
                    return -func_val

                # inner function of maKG acquisition function with its gradient.
                def inner_func_with_gradient(X_inner):
                    X_inner = np.atleast_2d(X_inner)
                    muX_inner = self.model.posterior_mean(X_inner)
                    dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                    cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :, 0]
                    dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner, x)
                    a = np.matmul(utility_params_samples[l], muX_inner)
                    # a = support[t]*muX_inner
                    da_dX_inner = np.tensordot(utility_params_samples[l], dmu_dX_inner, axes=1)
                    b = np.sqrt(np.matmul(aux, np.square(cov)))
                    for k in range(X_inner.shape[1]):
                        dcov_dX_inner[:, :, k] = np.multiply(cov, dcov_dX_inner[:, :, k])
                    db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                    db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                    func_val = np.reshape(a + b * Z, (len(X_inner), 1))
                    func_gradient = np.reshape(da_dX_inner + db_dX_inner * Z, X_inner.shape)
                    return -func_val, -func_gradient

                marginal_acqx[l] -= self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)[1]

        return marginal_acqx

    def _compute_acq_withGradients(self, X):
        """
        """
        X = np.atleast_2d(X)
        # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples,
        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X, self.utility_params_samples)
        if self.use_full_support:
            acqX = np.matmul(marginal_acqX, self.utility_prob_dist)
            dacq_dX = np.tensordot(marginal_dacq_dX, self.utility_prob_dist, 1)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(self.utility_params_samples)
            dacq_dX = np.sum(marginal_dacq_dX, axis=2) / len(self.utility_params_samples)
        acqX = np.reshape(acqX, (X.shape[0], 1))
        dacq_dX = np.reshape(dacq_dX, X.shape)
        acqX = (acqX - self.acq_mean) / self.acq_std
        dacq_dX /= self.acq_std
        return acqX, dacq_dX

    def _marginal_acq_with_gradient(self, X, utility_params_samples):
        """
        """
        marginal_acqX = np.zeros((X.shape[0], len(utility_params_samples)))
        marginal_dacq_dX = np.zeros((X.shape[0], X.shape[1], len(utility_params_samples)))
        Z_samples2 = np.random.normal(size=1)
        n_z = len(Z_samples2)
        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            varX = self.model.posterior_variance(X)
            dvar_dX = self.model.posterior_variance_gradient(X)
            for i in range(len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                for l in range(len(utility_params_samples)):
                    # Precompute aux1 and aux2 for computational efficiency.
                    aux = np.multiply(np.square(utility_params_samples[l]), np.reciprocal(varX[:, i]))
                    aux2 = np.multiply(np.square(utility_params_samples[l]), np.square(np.reciprocal(varX[:, i])))
                    for Z in Z_samples2:  # self.Z_samples:
                        # inner function of maKG acquisition function.
                        def inner_func(X_inner):
                            # X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)  # self.model.predict(X_inner)[0]
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = utility_params_samplesl]*muX_inner
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            func_val = np.reshape(a + b * Z, (X_inner.shape[0], 1))
                            return -func_val

                        # inner function of maKG acquisition function with its gradient.
                        def inner_func_with_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)  # Necessary
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner, x)
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = utility_params_samples[l]*muX_inner
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            da_dX_inner = np.tensordot(utility_params_samples[l], dmu_dX_inner, axes=1)
                            for k in range(X_inner.shape[1]):
                                dcov_dX_inner[:, :, k] = np.multiply(cov, dcov_dX_inner[:, :, k])
                            db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                            func_val = np.reshape(a + b * Z, (X_inner.shape[0], 1))
                            func_gradient = np.reshape(da_dX_inner + db_dX_inner * Z, X_inner.shape)
                            return -func_val, -func_gradient

                        x_opt, opt_val = self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)
                        marginal_acqX[i, l] -= opt_val
                        # x_opt = np.atleast_2d(x_opt)
                        cov_opt = self.model.posterior_covariance_between_points_partially_precomputed(x_opt, x)[:, 0,
                                  0]
                        dcov_opt_dx = self.model.posterior_covariance_gradient(x, x_opt)[:, 0, :]
                        b = np.sqrt(np.dot(aux, np.square(cov_opt)))
                        marginal_dacq_dX[i, :, l] += 0.5 * Z * np.reciprocal(b) * np.matmul(aux2, (
                                    2 * np.multiply(varX[:, i] * cov_opt, dcov_opt_dx.T) - np.multiply(
                                np.square(cov_opt), dvar_dX[:, i, :].T)).T)

        marginal_acqX = marginal_acqX / (self.n_hyps_samples * n_z)
        marginal_dacq_dX = marginal_dacq_dX / (self.n_hyps_samples * n_z)
        return marginal_acqX, marginal_dacq_dX

    def update_Z_samples(self):
        print('Update utility parameter and Z samples')
        self.Z_samples = np.random.normal(size=len(self.Z_samples))
        if not self.use_full_support:
            self.utility_params_samples = self.utility.parameter_dist.sample(len(self.utility_params_samples))