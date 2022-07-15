# Copyright (c) 2018, Raul Astudillo

import numpy as np
import GPyOpt
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Process


class multi_outputGP(object):
    """
    General class for handling a multi-output Gaussian proces based on GPyOpt.

    :param output_dim: number of outputs.
    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """
    analytical_gradient_prediction = True

    def __init__(self, output_dim, kernel=None, noise_var=None, exact_feval=None, n_samples=5, ARD=None, fixed_hyps=False):
        
        self.output_dim = output_dim
        
        if kernel is None:
            self.kernel = [None]*output_dim
        else:
            self.kernel = kernel
            
        if noise_var is None:
            self.noise_var = [None]*output_dim
        else:
            self.noise_var = noise_var
            
        if exact_feval is None:
            self.exact_feval = [False]*output_dim
        else:
            self.exact_feval = exact_feval
            
        self.n_samples = n_samples
            
        if ARD is None:
            self.ARD = [False]*output_dim
        else:
            self.ARD = ARD

        self.fixed_hyps =fixed_hyps
        self.output = [None]*output_dim
        for j in range(output_dim):
            if self.fixed_hyps:
                self.output[j] = GPyOpt.models.GPModelFixedHyps(kernel=self.kernel[j],noise_var=self.noise_var[j],exact_feval=self.exact_feval[j], n_samples=self.n_samples, ARD=self.ARD[j],verbose=False)
            else:
                self.output[j] = GPyOpt.models.GPModel(kernel=self.kernel[j], noise_var=self.noise_var[j],
                                                                exact_feval=self.exact_feval[j],
                                                                n_samples=self.n_samples, ARD=self.ARD[j],
                                                                verbose=False)

    #@staticmethod
    #def fromConfig(config):
        #return multi_outputGP(**config)
        
    def updateModel2(self, X_all, Y_all):
        """
        Updates the model with new observations.
        """
        self.Y_all = Y_all
        for j in range(self.output_dim):
            self.X_all[j] = np.copy(X_all)  
        jobs = []   
        for j in range(self.output_dim):
            p = Process(target=self.updateModel_single_output, args=(j,))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        
        print(self.output[0].predict(X_all[0]))
    

        
    def updateModel3(self, X_all, Y_all):
        """
        Updates the model with new observations.
        """
        self.Y_all = Y_all
        for j in range(self.output_dim):
            self.X_all[j] = np.copy(X_all)
        pool = Pool(4)
        print(list(range(self.output_dim)))
        pool.starmap(self.updateModel_single_output, [(0,),(1,)])
        #pool.close()
        #pool.join()
        #print('finished!')    

    def updateModel(self, X_all, Y_all):
        """
        Updates the model with new observations.
        """
        for j in range(self.output_dim):
            self.output[j].updateModel(X_all,Y_all[j],None,None)
            
            
    def updateModel_single_output(self, index):
        self.output[index].updateModel(self.X_all[index],self.Y_all[index],None,None)
        
    
    def number_of_hyps_samples(self):
        return self.n_samples
                  
    
    def set_hyperparameters(self, n):
        for j in range(self.output_dim):
            self.output[j].set_hyperparameters(n)
            
    
    def set_hyperparameters2(self, hyperparameters):
        for j in range(self.output_dim):
            self.output[j].set_hyperparameters(hyperparameters[j])
            
            
    def get_hyperparameters_samples(self, n_samples=1):
        hyperparameters = [[None]*self.output_dim]*n_samples
        for j in range(self.output_dim):
            for i in range(n_samples):
                hyperparameters[i][j] = self.output[j].get_hyperparameters_samples(n_samples)[i] 
        return hyperparameters


    def get_evaluated_points(self):
        """
        Returns posterior mean at the points that have been already evaluated.
        """
        return np.copy(self.output[0].model.X)
            

    def predict(self,  X,  full_cov=False):
        """
        Predictions with the model. Returns posterior means and variance at X.
        """
        X = np.atleast_2d(X)
        m = np.empty((self.output_dim,X.shape[0]))
        cov = np.empty((self.output_dim,X.shape[0]))
        for j in range(self.output_dim):
            tmp1, tmp2= self.output[j].predict(X,full_cov)
            m[j,:] = tmp1[:,0]
            cov[j,:] = tmp2[:,0]
        return m, cov
    
    def predict_noiseless(self,  X,  full_cov=False):
        """
        Predictions with the model. Returns posterior means and variance at X.
        """
        X = np.atleast_2d(X)
        m = np.empty((self.output_dim,X.shape[0]))
        cov = np.empty((self.output_dim,X.shape[0]))
        for j in range(self.output_dim):
            tmp1, tmp2= self.output[j].predict_noiseless(X,full_cov)
            m[j,:] = tmp1[:,0]
            cov[j,:] = tmp2[:,0]
        return m, cov
    
    
    def posterior_mean(self,  X):
        """
        Predictions with the model. Returns posterior mean at X.
        """
        #X = np.atleast_2d(X)
        m = np.empty((self.output_dim,X.shape[0]))
        for j in range(self.output_dim):
          m[j,:]  = self.output[j].posterior_mean(X)[:,0]
        return m
    
    
    def posterior_mean_at_evaluated_points(self):
        """
        Returns posterior mean at the points that have been already evaluated.
        """
        return self.posterior_mean(self.output[0].model.X)
    
    
    def posterior_variance(self,  X):
        """
        Returns posterior variance at X.
        """
        #X = np.atleast_2d(X)
        var = np.empty((self.output_dim,X.shape[0]))
        for j in range(self.output_dim):
            var[j,:] = self.output[j].posterior_variance(X)[:,0]
        return var
  
    
    def posterior_variance_noiseless(self, X):
        """
        """
        var = np.empty((self.output_dim,X.shape[0]))
        for j in range(self.output_dim):
            var[j,:] = self.output[j].posterior_variance_noiseless(X)[:,0]
        return var
   
    
    def partial_precomputation_for_covariance(self, X):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        for j in range(self.output_dim):
            self.output[j].partial_precomputation_for_covariance(X)
            
    
    def partial_precomputation_for_covariance_gradient(self, x):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        for j in range(self.output_dim):
            self.output[j].partial_precomputation_for_covariance_gradient(x)
            
            
    def partial_precomputation_for_variance_conditioned_on_next_point(self, next_point):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        for j in range(self.output_dim):
            self.output[j].partial_precomputation_for_variance_conditioned_on_next_point(next_point)
        
        
    def posterior_variance_conditioned_on_next_point(self, X):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        var = np.empty((self.output_dim,X.shape[0]))
        for j in range(self.output_dim):
            var[j,:] = self.output[j].posterior_variance_conditioned_on_next_point(X)[:,0]
        return var
    
    
    def posterior_variance_gradient_conditioned_on_next_point(self, X):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        dvar_dX = np.empty((self.output_dim,X.shape[0],X.shape[1]))
        for j in range(0,self.output_dim):
            dvar_dX[j,:,:] = self.output[j].posterior_variance_gradient_conditioned_on_next_point(X)    
        return dvar_dX


    def posterior_covariance_between_points(self,  X1,  X2):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        cov = np.empty((self.output_dim,X1.shape[0],X2.shape[0]))
        for j in range(0,self.output_dim):
            cov[j,:,:] = self.output[j].posterior_covariance_between_points(X1, X2)
        return cov
    
    
    def posterior_covariance_between_points_partially_precomputed(self, X1, X2):
        """
        Computes the posterior covariance between points.

        :param kern: GP kernel
        :param X: current input observations
        :param X1: some input observations
        :param X2: other input observations
        """
        cov = np.empty((self.output_dim,X1.shape[0],X2.shape[0]))
        for j in range(0,self.output_dim):
            cov[j,:,:] = self.output[j].posterior_covariance_between_points_partially_precomputed(X1, X2)
        return cov
    
    
    def posterior_mean_gradient(self,  X):
        """
        Computes dmu/dX(X).
        :param X:  input observations
        """
        dmu_dX = np.empty((self.output_dim,X.shape[0],X.shape[1]))
        for j in range(0,self.output_dim):
            tmp = self.output[j].posterior_mean_gradient(X)
            dmu_dX[j,:,:] = tmp

        return dmu_dX
    
    
    def posterior_variance_gradient(self,  X):
        """
        Computes dmu/dX(X).
        :param X:  input observations
        """
        dvar_dX = np.empty((self.output_dim,X.shape[0],X.shape[1]))
        for j in range(0,self.output_dim):
            dvar_dX[j,:,:] = self.output[j].posterior_variance_gradient(X)
            
        return dvar_dX
    
    
    def posterior_covariance_gradient(self, X, x2):
        """
        Computes dK/dX(X,x2).
        :param X: input obersevations.
        :param x2:  input observation.
        """
        dK_dX = np.empty((self.output_dim,X.shape[0],X.shape[1]))
        for j in range(0,self.output_dim):
            dK_dX[j,:,:] = self.output[j].posterior_covariance_gradient(X, x2)
        return dK_dX
    
    
    def posterior_covariance_gradient_partially_precomputed(self, X, x2):
        """
        Computes dK/dX(X,x2).
        :param X: input obersevations.
        :param x2:  input observation.
        """
        dK_dX = np.empty((self.output_dim,X.shape[0],X.shape[1]))
        for j in range(self.output_dim):
            dK_dX[j,:,:] = self.output[j].posterior_covariance_gradient_partially_precomputed(X, x2)
        return dK_dX
    
    
    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        model_parameters = [None]*self.output_dim
        for j in range(0,self.output_dim):
            model_parameters[j] = self.output[j].get_model_parameters()
            

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        model_parameters_names = [None]*self.output_dim
        for j in range(0,self.output_dim):
            model_parameters_names[j] = self.output[j].get_model_parameters_names()


class multi_outputGP_fixed_hyps(object):
    """
    General class for handling a multi-output Gaussian proces based on GPyOpt.

    :param output_dim: number of outputs.
    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """
    analytical_gradient_prediction = True

    def __init__(self, output_dim, kernel=None, noise_var=None, ARD=None, kernel_hyps=None):

        self.output_dim = output_dim

        if kernel is None:
            self.kernel = [None] * output_dim
        else:
            self.kernel = kernel

        if noise_var is None:
            self.noise_var = [None] * output_dim
        else:
            self.noise_var = noise_var

        if ARD is None:
            self.ARD = [False] * output_dim
        else:
            self.ARD = ARD

        self.kernel_hyps = kernel_hyps

        self.output = [None] * output_dim
        for j in range(output_dim):
            self.output[j] = GPyOpt.models.GPModelFixedHyps(kernel=self.kernel[j], noise_var=self.noise_var[j], ARD=self.ARD[j], kernel_hyps=self.kernel_hyps[j])

    # @staticmethod
    # def fromConfig(config):
    # return multi_outputGP(**config)

    def updateModel2(self, X_all, Y_all):
        """
        Updates the model with new observations.
        """
        self.Y_all = Y_all
        for j in range(self.output_dim):
            self.X_all[j] = np.copy(X_all)
        jobs = []
        for j in range(self.output_dim):
            p = Process(target=self.updateModel_single_output, args=(j,))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()

        print(self.output[0].predict(X_all[0]))

    def updateModel3(self, X_all, Y_all):
        """
        Updates the model with new observations.
        """
        self.Y_all = Y_all
        for j in range(self.output_dim):
            self.X_all[j] = np.copy(X_all)
        pool = Pool(4)
        print(list(range(self.output_dim)))
        pool.starmap(self.updateModel_single_output, [(0,), (1,)])
        # pool.close()
        # pool.join()
        # print('finished!')

    def updateModel(self, X_all, Y_all):
        """
        Updates the model with new observations.
        """
        for j in range(self.output_dim):
            self.output[j].updateModel(X_all, Y_all[j], None, None)

    def updateModel_single_output(self, index):
        self.output[index].updateModel(self.X_all[index], self.Y_all[index], None, None)

    def number_of_hyps_samples(self):
        return self.n_samples

    def set_hyperparameters(self, n):
        for j in range(self.output_dim):
            self.output[j].set_hyperparameters(n)

    def set_hyperparameters2(self, hyperparameters):
        for j in range(self.output_dim):
            self.output[j].set_hyperparameters(hyperparameters[j])

    def get_hyperparameters_samples(self, n_samples=1):
        hyperparameters = [[None] * self.output_dim] * n_samples
        for j in range(self.output_dim):
            for i in range(n_samples):
                hyperparameters[i][j] = self.output[j].get_hyperparameters_samples(n_samples)[i]
        return hyperparameters

    def predict(self, X, full_cov=False):
        """
        Predictions with the model. Returns posterior means and variance at X.
        """
        X = np.atleast_2d(X)
        m = np.empty((self.output_dim, X.shape[0]))
        cov = np.empty((self.output_dim, X.shape[0]))
        for j in range(self.output_dim):
            tmp1, tmp2 = self.output[j].predict(X, full_cov)
            m[j, :] = tmp1[:, 0]
            cov[j, :] = tmp2[:, 0]
        return m, cov

    def predict_noiseless(self, X, full_cov=False):
        """
        Predictions with the model. Returns posterior means and variance at X.
        """
        X = np.atleast_2d(X)
        m = np.empty((self.output_dim, X.shape[0]))
        cov = np.empty((self.output_dim, X.shape[0]))
        for j in range(self.output_dim):
            tmp1, tmp2 = self.output[j].predict_noiseless(X, full_cov)
            m[j, :] = tmp1[:, 0]
            cov[j, :] = tmp2[:, 0]
        return m, cov

    def posterior_mean(self, X):
        """
        Predictions with the model. Returns posterior mean at X.
        """
        # X = np.atleast_2d(X)
        m = np.empty((self.output_dim, X.shape[0]))
        for j in range(self.output_dim):
            m[j, :] = self.output[j].posterior_mean(X)[:, 0]
        return m

    def posterior_mean_at_evaluated_points(self):
        """
        Returns posterior mean at the points that have been already evaluated.
        """
        return self.posterior_mean(self.output[0].model.X)

    def posterior_variance(self, X):
        """
        Returns posterior variance at X.
        """
        # X = np.atleast_2d(X)
        var = np.empty((self.output_dim, X.shape[0]))
        for j in range(self.output_dim):
            var[j, :] = self.output[j].posterior_variance(X)[:, 0]
        return var

    def posterior_variance_noiseless(self, X):
        """
        """
        var = np.empty((self.output_dim, X.shape[0]))
        for j in range(self.output_dim):
            var[j, :] = self.output[j].posterior_variance_noiseless(X)[:, 0]
        return var

    def partial_precomputation_for_covariance(self, X):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        for j in range(self.output_dim):
            self.output[j].partial_precomputation_for_covariance(X)

    def partial_precomputation_for_covariance_gradient(self, x):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        for j in range(self.output_dim):
            self.output[j].partial_precomputation_for_covariance_gradient(x)

    def partial_precomputation_for_variance_conditioned_on_next_point(self, next_point):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        for j in range(self.output_dim):
            self.output[j].partial_precomputation_for_variance_conditioned_on_next_point(next_point)

    def posterior_variance_conditioned_on_next_point(self, X):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        var = np.empty((self.output_dim, X.shape[0]))
        for j in range(self.output_dim):
            var[j, :] = self.output[j].posterior_variance_conditioned_on_next_point(X)[:, 0]
        return var

    def posterior_variance_gradient_conditioned_on_next_point(self, X):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        dvar_dX = np.empty((self.output_dim, X.shape[0], X.shape[1]))
        for j in range(0, self.output_dim):
            dvar_dX[j, :, :] = self.output[j].posterior_variance_gradient_conditioned_on_next_point(X)
        return dvar_dX

    def posterior_covariance_between_points(self, X1, X2):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        cov = np.empty((self.output_dim, X1.shape[0], X2.shape[0]))
        for j in range(0, self.output_dim):
            cov[j, :, :] = self.output[j].posterior_covariance_between_points(X1, X2)
        return cov

    def posterior_covariance_between_points_partially_precomputed(self, X1, X2):
        """
        Computes the posterior covariance between points.

        :param kern: GP kernel
        :param X: current input observations
        :param X1: some input observations
        :param X2: other input observations
        """
        cov = np.empty((self.output_dim, X1.shape[0], X2.shape[0]))
        for j in range(0, self.output_dim):
            cov[j, :, :] = self.output[j].posterior_covariance_between_points_partially_precomputed(X1, X2)
        return cov

    def posterior_mean_gradient(self, X):
        """
        Computes dmu/dX(X).
        :param X:  input observations
        """
        dmu_dX = np.empty((self.output_dim, X.shape[0], X.shape[1]))
        for j in range(0, self.output_dim):
            tmp = self.output[j].posterior_mean_gradient(X)
            dmu_dX[j, :, :] = tmp

        return dmu_dX

    def posterior_variance_gradient(self, X):
        """
        Computes dmu/dX(X).
        :param X:  input observations
        """
        dvar_dX = np.empty((self.output_dim, X.shape[0], X.shape[1]))
        for j in range(0, self.output_dim):
            dvar_dX[j, :, :] = self.output[j].posterior_variance_gradient(X)

        return dvar_dX

    def posterior_covariance_gradient(self, X, x2):
        """
        Computes dK/dX(X,x2).
        :param X: input obersevations.
        :param x2:  input observation.
        """
        dK_dX = np.empty((self.output_dim, X.shape[0], X.shape[1]))
        for j in range(0, self.output_dim):
            dK_dX[j, :, :] = self.output[j].posterior_covariance_gradient(X, x2)
        return dK_dX

    def posterior_covariance_gradient_partially_precomputed(self, X, x2):
        """
        Computes dK/dX(X,x2).
        :param X: input obersevations.
        :param x2:  input observation.
        """
        dK_dX = np.empty((self.output_dim, X.shape[0], X.shape[1]))
        for j in range(self.output_dim):
            dK_dX[j, :, :] = self.output[j].posterior_covariance_gradient_partially_precomputed(X, x2)
        return dK_dX

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        model_parameters = [None] * self.output_dim
        for j in range(0, self.output_dim):
            model_parameters[j] = self.output[j].get_model_parameters()

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        model_parameters_names = [None] * self.output_dim
        for j in range(0, self.output_dim):
            model_parameters_names[j] = self.output[j].get_model_parameters_names()