import numpy as np
import scipy
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from uKG_SGA import uKG_SGA
from uKG_cf import uKG_cf
from uEI_noiseless import uEI_noiseless
from parameter_distribution import ParameterDistribution
from utility import Utility
from expectation_utility import ExpectationUtility
import cbo
import sys
import time

# --- Function to optimize
m = 5  # Number of attributes
aux_model = []
I = np.linspace(0., 1., 6)
aux_grid = np.meshgrid(I, I, I, I)
grid = np.array([a.flatten() for a in aux_grid]).T
kernel = GPy.kern.SE(input_dim=4, variance=2., lengthscale=0.3)
cov = kernel.K(grid)
mean = np.zeros((6 ** 4,))
for j in range(m):
    r = np.random.RandomState(j+7)
    Y = r.multivariate_normal(mean, cov)
    Y = np.reshape(Y, (6 ** 4, 1))
    print(Y[:5, 0])
    aux_model.append(GPy.models.GPRegression(grid, Y, kernel, noise_var=1e-10))


def f(X):
    X = np.atleast_2d(X)
    fX = np.empty((m, X.shape[0]))
    for j in range(m):
        fX[j, :] = aux_model[j].posterior_mean(X)[:, 0]
    return fX


#noise_var = [0.25]*m
objective = MultiObjective(f, as_list=False, output_dim=m)
#objective = MultiObjective(f, noise_var=noise_var, as_list=False, output_dim=m)

# --- Space
space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 4}])

# --- Model (Multi-output GP)
n_attributes = m
model = multi_outputGP(output_dim=n_attributes, exact_feval=[False] * m, fixed_hyps=False)
# model = multi_outputGP(output_dim=n_attributes, noise_var=noise_var, fixed_hyps=True)

# --- Aquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs2', inner_optimizer='lbfgs2', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 10)


# --- Parameter distribution
bounds = [(0, 1)] * 4
starting_points = np.random.rand(100, 4)
#parameter_support = np.empty((1,m))
for j in range(1):
    def marginal_func(x):
        x_copy = np.atleast_2d(x)
        val = aux_model[j].posterior_mean(x_copy)[:, 0]
        return -val

    best_val_found = np.inf
    for x0 in starting_points:
        res = scipy.optimize.fmin_l_bfgs_b(marginal_func, x0, approx_grad=True, bounds=bounds)
        if best_val_found > res[1]:
            # print(res)
            best_val_found = res[1]
            marginal_opt = res[0]
    parameter_support = f(marginal_opt).transpose()


#parameter_support = f(x_opt).T #+ r.normal(scale=1., size=(6, 3))
parameter_dist = np.ones((1,)) / 1
parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support, prob_dist=parameter_dist)



# --- Utility function
def U_func(parameter, y):
    #y_aux = np.squeeze(y)
    aux = (y.transpose() - parameter).transpose()
    return -np.sum(np.square(aux), axis=0)

def dU_func(parameter, y):
    y_aux = np.squeeze(y)
    return -2*(y_aux - parameter)

U = Utility(func=U_func, dfunc=dU_func, parameter_dist=parameter_distribution, linear=False)

# --- Expectation of utility
def psi(parameter,mu, var):
    #mu_aux = np.squeeze(mu)
    #var_aux = np.squeeze(var)
    aux = (mu.transpose() - parameter).transpose()
    val = -np.sum(np.square(aux), axis=0) - np.sum(var, axis=0)
    return val

def psi_gradient(parameter,mu,var):
    mu_aux = np.squeeze(mu)
    var_aux = np.squeeze(var)
    gradient = -np.concatenate((2*(mu_aux - parameter), np.ones((len(var_aux),))))
    return gradient

expectation_U = ExpectationUtility(psi, psi_gradient)

# --- Compute real optimum value
if True:
    bounds = [(0, 1)] * 4
    starting_points = np.random.rand(100, 4)
    opt_val = 0
    parameter_samples = parameter_support
    for parameter in parameter_samples:
        def marginal_func(x):
            x_copy = np.atleast_2d(x)
            fx = f(x_copy)
            # print('test begin')
            # print(parameter)
            # print(fx)
            val = U_func(parameter, fx)
            return -val


        best_val_found = np.inf
        for x0 in starting_points:
            res = scipy.optimize.fmin_l_bfgs_b(marginal_func, x0, approx_grad=True, bounds=bounds)
            if best_val_found > res[1]:
                # print(res)
                best_val_found = res[1]
                marginal_opt = res[0]
        print('marginal opt')
        print(parameter)
        print(marginal_opt)
        print(f(marginal_opt))
        print(-best_val_found)
        opt_val -= best_val_found
    opt_val /= len(parameter_samples)
    print('real optimum')
    print(opt_val)

# --- Aquisition function
acquisition = uEI_noiseless(model, space, optimizer=acq_opt, utility=U)
#acquisition = uKG_cf(model, space, optimizer=acq_opt, utility=U, expectation_utility=expectation_U)
# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# standard BO

max_iter = 50
for i in range(1):
    filename = './experiments/test1_EIh_noisy_' + str(i) + '.txt'
    bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design, expectation_utility=expectation_U)
    bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)