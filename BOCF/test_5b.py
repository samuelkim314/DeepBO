import numpy as np
import scipy
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maEI import maEI
from parameter_distribution import ParameterDistribution
from utility import Utility
import cbo
from scipy.stats import norm

# --- Function to optimize
d = 3 # Input dimension
m = 4  # Number of attributes
aux_model = []
I = np.linspace(0., 1., 8)
aux_grid = np.meshgrid(I, I, I)
grid = np.array([a.flatten() for a in aux_grid]).T
kernel = GPy.kern.SE(input_dim=d, variance=2., lengthscale=0.3)
cov = kernel.K(grid)
mean = np.zeros((8 ** d,))
for j in range(m):
    r = np.random.RandomState(j+7)
    Y = r.multivariate_normal(mean, cov)
    Y = np.reshape(Y, (8 ** d, 1))
    print(Y[:5, 0])
    aux_model.append(GPy.models.GPRegression(grid, Y, kernel, noise_var=1e-10))


def h(X):
    X = np.atleast_2d(X)
    hX = np.empty((m, X.shape[0]))
    for j in range(m):
        hX[j, :] = aux_model[j].posterior_mean(X)[:, 0]
    return hX

parameter = np.atleast_1d([1.25, 1.50, 1.25, 1.50])

def g(y):
    return np.squeeze(np.sum(-np.exp(y), axis=0))

def f(X):
    return g(h(X))
# --- Objective
objective = MultiObjective([f], as_list=True, output_dim=1)

# --- Space
space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

# --- Model
model = multi_outputGP(output_dim=1, exact_feval=[True], fixed_hyps=False)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 2*(d+1))

# --- Parameter distribution
parameter_support = np.ones((1,1))
parameter_dist = np.ones((1,))
parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support, prob_dist=parameter_dist)

# --- Utility function
def U_func(parameter,y):
    return np.dot(parameter,y)

def dU_func(parameter,y):
    return parameter

U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=True)

# --- Acquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs2', inner_optimizer='lbfgs2', space=space)

# --- Acquisition function
acquisition = maEI(model, space, optimizer=acq_opt,utility=U)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# --- Compute real optimum value
bounds = [(0, 1)] * d
starting_points = np.random.rand(100, d)
opt_val = 0
#parameter = parameter_support[0,:]

def func(x):
    x_copy = np.atleast_2d(x)
    val = f(x_copy)
    return -val

best_val_found = np.inf

for x0 in starting_points:
    res = scipy.optimize.fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=bounds)
    if best_val_found > res[1]:
        best_val_found = res[1]
        x_opt = res[0]
print('optimum')
print(x_opt)
print('optimal value')
print(-best_val_found)

max_iter = 50
for i in range(1):
    filename = './experiments/test5_EIf_noiseless_' + str(i) + '.txt'
    bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design)
    bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)