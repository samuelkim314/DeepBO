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

bounds = [(0, 1)] * 4
starting_points = np.random.rand(100, 4)
y_opt = np.empty((1,m))
for j in range(1):
    def marginal_func(x):
        x_copy = np.atleast_2d(x)
        val = aux_model[j].posterior_mean(x_copy)[:, 0]
        return -val

    best_val_found = np.inf
    for x0 in starting_points:
        res = scipy.optimize.fmin_l_bfgs_b(marginal_func, x0, approx_grad=True, bounds=bounds)
        if best_val_found > res[1]:
            best_val_found = res[1]
            marginal_opt = res[0]
    y_opt = f(marginal_opt).transpose()

def g(y):
    #y_aux = np.squeeze(y)
    aux = (y.transpose() - y_opt).transpose()
    return -np.sum(np.square(aux), axis=0)

def objective(X):
    return g(f(X))

objective = MultiObjective([objective], as_list=True, output_dim=1)
# objective = MultiObjective(f, noise_var=noise_var)

# --- Space
space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 4}])

# --- Model (Multi-output GP)
model = multi_outputGP(output_dim=1, exact_feval=[True], fixed_hyps=False)
# model = multi_outputGP(output_dim=n_attributes, noise_var=noise_var, fixed_hyps=True)

# --- Aquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs2', inner_optimizer='lbfgs2', space=space)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 10)


# --- Parameter distribution
parameter_support = np.ones((1,1))
parameter_dist = np.ones((1,)) / 1
parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support, prob_dist=parameter_dist)

# --- Utility function
def U_func(parameter,y):
    return np.dot(parameter,y)

def dU_func(parameter,y):
    return parameter

U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=True)
# --- Aquisition function
acquisition = maEI(model, space, optimizer=acq_opt,utility=U)
# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
# standard BO

if True:
    bounds = [(0, 1)] * 4
    starting_points = np.random.rand(100, 4)
    def marginal_func(x):
        x_copy = np.atleast_2d(x)
        fx = f(x_copy)
        # print('test begin')
        # print(parameter)
        # print(fx)
        val = g(fx)
        return -val


    best_val_found = np.inf
    for x0 in starting_points:
        res = scipy.optimize.fmin_l_bfgs_b(marginal_func, x0, approx_grad=True, bounds=bounds)
        if best_val_found > res[1]:
            # print(res)
            best_val_found = res[1]
            opt = res[0]
    print('opt')
    print(opt)
    print(f(opt))
    print(y_opt)
    print('real optimum')
    print(-best_val_found)

max_iter = 50
for i in range(1):
    filename = './experiments/test1_EIf_' + str(i) + '.txt'
    bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design)
    bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)