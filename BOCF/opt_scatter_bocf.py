"""Optimize Mie scattering spectrum of a multilayered spherical nanoparticle using BO with composite functions code."""
import argparse
import time
import os
from scattering import calc_spectrum
import numpy as np
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maEI import maEI
from parameter_distribution import ParameterDistribution
from utility import Utility
import cbo as cbo
from uEI_noiseless import uEI_noiseless


def get_obj(params, objective='narrow'):
    if objective == 'narrow':
        # Maximize the min scattering in 600-640nm range, minimize the max scattering outside of that
        lam = params.lam
        i1, = np.where(lam == 600)
        i2, = np.where(lam == 640)  # non-inclusive
        i1 = i1[0]
        i2 = i2[0]

        def obj_fun(y):
            return np.sum(y[:, i1:i2], axis=1) / np.sum(np.delete(y, np.arange(i1, i2), axis=1), axis=1)
    elif objective == 'hipass':
        lam = params.lam
        i1, = np.where(lam == 600)
        i1 = i1[0]

        def obj_fun(y):
            return np.sum(y[:, i1:], axis=1) / np.sum(y[:, :i1], axis=1)
    else:
        raise ValueError("Could not find an objective function with that name.")
    return obj_fun


def get_d_obj(params, objective='narrow'):
    if objective == 'narrow':
        # Maximize the min scattering in 600-640nm range, minimize the max scattering outside of that
        lam = params.lam
        i1, = np.where(lam == 600)
        i2, = np.where(lam == 640)  # non-inclusive
        i1 = i1[0]
        i2 = i2[0]

        def obj_fun(y):
            len1 = i1
            len2 = y.shape[1] - i2
            denominator = np.sum(np.delete(y, np.arange(i1, i2), axis=1), axis=1)   # shape (n)
            grad1 = np.ones(y[:, i1:i2].shape) / denominator[:, np.newaxis]
            grad2 = -np.sum(y[:, i1:i2], axis=1) / denominator**2
            grad2 = grad2[:, np.newaxis]
            return np.hstack((np.repeat(grad2, len1, axis=1), grad1, np.repeat(grad2, len2, axis=1)))
    elif objective == 'hipass':
        lam = params.lam
        i1, = np.where(lam == 600)
        i1 = i1[0]

        def obj_fun(y):
            denominator = np.sum(y[:, :i1], axis=1)
            grad1 = np.ones(y[:, i1:].shape) / denominator[:, np.newaxis]
            grad2 = -np.sum(y[:, i1:], axis=1) / denominator**2
            grad2 = grad2[:, np.newaxis]
            return np.hstack((np.repeat(grad2, i1, axis=1), grad1))
    else:
        raise ValueError("Could not find an objective function with that name.")
    return obj_fun


def get_problem(params, objective="narrow"):
    """Get objective function and problem parameters"""

    # Different objective functions to maximize during optimization
    if objective == "narrow" or objective == 'hipass':
        def prob(x):
            return params.calc_data(x)

    else:
        raise ValueError("No objective function specified.")

    return prob


def get_trial_dir(dir_format, i0=0):
    i = i0
    while True:
        results_dir_i = dir_format % i
        if os.path.isdir(results_dir_i):
            i += 1
        else:
            try:
                os.makedirs(results_dir_i)
                break
            except FileExistsError:
                pass
    return results_dir_i, i


def main(results_dir, objective="narrow", x_dim=6, trials=1, trial_i=0,):
    results_dir_i, _ = get_trial_dir(os.path.join(results_dir, 'trial%d'), trial_i)

    params = calc_spectrum.MieScattering(n_layers=x_dim)

    prob_fun = get_problem(params, objective=objective)
    obj_fun = get_obj(params, objective=objective)
    d_obj_fun = get_d_obj(params, objective=objective)

    time_list = []
    x_list = []
    y_list = []

    def g(X):
        z = prob_fun(X)

        # The rest is for logging
        y = obj_fun(z)
        x_list.append(X)
        y_list.append(y)
        time_list.append(time.time())
        np.savez(os.path.join(results_dir_i, 'info'), time_list=time_list, x_list=x_list, y_list=y_list)
        return np.transpose(z)
    m = 201

    # --- Objective
    objective = MultiObjective(g, as_list=False, output_dim=m)

    # --- Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (params.th_min, params.th_max),
                                        'dimensionality': params.n_layers}])

    # --- Model (Multi-output GP)
    n_attributes = m
    model = multi_outputGP(output_dim=n_attributes, exact_feval=[True] * m, fixed_hyps=False)

    # --- Initial design
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 5)

    # --- Parameter distribution
    parameter_support = np.ones((1,))
    parameter_dist = np.ones((1,))
    parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support,
                                                   prob_dist=parameter_dist)

    # --- Utility function
    def U_func(parameter, y):
        # print(np.shape(y))
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y2 = obj_fun(np.transpose(y))[0]
        return y2

    def dU_func(parameter, y):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        return d_obj_fun(np.transpose(y))[0, :]

    U = Utility(func=U_func, dfunc=dU_func, parameter_dist=parameter_distribution, linear=False)

    # # --- Compute real optimum value
    # bounds = [(params.th_min, params.th_max)] * 6
    # starting_points = np.random.rand(100, 6)
    # opt_val = 0
    # parameter = parameter_support[0]
    #
    # def func(x):
    #     x_copy = np.atleast_2d(x)
    #     fx = g(x_copy)
    #     # print('test begin')
    #     # print(parameter)
    #     # print(fx)
    #     val = U_func(parameter, fx)
    #     return -val
    #
    # best_val_found = np.inf
    #
    # for x0 in starting_points:
    #     res = scipy.optimize.fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=bounds)
    #     if best_val_found > res[1]:
    #         best_val_found = res[1]
    #         x_opt = res[0]
    # print('optimum')
    # print(x_opt)
    # print('optimal value')
    # print(-best_val_found)

    # --- Acquisition optimizer
    acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs2', inner_optimizer='lbfgs2', space=space)

    # --- Acquisition function
    # acquisition = maEI(model, space, optimizer=acq_opt, utility=U)
    # acquisition = uKG_cf(model, space, optimizer=acq_opt, utility=U, expectation_utility=expectation_U)
    acquisition = uEI_noiseless(model, space, optimizer=acq_opt, utility=U)

    # --- Evaluator
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    # --- Run CBO algorithm

    time1 = time.time()
    max_iter = 50
    for i in range(1):
        filename = os.path.join(results_dir_i, 'report.txt')
        bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design)
        bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)
        # bo_model.save_evaluations(os.path.join(results_dir, 'eval'))
        bo_model.save_results(os.path.join(results_dir_i, 'results'))
        np.savez(os.path.join(results_dir_i, 'info'), time_list=time_list, x_list=x_list, y_list=y_list)
    time2 = time.time()
    print('Total time %f' % (time2 - time1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize nanoparticle scattering")
    parser.add_argument("--results-dir", type=str, default='results/opt/test')
    # Optimization
    parser.add_argument("--objective", type=str, default="narrow")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument('--trial-i', type=int, default=0)

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        try:
            os.makedirs(kwargs['results_dir'])
        except FileExistsError:
            pass
    meta = open(os.path.join(kwargs['results_dir'], 'meta.txt'), 'a')
    import json
    meta.write(json.dumps(kwargs))
    meta.close()

    main(**kwargs)
