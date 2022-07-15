"""Optimize photonic crystal using the BO of composite functions code"""
import argparse
import time
import os
import numpy as np
import sys
from pc import level_set, DOS_GGR
import GPyOpt
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maEI import maEI
from parameter_distribution import ParameterDistribution
from utility import Utility
import cbo as cbo
from uEI_noiseless import uEI_noiseless

xdim=51


def get_obj(objective='dos10'):
    if objective == 'dos':
        def obj_fun(dos):
            dosmin = np.sum(dos[:, 300:350], axis=1)
            dosmax = np.sum(dos[:, :300], axis=1) + np.sum(dos[:, 350:], axis=1)
            obj = dosmax / (dosmin + 1)     # Avoid dividing by small numbers
            return obj
    else:
        raise ValueError("Could not find an objective function with that name.")
    return obj_fun


def get_d_obj(objective='dos10'):
    if objective == 'dos':
        def obj_fun(dos):
            grad1 = 1 / (1 + np.sum(dos[:, 300:350], axis=1))
            dosmax = np.sum(dos[:, :300], axis=1) + np.sum(dos[:, 350:], axis=1)
            grad2 = -dosmax / (1 + np.sum(dos[:, 300:350], axis=1))**2
            grad1 = grad1[:, np.newaxis]
            grad2 = grad2[:, np.newaxis]

            return np.hstack((np.repeat(grad1, 300, axis=1),
                              np.repeat(grad2, 50, axis=1),
                              np.repeat(grad1, 150, axis=1)))
    else:
        raise ValueError("Could not find an objective function with that name.")
    return obj_fun


def get_problem(leveller=None, objective="dos", sample_full=True):
    """Get objective function and problem parameters"""

    if objective == "dos":
        def obj_fun(x):
            sys.stdout = open(os.devnull, 'w')
            eps_arr = leveller.calc_data(x, sample_level=True, sample_full=sample_full)
            dos_arr = []
            for eps_i in eps_arr:
                _, dos = DOS_GGR.main(eps_i, Nk=10)

                dos_arr.append(dos)
            sys.stdout = sys.__stdout__

            return np.array(dos_arr)
    else:
        raise ValueError("Could not find an objective function with that name.")

    return obj_fun


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


def main(results_dir,  sample_full=True, objective="dos", trials=1, trial_i=0):
    results_dir_i, _ = get_trial_dir(os.path.join(results_dir, 'trial%d'), trial_i)

    leveller = level_set.FourierLevelSet(eps_in=1.0, eps_out=11.4)

    prob_fun = get_problem(leveller=leveller, objective=objective, sample_full=sample_full)
    obj_fun = get_obj(objective=objective)
    d_obj_fun = get_d_obj(objective=objective)

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
    m = 500

    # --- Objective
    objective = MultiObjective(g, as_list=False, output_dim=m)

    # --- Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1),
                                        'dimensionality': xdim}])

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
    max_iter = 100
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
    parser.add_argument("--n-start", type=int, default=5, help='Size of initial dataset')
    # Dataset
    parser.add_argument('--sample-full', dest='sample_full', action='store_true')
    parser.add_argument('--no-sample-full', dest='sample_full', action='store_false')
    # Optimization
    parser.add_argument("--objective", type=str, default="dos")
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
