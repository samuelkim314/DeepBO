"""Optimize nanoparticle scattering using the Neural Tangents library to implement infinite-width and infinite-ensemble
approximations of neural networks as GPs."""
import argparse
import time
from lib import acquisition as bo
from lib import data_manager
from lib.helpers import get_trial_dir
import os
import numpy as np
from scattering import calc_spectrum
from jax import random
from jax.experimental import optimizers
from jax.api import jit, grad, vmap
import neural_tangents as nt
from neural_tangents import stax
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"


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


def get_problem(params, objective="narrow"):
    """Get objective function and problem parameters"""

    # Different objective functions to maximize during optimization
    if objective == "narrow" or objective == 'hipass':
        def prob(x):
            return params.calc_data(x)

    else:
        raise ValueError("No objective function specified.")

    return prob


def main(results_dir,  n_batch, n_train=1000, opt="ensemble", objective="narrow", x_dim=6, trials=1, kernel='nngp',
         trial_i=0):
    params = calc_spectrum.MieScattering(n_layers=x_dim)
    prob_fun = get_problem(params, objective=objective)
    obj_fun = get_obj(params, objective=objective)

    if opt == "ensemble":
        # Bayesian optimization using infinite ensemble of infinite width networks trained for infinite time

        n_start = 5     # Size of initial dataset

        for _ in range(trials):
            # Setup initial dataset
            results_dir_i, _ = get_trial_dir(os.path.join(results_dir, 'trial%d'), i0=trial_i)
            X_train = params.sample_x(n_start)
            X_nn_train = params.to_nn_input(X_train)
            Y_train = obj_fun(prob_fun(X_train))[:, np.newaxis]
            dm = data_manager.DataManager(X_nn_train, Y_train, n_batch)

            parameterization = 'standard'

            # Define network architecture
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Dense(64, parameterization=parameterization), stax.Relu(),
                stax.Dense(128, parameterization=parameterization), stax.Relu(),
                stax.Dense(256, parameterization=parameterization), stax.Relu(),
                stax.Dense(512, parameterization=parameterization), stax.Relu(),
                stax.Dense(512, parameterization=parameterization), stax.Relu(),
                stax.Dense(512, parameterization=parameterization), stax.Relu(),
                stax.Dense(256, parameterization=parameterization), stax.Relu(),
                stax.Dense(128, parameterization=parameterization), stax.Relu(),
                stax.Dense(1)
            )
            kernel_fn = jit(kernel_fn, static_argnums=(2,))
            # nngp = kernel_fn(x1, x2, 'nngp')  # (10, 20) np.ndarray
            # ntk = kernel_fn(x1, x2, 'ntk')  # (10, 20) np.ndarray

            start_time = time.time()

            # Best candidate data point so far
            y_best_i = np.argmax(Y_train, axis=0)
            x_best = X_train[y_best_i, :]
            y_best = np.max(Y_train)

            n_data_arr = []
            y_best_list = []

            x_nn_new = None
            y_new = None

            for i in range(n_train):
                # Random set of unlabelled x points - we will use Bayesian optimization to choose which one to label
                x_sample = params.sample_x(int(1e3))
                x_nn_sample = params.to_nn_input(x_sample)

                # predict_fn = nt.predict.gp_inference(kernel_fn, X_nn_train, Y_train, x_nn_sample, 'ntk')
                predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_nn_train, Y_train, diag_reg=1e-4)

                def f(x):
                    mean, covariance = predict_fn(x_test=x, get=kernel, compute_cov=True)
                    return mean, np.sqrt(np.diag(covariance)[:, np.newaxis])

                # Pick out and label new data point
                i_new, x_nn_new = bo.ei_direct_batch(x_nn_sample, f, y_best)
                x_new = x_sample[i_new][np.newaxis, :]
                y_new = obj_fun(prob_fun(x_new))[:, np.newaxis]

                # Add the labelled data point to our training data set
                X_train = np.vstack((X_train, x_new))
                X_nn_train = np.vstack((X_nn_train, x_nn_new))
                Y_train = np.vstack((Y_train, y_new))
                dm.add_data(x_nn_new, y_new)

                # Update the best data point so far
                i_best = np.argmax(Y_train)
                x_best = X_train[i_best]
                x_nn_best = X_nn_train[i_best]
                y_best = Y_train[i_best]

                # Save results to a file
                n_data_arr.append(dm.n)
                y_best_list.append(y_best)
                print("Trained with %d data points. Best value=%f" % (dm.n, y_best))
                np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=y_best_list, best_x=x_best,
                         best_x_nn=x_nn_best)
            time_tot = time.time() - start_time
            print("Took %f seconds" % time_tot)
            np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=y_best_list,
                     time_tot=time_tot, best_x=x_best, best_x_nn=x_nn_best)
            print(x_best)
            print(y_best)
    elif opt == "single":
        # Bayesian optimization using single infinite width network trained for infinite time

        n_start = 5  # Size of initial dataset

        for _ in range(trials):
            results_dir_i, _ = get_trial_dir(os.path.join(results_dir, 'trial%d'), i0=trial_i)
            X_train = params.sample_x(n_start)
            X_nn_train = params.to_nn_input(X_train)
            Y_train = obj_fun(prob_fun(X_train))[:, np.newaxis]
            dm = data_manager.DataManager(X_nn_train, Y_train, n_batch)

            parameterization = 'standard'

            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Dense(64, parameterization=parameterization), stax.Relu(),
                stax.Dense(128, parameterization=parameterization), stax.Relu(),
                stax.Dense(256, parameterization=parameterization), stax.Relu(),
                stax.Dense(512, parameterization=parameterization), stax.Relu(),
                stax.Dense(512, parameterization=parameterization), stax.Relu(),
                stax.Dense(512, parameterization=parameterization), stax.Relu(),
                stax.Dense(256, parameterization=parameterization), stax.Relu(),
                stax.Dense(128, parameterization=parameterization), stax.Relu(),
                stax.Dense(1)
            )
            if kernel == 'nngp':
                kernel_fn = jit(kernel_fn, static_argnums=(2,))

            start_time = time.time()

            # Best candidate data point so far
            y_best_i = np.argmax(Y_train, axis=0)
            x_best = X_train[y_best_i, :]
            y_best = np.max(Y_train)

            n_data_arr = []
            y_best_list = []

            x_nn_new = None
            y_new = None

            for i in range(n_train):
                # Random set of unlabelled x points - we will use Bayesian optimization to choose which one to label
                x_sample = params.sample_x(int(1e3))
                x_nn_sample = params.to_nn_input(x_sample)

                if kernel == 'nngp':
                    kernel_train_train = kernel_fn(X_nn_train, X_nn_train, 'nngp')
                else:
                    kernel_train_train = kernel_fn(X_nn_train, X_nn_train)  # Need both NTK and NNGP covariances
                predict_fn = nt.predict.gp_inference(kernel_train_train, Y_train)

                def f(x):
                    if kernel == 'nngp':
                        k_test_train = kernel_fn(x, X_nn_train, 'nngp')
                    else:
                        k_test_train = kernel_fn(x, X_nn_train)
                    nngp_test_test = kernel_fn(x, None, 'nngp')
                    mean, covariance = predict_fn(get=kernel, k_test_train=k_test_train, nngp_test_test=nngp_test_test)
                    return mean, np.sqrt(np.diag(covariance)[:, np.newaxis])

                # Pick out and label new data point
                i_new, x_nn_new = bo.ei_direct_batch(x_nn_sample, f, y_best)
                x_new = x_sample[i_new][np.newaxis, :]
                y_new = obj_fun(prob_fun(x_new))[:, np.newaxis]

                # Add the labelled data point to our training data set
                X_train = np.vstack((X_train, x_new))
                X_nn_train = np.vstack((X_nn_train, x_nn_new))
                Y_train = np.vstack((Y_train, y_new))
                dm.add_data(x_nn_new, y_new)

                # Update the best data point so far
                i_best = np.argmax(Y_train)
                x_best = X_train[i_best]
                x_nn_best = X_nn_train[i_best]
                y_best = Y_train[i_best]

                # Save results to a file
                n_data_arr.append(dm.n)
                y_best_list.append(y_best)
                print("Trained with %d data points. Best value=%f" % (dm.n, y_best))
                np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=y_best_list, best_x=x_best,
                         best_x_nn=x_nn_best)
            time_tot = time.time() - start_time
            print("Took %f seconds" % time_tot)
            np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=y_best_list,
                     time_tot=time_tot, best_x=x_best, best_x_nn=x_nn_best)
            print(x_best)
            print(y_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for optimization using neural tangents")
    parser.add_argument("--results-dir", type=str, default='results/opt/test')
    parser.add_argument("--n-batch", type=int, default=10)
    parser.add_argument("--n_train", type=int, default=1000)
    # Optimization
    parser.add_argument("--opt", type=str, default="ensemble",
                        choices=["ensemble", "single"],
                        help="Infinite-width model for surrogate. `Ensemble` refers to infinite ensemble.")
    parser.add_argument("--kernel", type=str, default="nngp",
                        choices=["nngp", "ntk"],
                        help="Kernel for optimization")
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
