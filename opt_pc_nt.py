"""Optimize photonic crystal using the Neural Tangents library to implement infinite-width and infinite-ensemble
approximations of neural networks as GPs."""
import argparse
import time
from lib import acquisition as bo
from lib import data_manager
import os
import sys
import numpy as np
from pc import level_set, DOS_GGR
from lib.helpers import get_trial_dir
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

xdim = 51


def get_obj(objective='gap'):
    if objective == 'dos10':
        def obj_fun(dos):
            dosmin = np.sum(dos[:, 300:350], axis=1)
            dosmax = np.sum(dos[:, :300], axis=1) + np.sum(dos[:, 350:], axis=1)
            obj = dosmax / (dosmin + 1)     # Avoid dividing by small numbers
            return obj
    else:
        raise ValueError("Could not find an objective function with that name.")
    return obj_fun


def get_problem(leveller=None, objective="gap", sample_full=True):
    """Get objective function and problem parameters"""
    if objective == "dos10":
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


def main(results_dir, n_batch, n_train=1000, sample_full=True,
         opt="random", objective="gap", trials=1, kernel='nngp', trial_i=0):
    leveller = level_set.FourierLevelSet(eps_in=1.0, eps_out=11.4)
    obj_fun = get_problem(leveller=leveller, objective=objective, sample_full=sample_full)
    obj_fun2 = get_obj(objective=objective)

    if opt == 'ensemble' or opt == 'single':
        # import jax.numpy as np
        os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
        from jax import random
        from jax.experimental import optimizers
        from jax.api import jit, grad, vmap
        import neural_tangents as nt
        from neural_tangents import stax

        n_start = 5     # Size of initial dataset

        for _ in range(trials):
            results_dir_i, _ = get_trial_dir(os.path.join(results_dir, 'trial%d'), trial_i)

            X_train = np.random.rand(n_start, xdim)
            X_nn_train = leveller.calc_data(X_train, sample_level=True, sample_full=sample_full)[:, :, :, np.newaxis]
            Y_train = obj_fun2(obj_fun(X_train))[:, np.newaxis]

            # Object to control minibatches
            dm = data_manager.ImageDataManager(X_nn_train, Y_train, n_batch)

            # Define network architecture
            parameterization = 'standard'   # 'standard' or 'ntk'
            pool = False
            if pool:
                init_fn, apply_fn, kernel_fn = stax.serial(
                    stax.Conv(32, (3, 3), (1, 1), padding='CIRCULAR'), stax.Relu(),
                    stax.AvgPool((2, 2), (2, 2), padding='CIRCULAR'),
                    stax.Conv(32, (3, 3), (1, 1), padding='CIRCULAR'), stax.Relu(),
                    stax.AvgPool((2, 2), (2, 2), padding='CIRCULAR'),
                    stax.Conv(32, (3, 3), (2, 2), padding='CIRCULAR'), stax.Relu(),
                    stax.AvgPool((2, 2), (1, 1), padding='CIRCULAR'),
                    stax.Conv(32, (3, 3), (2, 2), padding='CIRCULAR'), stax.Relu(),
                    stax.AvgPool((2, 2), (2, 2), padding='CIRCULAR'),
                    stax.Conv(32, (3, 3), (2, 2), padding='CIRCULAR'), stax.Relu(),
                    stax.AvgPool((2, 2), (2, 2), padding='CIRCULAR'),
                    stax.Flatten(),
                    stax.Dense(128), stax.Relu(),
                    stax.Dense(64), stax.Relu(),
                    stax.Dense(16), stax.Relu(),
                    stax.Dense(4), stax.Relu(),
                    stax.Dense(1)
                )
            else:
                init_fn, apply_fn, kernel_fn = stax.serial(
                    stax.Conv(32, (5, 5), (1, 1), padding='CIRCULAR', parameterization=parameterization), stax.Relu(),
                    stax.Conv(32, (5, 5), (1, 1), padding='CIRCULAR', parameterization=parameterization), stax.Relu(),
                    stax.Conv(32, (5, 5), (1, 1), padding='CIRCULAR', parameterization=parameterization), stax.Relu(),
                    stax.Conv(32, (5, 5), (1, 1), padding='CIRCULAR', parameterization=parameterization), stax.Relu(),
                    stax.Conv(32, (5, 5), (1, 1), padding='CIRCULAR', parameterization=parameterization), stax.Relu(),
                    stax.Flatten(),
                    stax.Dense(128, parameterization=parameterization), stax.Relu(),
                    stax.Dense(64, parameterization=parameterization), stax.Relu(),
                    stax.Dense(16, parameterization=parameterization), stax.Relu(),
                    stax.Dense(4, parameterization=parameterization), stax.Relu(),
                    stax.Dense(1, parameterization=parameterization)
                )
            if opt == 'ensemble':
                kernel_fn = jit(kernel_fn, static_argnums=(2,))

            start_time = time.time()

            # Best candidate data point so far
            y_best_i = np.argmax(Y_train, axis=0)
            x_best = X_train[y_best_i, :]
            y_best = np.max(Y_train)

            n_data_arr = []
            y_best_list = []

            x_nn_best = None
            x_nn_new = None
            y_new = None

            for i in range(n_train):

                x_sample = np.random.rand(int(1e5), xdim)
                x_nn_sample = leveller.calc_data(x_sample, sample_level=True,
                                                 sample_full=sample_full)[:, :, :, np.newaxis]

                if opt == 'ensemble':
                    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_nn_train, Y_train, diag_reg=1e-4)

                    def f(x):
                        mean, covariance = predict_fn(x_test=x, get=kernel, compute_cov=True)
                        return mean, np.sqrt(np.diag(covariance))[:, np.newaxis]
                else:   # opt == 'single'
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
                        mean, covariance = predict_fn(get=kernel, k_test_train=k_test_train,
                                                      nngp_test_test=nngp_test_test)
                        return mean, np.sqrt(np.diag(covariance)[:, np.newaxis])

                # Pick out and label new data point
                i_new, x_nn_new = bo.ei_direct(x_nn_sample, f, y_best, batch_size=256)
                x_nn_new = x_nn_new[np.newaxis]     # Shape (1, 32, 32, 1)
                x_new = x_sample[i_new][np.newaxis, :]

                y_new = obj_fun2(obj_fun(x_new))[:, np.newaxis]
                # Add the labelled data point to our training data set
                X_train = np.vstack((X_train, x_new))
                print(X_nn_train.shape)
                print(x_nn_new.shape)
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
    # Dataset
    parser.add_argument('--sample-full', dest='sample_full', action='store_true')
    parser.add_argument('--no-sample-full', dest='sample_full', action='store_false')
    parser.set_defaults(sample_full=True)
    # Optimization
    parser.add_argument("--opt", type=str, default="ensemble",
                        choices=['ensemble', 'single'],
                        help="Infinite-width model for surrogate. `Ensemble` refers to infinite ensemble.")
    parser.add_argument("--kernel", type=str, default="nngp",
                        choices=["nngp", "ntk"],
                        help="Kernel for NTK library")
    parser.add_argument("--objective", type=str, default="gap", choices=['dos10'])
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
