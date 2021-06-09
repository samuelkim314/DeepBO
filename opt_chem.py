"""Optimize - maximize chemical property from QM9 dataset"""
import argparse
import time
from lib import acquisition as bo
from lib import data_manager
from lib.models import nn
from lib import helpers
import os
import numpy as np

from spektral.datasets import qm9
from spektral.utils import label_to_one_hot


def get_objective_y(y, objective='gap'):
    # Columns: A, B, C, mu, alpha, homo, lumo, gap, r2
    y1 = y.loc[:, 'A':'r2'].values
    y1[:, 0] = y1[:, 0] / 6.2e5
    y1[:, 1] = y1[:, 1] / 4.4e2
    y1[:, 2] = y1[:, 2] / 2.8e2
    y1[:, 3] = y1[:, 3] / 30
    y1[:, 4] = (y1[:, 4] - 6) / 191     # alpha
    y1[:, 5] = (y1[:, 5] + 0.4) / 0.5
    y1[:, 6] = (y1[:, 6] + 0.2) / 0.4
    y1[:, 7] = (y1[:, 7] - 0.02) / 0.6  # gap, eps_lumo - eps_homo
    y1[:, 8] = (y1[:, 8] - 19) / 3.4e3

    # Thermodynamic quantities, normalized. Columns: u298, h298, g298, cv
    y2 = y.loc[:, 'u298':'cv'].values
    y2[:, 0:3] = (y2[:, 0:3] + 700) / 750
    y2[:, 3] = (y2[:, 3] - 6) / 52

    # The objective function picks out the desired feature and un-normalizes it
    obj_fun = None
    if objective == 'gap':
        def obj_fun(z):
            return z[:, 7] * 0.6 + 0.02
    elif objective == 'mingap':
        def obj_fun(z):
            return -(z[:, 7] * 0.6 + 0.02)
    elif objective == 'alpha':
        def obj_fun(z):
            return z[:, 4] * 191 + 6
    elif objective == 'minalpha':
        def obj_fun(z):
            return -(z[:, 4] * 191 + 6)
    elif objective == 'cv':     # Heat capacity at 298.15K
        def obj_fun(z):
            return z[:, 3] * 52 + 6
    elif objective == 'mincv':
        def obj_fun(z):
            return - (z[:, 3] * 52 + 6)

    if objective == 'gap' or objective == 'mingap' or objective == 'alpha' or objective == 'minalpha':
        return y1, obj_fun
    else:
        return y2, obj_fun


def main(results_dir, n_batch, n_epochs, n_train=1000,
         opt="random", acquisition="ei", objective="gap", trials=1, nn_args=None,
         n_epochs_continue=10, iter_restart_training=100, n_mc=30, weighted_training=True, trial_i=0,
         n_start=5):
    A, X, E, y = qm9.load_data(return_type='numpy',
                               nf_keys='atomic_num',
                               ef_keys='type',
                               self_loops=True,
                               amount=None)  # Set to None to train on whole dataset

    print(y)
    # print(y[[objective]].values)

    x = y[['mol_id']].values  # String index
    z, obj_fun = get_objective_y(y, objective)
    y = obj_fun(z)[:, np.newaxis]

    print(np.max(y))

    if opt == "random":
        # Random selection, and then pick out the best candidate afterwards

        best_y = []

        for i in range(trials):
            # Randomly select n_train samples from the full dataset
            ind_i = np.random.choice(range(y.shape[0]), size=n_train, replace=False)
            x_i = x[ind_i]
            y_i = y[ind_i]

            best_y_i = []
            best_x_i = []
            for j in np.arange(n_train)+1:
                argmax_j = np.argmax(y_i[:j])
                best_y_i.append(y_i[argmax_j])
                best_x_i.append(x_i[argmax_j])
                print(best_y_i[-1])
            np.savez(os.path.join(results_dir, 'trial%d' % i), n_data=np.arange(n_train), best_y=best_y_i,
                     best_x=best_x_i)
            best_y.append(best_y_i)
        best_y_arr = np.mean(best_y, axis=0)
        best_y_std = np.std(best_y, axis=0)
        np.savez(os.path.join(results_dir, 'best'), n_data=np.arange(n_train), best_y=best_y_arr, best_y_std=best_y_std)
    elif opt == 'gp':
        import GPyOpt

        data_path = os.path.expanduser('~/.spektral/datasets/qm9')
        # structures = io.read(os.path.join(data_path, 'qm9.xyz'), index=':')
        feature_vectors = np.load(os.path.join(data_path, 'soap.npy'))
        # Normalize data
        x_min = np.min(feature_vectors)
        x_max = np.max(feature_vectors)
        feature_vectors = ((feature_vectors - x_min) / (x_max - x_min) - 0.5) * 2

        n = feature_vectors.shape[0]
        ind = np.arange(n)  # Indexing the original data - convenient IDs

        model = GPyOpt.models.gpmodel.GPModel(exact_feval=True)

        for i in range(trials):
            results_dir_i, _ = helpers.get_trial_dir(os.path.join(results_dir, 'trial%d'), i0=trial_i)

            # Data manager for the data pool
            dm_pool = data_manager.DataManager(feature_vectors, y, batch_size=n_batch)
            # Random initialization of initial dataset by choosing randomly from the pool
            ind_train = np.random.choice(range(y.shape[0]), size=n_start, replace=False)
            X_train, Y_train = dm_pool.get_data(ind_train)
            dm_pool.remove_data(ind_train)  # Removing the initial data from the pool
            ind_pool = np.delete(ind, ind_train, 0)
            # Data manager for labelled data
            dm = data_manager.DataManager(X_train, Y_train, batch_size=n_batch)

            # Best candidate data point so far
            x_best, y_best = dm.get_best()

            n_data_arr = []
            best_y_arr = []

            start_time = time.time()

            for j in range(n_train):
                model.updateModel(X_train, Y_train, X_new=None, Y_new=None)
                # Y_hat, Y_std = model.predict(dm_pool.X)   # shapes (n, 1)

                # Bayesian optimization to choose which new point to label
                i_new, x_new = bo.ei(dm_pool.X, model.predict, y_best, batch_size=8192)
                i_new = [i_new]
                x_new = x_new[np.newaxis, :]

                # Add data to training dataset
                _, y_new, = dm_pool.get_data(i_new)     # Get new data
                dm.add_data(x_new, y_new)              # Add to labelled data set

                ind_train = np.append(ind_train, ind_pool[i_new])
                ind_pool = np.delete(ind_pool, i_new, 0)

                dm_pool.remove_data(i_new)  # Remove new data from pool

                x_best, y_best = dm.get_best()

                # Save results to a file
                n_data_arr.append(dm.n)
                best_y_arr.append(y_best)
                print("Trained with %d data points. Best value=%f" % (dm.n, y_best))
                np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=best_y_arr, best_x=x_best, )
            time_tot = time.time() - start_time
            print("Took %f seconds" % time_tot)
            np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=best_y_arr,
                     time_tot=time_tot, best_x=x_best, ind_train=ind_train)
            print(x_best)
            print(y_best)

    elif opt == "nn":
        # Bayesian optimization using Bayesian neural networks with continued training - directly on objective
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            # Preprocessing
            uniq_X = np.unique(X)
            uniq_X = uniq_X[uniq_X != 0]
            X = label_to_one_hot(X, uniq_X)
            uniq_E = np.unique(E)
            uniq_E = uniq_E[uniq_E != 0]
            E = label_to_one_hot(E, uniq_E)
            ind = np.arange(X.shape[0])

            for _ in range(trials):
                results_dir_i, _ = helpers.get_trial_dir(os.path.join(results_dir, 'trial%d'), i0=trial_i)

                # Random initialization of initial dataset
                ind_train = np.random.choice(range(y.shape[0]), size=n_start, replace=False)
                X_train = X[ind_train]
                A_train = A[ind_train]
                E_train = E[ind_train]
                Y_train = y[ind_train]
                X_pool = np.delete(X, ind_train, 0)
                A_pool = np.delete(A, ind_train, 0)
                E_pool = np.delete(E, ind_train, 0)
                Y_pool = np.delete(y, ind_train, 0)
                ind_pool = np.delete(ind, ind_train, 0)

                dm = data_manager.ChemDataManager(X_train, A_train, E_train, Y_train, batch_size=n_batch)
                dm_pool = data_manager.ChemDataManager(X_pool, A_pool, E_pool, Y_pool, batch_size=n_batch)

                model = nn.choose_model(**nn_args, dm=dm, results_dir=results_dir_i,
                                        print_loss=False, opt_name="adam")

                def f(X, A, E):
                    """Return samples from the posterior distribution of predictions
                    Output shape: (n_data, n_features, n_sample) array"""
                    return model.predict_posterior(sess, X, A, E, dm, n=n_mc)

                init = tf.global_variables_initializer()
                sess.run(init)
                start_time = time.time()

                # Best candidate data point so far
                x_best, a_best, e_best, y_best = dm.get_best()

                n_data_arr = []
                best_y_arr = []

                dm_new = None   # data manager for the newly added data point

                for i in range(n_train):
                    if i % iter_restart_training == 0:
                        # model.reset(sess)  # Retrain the model from scratch
                        sess.run(init)
                        epochs_i = n_epochs
                        anneal_i = nn_args['anneal']
                    else:   # Continue training
                        epochs_i = n_epochs_continue
                        anneal_i = False

                    loss_final = model.train(sess, epochs_i, dm, early_stopping=True, dm_new=dm_new, anneal=anneal_i)
                    # print(loss_final)

                    # Bayesian optimization to choose which new point to label
                    if nn_args['uncertainty'] == 'graph_neurallinear':
                        i_new, x_new = bo.ei_direct_chem(dm_pool.X, dm_pool.A, dm_pool.E, f, y_best, batch_size=256)  # Data point to label
                    else:
                        i_new, x_new = bo.ei_mc_chem(dm_pool.X, dm_pool.A, dm_pool.E, f, y_best, batch_size=256)  # Data point to label
                    a_new = dm_pool.A[[i_new]]
                    e_new = dm_pool.E[[i_new]]
                    y_new = dm_pool.Y[[i_new]]
                    ind_train = np.append(ind_train, ind_pool[i_new])
                    ind_pool = np.delete(ind_pool, i_new, 0)

                    if weighted_training:
                        dm_new = data_manager.ChemDataManager(np.repeat(x_new, 5, axis=0), np.repeat(a_new, 5, axis=0),
                                                              np.repeat(e_new, 5, axis=0), np.repeat(y_new, 5, axis=0),
                                                              n_batch)
                    dm.add_data(x_new, a_new, e_new, y_new)
                    dm_pool.remove_data(i_new)

                    x_best, _, _, y_best = dm.get_best()

                    # Save results to a file
                    n_data_arr.append(dm.n)
                    best_y_arr.append(y_best)
                    print("Trained with %d data points. Best value=%f" % (dm.n, y_best))
                    np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=best_y_arr, best_x=x_best,)
                time_tot = time.time() - start_time
                print("Took %f seconds" % time_tot)
                np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=best_y_arr,
                         time_tot=time_tot, best_x=x_best, ind_train=ind_train)
                print(x_best)
                print(y_best)
    elif opt == "nn2":
        # Bayesian optimization using Bayesian neural networks with continued training - directly on objective
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            # Preprocessing
            uniq_X = np.unique(X)
            uniq_X = uniq_X[uniq_X != 0]
            X = label_to_one_hot(X, uniq_X)
            uniq_E = np.unique(E)
            uniq_E = uniq_E[uniq_E != 0]
            E = label_to_one_hot(E, uniq_E)
            ind = np.arange(X.shape[0])  # Indexing the original data - convenient IDs

            for _ in range(trials):
                results_dir_i, _ = helpers.get_trial_dir(os.path.join(results_dir, 'trial%d'), i0=trial_i)

                # Data manager for the data pool
                dm_pool = data_manager.ChemDataManager(X, A, E, y, Z=z, batch_size=n_batch)
                # Random initialization of initial dataset by choosing randomly from the pool
                ind_train = np.random.choice(range(y.shape[0]), size=n_start, replace=False)    # Choose random indices
                X_train, A_train, E_train, Y_train, Z_train = dm_pool.get_data(ind_train)
                dm_pool.remove_data(ind_train)  # Removing the initial data from the pool
                ind_pool = np.delete(ind, ind_train, 0)
                # Data manager for labelled data
                dm = data_manager.ChemDataManager(X_train, A_train, E_train, Y_train, Z=Z_train, batch_size=n_batch)

                model = nn.choose_model(**nn_args, dm=dm, results_dir=results_dir_i, print_loss=False, opt_name="adam")

                def f(X, A, E):
                    """Return samples from the posterior distribution of predictions
                    Output shape: (n_data, n_features, n_sample) array"""
                    return model.predict_posterior(sess, X, A, E, dm, n=n_mc)

                init = tf.global_variables_initializer()
                sess.run(init)
                start_time = time.time()

                # Best candidate data point so far
                _, _, _, y_best = dm.get_best()
                # print(y_best)

                n_data_arr = []
                best_y_arr = []

                dm_new = None  # data manager for the newly added data point - for weighted training

                for i in range(n_train):
                    if i % iter_restart_training == 0:
                        # model.reset(sess)  # Retrain the model from scratch
                        sess.run(init)
                        epochs_i = n_epochs
                        anneal_i = nn_args['anneal']
                    else:  # Continue training
                        epochs_i = n_epochs_continue
                        anneal_i = False

                    loss_final = model.train(sess, epochs_i, dm, early_stopping=True, dm_new=dm_new, anneal=anneal_i)
                    # print(loss_final)

                    # Bayesian optimization to choose which new point to label
                    i_new, x_new = bo.ei_mc_chem(dm_pool.X, dm_pool.A, dm_pool.E, f, y_best, obj_fun=obj_fun)
                    # Add data to training dataset and dm_new
                    _, a_new, e_new, y_new, z_new = dm_pool.get_data(i_new)     # Get new data
                    dm.add_data(x_new, a_new, e_new, y_new, z_new)              # Add to labelled data set

                    ind_train = np.append(ind_train, ind_pool[i_new])
                    ind_pool = np.delete(ind_pool, i_new, 0)

                    if weighted_training:
                        dm_new = data_manager.ChemDataManager(np.repeat(x_new, 5, axis=0), np.repeat(a_new, 5, axis=0),
                                                              np.repeat(e_new, 5, axis=0), np.repeat(z_new, 5, axis=0),
                                                              n_batch)

                    dm_pool.remove_data(i_new)  # Remove new data from pool

                    x_best, _, _, y_best = dm.get_best()

                    # print(dm.Y)

                    # Save results to a file
                    n_data_arr.append(dm.n)
                    best_y_arr.append(y_best)
                    print("Trained with %d data points. Best value=%f" % (dm.n, y_best))
                    np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=best_y_arr, best_x=x_best, )
                time_tot = time.time() - start_time
                print("Took %f seconds" % time_tot)
                np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=best_y_arr,
                         time_tot=time_tot, best_x=x_best, ind_train=ind_train)
                print(x_best)
                print(y_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural net")
    parser.add_argument("--results-dir", type=str, default='results/opt/test')
    parser.add_argument("--n-batch", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-epochs-continue", type=int, default=10)
    parser.add_argument("--iter-restart-training", type=int, default=100)
    parser.add_argument("--n-start", type=int, default=5)
    parser.add_argument("--n_train", type=int, default=500)
    # Weighted training for nn when adding new data point
    parser.add_argument('--weighted-training', dest='weighted_training', action='store_true')
    parser.add_argument('--no-weighted-training', dest='weighted_training', action='store_false')
    parser.set_defaults(weighted_training=True)
    # Optimization
    parser.add_argument("--opt", type=str, default="random",
                        choices=["random", 'gp', "nn", 'nn2'],
                        help="Model for optimization")
    parser.add_argument('--n-mc', type=int, default=30, help='Number of times to sample Bayesian model')
    parser.add_argument("--acquisition", type=str, default="EI", choices=["EI"],
                        help="Acquisition function to label a new point")
    parser.add_argument("--objective", type=str, default="gap",
                        choices=['gap', 'cv', 'alpha', 'mingap', 'mincv', 'minalpha'])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument('--trial-i', type=int, default=0)
    parser = nn.add_args(parser)

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

    kwargs, nn_args = nn.process_args(kwargs)
    main(**kwargs, nn_args=nn_args)
