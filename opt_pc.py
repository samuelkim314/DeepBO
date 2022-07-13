"""Optimize density of states (DOS) of a photonic crystal parameterized by Fourier components"""
import argparse
import time
from lib import acquisition as bo
from lib import data_manager
from lib.models import nn
from lib.helpers import get_trial_dir
import os
import sys
import numpy as np
from pc import level_set
from pc import DOS_GGR
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

xdim = 51


def get_obj(objective='dos'):
    if objective == 'dos':
        def obj_fun(dos):
            dosmin = np.sum(dos[:, 300:350], axis=1)
            dosmax = np.sum(dos[:, :300], axis=1) + np.sum(dos[:, 350:], axis=1)
            obj = dosmax / (dosmin + 1)     # Avoid dividing by small numbers
            return obj
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


def main(results_dir, n_batch, n_epochs, n_train=1000, sample_full=True,
         opt="random", acquisition="ei", objective="dos", trials=1, nn_args=None, kernel='nngp',
         n_epochs_continue=10, iter_restart_training=100, af_n=30, weighted_training=True, trial_i=0, augment=False,
         af_m=int(1e5), n_units=0, n_layers=0, lr_cycle=False, lr_cycle_base=False):
    leveller = level_set.FourierLevelSet(eps_in=1.0, eps_out=11.4)
    obj_fun = get_problem(leveller=leveller, objective=objective, sample_full=sample_full)
    obj_fun2 = get_obj(objective=objective)

    if opt == "random":
        batch_size = 10

        # Random selection, and then pick out the best candidate afterwards
        n_data_arr = range(n_batch, n_train, batch_size)

        best_y = []
        for i in range(trials):
            x = np.zeros((0, xdim))
            y = []
            best_y_i = []
            best_x_i = []
            for _ in n_data_arr:
                x_i = np.random.rand(batch_size, xdim)
                y_i = obj_fun2(obj_fun(x_i))
                x = np.concatenate((x, x_i), axis=0)
                y.append(y_i)
                best_i = np.argmax(y)
                best_x_i.append(x[best_i, :])
                best_y_i.append(np.max(y))
                print(best_y_i[-1])
            np.savez(os.path.join(results_dir, 'trial%d' % i), n_data=n_data_arr, best_y=best_y_i,
                     best_x=best_x_i)
            best_y.append(best_y_i)
        y_best_list = np.mean(best_y, axis=0)
        best_y_std = np.std(best_y, axis=0)
        np.savez(os.path.join(results_dir, 'best'), n_data=n_data_arr, best_y=y_best_list, best_y_std=best_y_std)
    elif opt == "gp":
        # Bayesian optimization using Gaussian processes
        import GPyOpt

        def fun(x):
            return -obj_fun2(obj_fun(x))[:, np.newaxis]

        # Bounds on the values of x
        bounds = [{'name': 'x', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': xdim}]

        max_iter = n_train

        for _ in range(trials):
            # TODO: Noiseless evaluation
            prob = GPyOpt.methods.BayesianOptimization(fun, bounds,
                                                       model_type='GP',
                                                       acquisition_type=acquisition)  # EI, LCB, MPI
            prob.run_optimization(max_iter, verbosity=True, report_file=os.path.join(results_dir, 'report'))

            print(prob.x_opt)
            print(prob.fx_opt)

            file_format = os.path.join(results_dir, 'eval%d')
            i = 0
            while True:
                results_i = file_format % i
                if os.path.exists(results_i):
                    i += 1
                else:
                    break
            prob.save_evaluations(os.path.join(results_dir, 'eval%d' % i))
            prob.save_report(os.path.join(results_dir, 'report%d' % i))
            # # TODO: This is a hack because GPyOpt complains it doesn't have initial_design_numdata
            # try:
            #     prob.save_report(os.path.join(results_dir, 'report%d' % i))
            # except Exception as e:
            #     print("Error in saving report")
            #     print(e)
    elif opt == "nn" or opt == 'cnn':
        # Bayesian optimization using Bayesian neural networks with continued training - directly on objective
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        import tensorflow as tf

        n_start = 5     # Size of initial dataset

        # n_channels = [8, 8, 16, 32, 32]
        n_channels = [16, 32, 64, 128, 256]
        n_units = [n_units] * n_layers

        if nn_args['uncertainty'] == 'neurallinear' or nn_args['uncertainty'] == 'conv_neurallinear':
            mc = False
        else:
            mc = True

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for _ in range(trials):
                results_dir_i, _ = get_trial_dir(os.path.join(results_dir, 'trial%d'), trial_i)

                X_train = np.random.rand(n_start, xdim)
                if opt == 'cnn':
                    X_nn_train = leveller.calc_data(X_train, sample_level=True, sample_full=sample_full)[:, :, :, np.newaxis]
                else:
                    X_nn_train = X_train
                Y_train = obj_fun2(obj_fun(X_train))[:, np.newaxis]

                if opt == 'cnn':
                    # Object to control minibatches
                    dm = data_manager.ImageDataManager(X_nn_train, Y_train, n_batch)
                    # Initialize Bayesian neural network
                    model = nn.choose_model(**nn_args, dm=dm, opt_name="adam", n_channels=n_channels, n_units=n_units)
                else:
                    dm = data_manager.DataManager(X_nn_train, Y_train, n_batch)
                    model = nn.choose_model(**nn_args, dm=dm,  opt_name="adam", n_units=n_units)

                def f(samples):
                    """Return samples from the posterior distribution of predictions"""
                    return model.predict_posterior(sess, samples, dm, n=af_n)

                init = tf.global_variables_initializer()
                sess.run(init)
                start_time = time.time()

                # Best candidate data point so far
                y_best_i = np.argmax(Y_train, axis=0)
                x_best = X_train[y_best_i, :]
                y_best = np.max(Y_train)

                n_data_arr = []
                y_best_list = []

                for i in range(n_train):
                    if i % iter_restart_training == 0:
                        model.reset(sess)  # Retrain the model from scratch after we collect each point
                        epochs_i = n_epochs
                        cycle_i = False
                    else:   # Continue training
                        epochs_i = n_epochs_continue
                        cycle_i = lr_cycle

                    # Training step
                    loss_final = model.train(sess, epochs_i, dm, X_val=X_nn_train, Y_val=Y_train,
                                             save_model=False, augment=augment, augment_sg11=augment, cycle=cycle_i)

                    # Random set of unlabelled x points - we will use Bayesian optimization to choose which one to label
                    if opt == 'cnn':
                        x_sample = np.random.rand(int(af_m), xdim)
                        x_nn_sample = leveller.calc_data(x_sample, sample_level=True,
                                                         sample_full=sample_full)[:, :, :, np.newaxis]
                        if mc:
                            i_max, x_nn_new = bo.ei_mc(x_nn_sample, f, y_best, batch_size=512)
                        else:
                            i_max, x_nn_new = bo.ei_direct(x_nn_sample, f, y_best)
                            x_nn_new = np.array([x_nn_new])
                        x_new = x_sample[[i_max]]
                    else:
                        x_sample = np.random.rand(int(af_m), xdim)
                        if mc:
                            i_max, x_new = bo.ei_mc(x_sample, f, y_best)
                        else:
                            i_max, x_new = bo.ei_direct(x_sample, f, y_best)
                            x_new = np.array([x_new])
                        x_nn_new = x_new

                    y_new = obj_fun2(obj_fun(x_new))[:, np.newaxis]
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
    elif opt == "nn2" or opt == 'cnn2':
        # Bayesian optimization using Bayesian neural networks with continued training and auxiliary information
        # NN predicts DOS, not objective function
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        import tensorflow as tf

        n_start = 5  # Size of initial dataset

        # n_channels = [8, 8, 16, 32, 32]
        n_channels = [16, 32, 64, 128, 256]
        n_units = [n_units] * n_layers

        if not nn_args['uncertainty'] == 'neurallinear':
            mc = True

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for _ in range(trials):
                results_dir_i, _ = get_trial_dir(os.path.join(results_dir, 'trial%d'), trial_i)

                # Random initialization of initial dataset
                X_train = np.random.rand(n_start, xdim)
                if opt == 'cnn2':
                    X_nn_train = leveller.calc_data(X_train, sample_level=True, sample_full=sample_full)[:, :, :, np.newaxis]
                else:
                    X_nn_train = X_train
                Z_train = obj_fun(X_train)  # Calculate DOS using MPB
                Y_train = obj_fun2(Z_train)

                if opt == 'cnn2':
                    # Object to control minibatches
                    dm = data_manager.ImageDataManager(X_nn_train, Z_train, n_batch)
                    # Initialize Bayesian neural network
                    model = nn.choose_model(**nn_args, dm=dm, opt_name="adam", n_channels=n_channels, n_units=n_units)
                else:
                    dm = data_manager.DataManager(X_nn_train, Z_train, n_batch)
                    model = nn.choose_model(**nn_args, dm=dm, opt_name="adam", n_units=n_units)

                def f(samples):
                    """Return samples from the posterior distribution of predictions"""
                    return model.predict_posterior(sess, samples, dm, n=af_n)

                init = tf.global_variables_initializer()
                sess.run(init)
                start_time = time.time()

                # Best candidate data point so far
                y_best_i = np.argmax(Y_train, axis=0)
                x_best = X_train[y_best_i, :]
                y_best = np.max(Y_train)

                n_data_arr = []
                y_best_list = []

                for i in range(n_train):
                    if i % iter_restart_training == 0:
                        model.reset(sess)  # Retrain the model from scratch after we collect each point
                        epochs_i = n_epochs
                        cycle_i = lr_cycle_base
                    else:  # Continue training
                        epochs_i = n_epochs_continue
                        cycle_i = lr_cycle

                    # Training step
                    loss_final = model.train(sess, epochs_i, dm, X_val=X_nn_train, Y_val=Z_train,
                                             save_model=False, augment=augment, augment_sg11=augment, cycle=cycle_i)

                    # Random set of unlabelled x points - we will use Bayesian optimization to choose which one to label
                    # x_new has shape (1, xdim)
                    if opt == 'cnn2':
                        x_sample = np.random.rand(int(af_m), xdim)
                        x_nn_sample = leveller.calc_data(x_sample, sample_level=True,
                                                         sample_full=sample_full)[:, :, :, np.newaxis]

                        # Data point to label
                        # if acquisition == 'EI':
                        #     x_new, x_nn_new = bo.ei_im(x_sample, x_nn_sample, f, y_best, obj_fun2)
                        # else:
                        i_max, x_nn_new = bo.ei_mc(x_nn_sample, f, y_best, obj_fun2, batch_size=512)
                        x_new = x_sample[[i_max]]
                    else:
                        x_sample = np.random.rand(int(af_m), xdim)
                        # if acquisition == 'EI':
                        #     x_new = bo.ei_batched(x_sample, f, y_best, obj_fun2)  # Data point to label
                        # else:
                        i_max, x_new = bo.ei_mc(x_sample, f, y_best, obj_fun2)
                        x_nn_new = x_new

                    z_new = obj_fun(x_new)  # Run MPB calculation
                    # Add the labelled data point to our training data set
                    X_train = np.vstack((X_train, x_new))
                    X_nn_train = np.vstack((X_nn_train, x_nn_new))
                    Z_train = np.vstack((Z_train, z_new))
                    dm.add_data(x_nn_new, z_new)

                    # Update the best data point so far
                    i_best = np.argmax(obj_fun2(Z_train))
                    x_best = X_train[i_best]
                    x_nn_best = X_nn_train[i_best]
                    y_best = obj_fun2(Z_train)[i_best]
                    z_best = Z_train[i_best]

                    # Save results to a file
                    n_data_arr.append(dm.n)
                    y_best_list.append(y_best)
                    print("Trained with %d data points. Best value=%f" % (dm.n, y_best))
                    np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=y_best_list,
                             best_x=x_best, best_x_nn=x_nn_best, z_best=z_best)
                time_tot = time.time() - start_time
                print("Took %f seconds" % time_tot)
                np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=y_best_list,
                         time_tot=time_tot, best_x=x_best, best_x_nn=x_nn_best, z_best=z_best)
                print(x_best)
                print(y_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural net")
    parser.add_argument("--results-dir", type=str, default='results/opt/test')
    parser.add_argument("--n-batch", type=int, default=10)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-epochs-continue", type=int, default=10)
    parser.add_argument("--iter-restart-training", type=int, default=100)
    parser.add_argument("--n_train", type=int, default=1000)
    # Dataset. sample_full=True corresponds to PC-A dataset
    parser.add_argument('--sample-full', dest='sample_full', action='store_true')
    parser.add_argument('--no-sample-full', dest='sample_full', action='store_false')
    parser.set_defaults(sample_full=True)
    # Optional arguments for size of neural network - just the fully-connected layers
    parser.add_argument("--n-units", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    # Weighted training for nn when adding new data point
    parser.add_argument('--weighted-training', dest='weighted_training', action='store_true')
    parser.add_argument('--no-weighted-training', dest='weighted_training', action='store_false')
    parser.set_defaults(weighted_training=True)
    # LR cycle
    parser.add_argument('--lr-cycle', dest='lr_cycle', action='store_true')
    parser.add_argument('--no-lr-cycle', dest='lr_cycle', action='store_false')
    parser.set_defaults(lr_cycle=False)
    parser.add_argument('--lr-cycle-base', dest='lr_cycle', action='store_true')
    parser.add_argument('--no-lr-cycle-base', dest='lr_cycle', action='store_false')
    parser.set_defaults(lr_cycle_base=False)
    # Data augmentation
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.set_defaults(augment=False)
    # Optimization
    parser.add_argument("--opt", type=str, default="random",
                        choices=["random", "gp", "nn", "nn2", "dlib", "cnn", 'cnn2', "nlopt", 'ntk', 'nt-gp', 'cma'],
                        help="Model for optimization")
    parser.add_argument("--kernel", type=str, default="nngp",
                        choices=["nngp", "ntk"],
                        help="Kernel for NTK library")
    parser.add_argument('--af-n', type=int, default=30, help='Number of times to sample Bayesian model')
    parser.add_argument('--af-m', type=int, default=int(1e4), help='Number of data points to sample for acquisition')
    parser.add_argument("--acquisition", type=str, default="EI",
                        choices=["EI", "MPI", "LCB", 'EI-MC'],
                        help="Acquisition function to label a new point")
    parser.add_argument("--objective", type=str, default="dos", choices=["dos"])
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
