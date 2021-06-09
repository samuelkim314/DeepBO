"""Optimize Mie scattering spectrum of a multilayered spherical nanoparticle.
The nanoparticle is assumed to have 6 layers of alternating silica and TiO2, and scattering is measured in 350-750nm."""
import argparse
import time
from lib import acquisition as bo
from lib import data_manager
from lib.models import nn
import os
import numpy as np
from scattering import calc_spectrum


def get_obj(params, objective='orange'):
    if objective == 'orange':
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


def get_problem(params, objective="orange"):
    """Get objective function and problem parameters"""

    # Different objective functions to maximize during optimization
    if objective == "orange" or objective == 'hipass':
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


def main(results_dir,  n_batch, n_epochs, n_train=1000,
         opt="random", acquisition="ei", objective="orange", x_dim=6, trials=1, nn_args=None,
         n_epochs_continue=10, iter_restart_training=100, af_n=30, af_m=int(1e4), trial_i=0,
         n_units=0, n_layers=0, n_start=5, lr_cycle=False, lr_cycle_base=False):
    params = calc_spectrum.MieScattering(n_layers=x_dim)

    prob_fun = get_problem(params, objective=objective)
    obj_fun = get_obj(params, objective=objective)

    if opt == "random":
        # Random selection
        batch_size = 10
        n_data_list = range(n_batch, n_train, batch_size)
        for i in range(trials):
            x = np.empty((0, x_dim))
            y = np.empty(0)
            best_y_i = []
            best_x_i = []
            for _ in n_data_list:
                x_i = params.sample_x(batch_size)
                y_i = obj_fun(prob_fun(x_i))
                x = np.vstack((x, x_i))
                y = np.concatenate((y, y_i))
                best_i = np.argmax(y)
                best_x_i.append(x[best_i])
                best_y_i.append(np.max(y))
                print(best_y_i[-1])
            np.savez(os.path.join(results_dir, 'trial%d' % i), n_data=n_data_list, best_y=best_y_i, best_x=best_x_i)
    elif opt == "gp":
        # Bayesian optimization using Gaussian processes
        import GPyOpt

        def fun(x):
            return -obj_fun(prob_fun(x))[:, np.newaxis]

        # Bounds on the values of x (layer thicknesses)
        domain = [{'name': 'x2', 'type': 'continuous', 'domain': (params.th_min, params.th_max),
                   'dimensionality': params.n_layers}]

        max_iter = n_train

        for _ in range(trials):
            prob = GPyOpt.methods.BayesianOptimization(fun, domain,
                                                       model_type='GP',
                                                       acquisition_type=acquisition,    # EI, LCB, MPI
                                                       exact_feval=True)
            prob.run_optimization(max_iter, verbosity=True, report_file=os.path.join(results_dir, 'report'))

            print(prob.x_opt)
            print(prob.fx_opt)

            i = trial_i
            file_format = os.path.join(results_dir, 'eval%d')
            while True:
                results_file_i = file_format % i
                if os.path.exists(results_file_i):
                    i += 1
                else:
                    break

            prob.save_evaluations(os.path.join(results_dir, 'eval%d' % i))
            prob.save_report(os.path.join(results_dir, 'report%d' % i))
    elif opt == "nn":
        # Bayesian optimization using Bayesian neural networks - learning the objective function directly
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        import tensorflow as tf

        n_units = [n_units] * n_layers

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for _ in range(trials):
                results_dir_i, _ = get_trial_dir(os.path.join(results_dir, 'trial%d'), i0=trial_i)
                # X_train is in the original input space, x\in[30,70]. X_nn_train is normalized to the range [0,1]
                X_train = params.sample_x(n_start)      # Randomly sample input space to get initial data set
                X_nn_train = params.to_nn_input(X_train)
                Y_train = obj_fun(prob_fun(X_train))[:, np.newaxis]
                dm = data_manager.DataManager(X_nn_train, Y_train, n_batch)     # Holds data for mini-batches
                model = nn.choose_model(**nn_args, dm=dm, results_dir=results_dir_i, print_loss=False, opt_name="adam",
                                        n_units=n_units)

                def f(samples):
                    """Return samples from the posterior distribution of predictions"""
                    return model.predict_posterior(sess, samples, dm, n=30)

                init = tf.global_variables_initializer()
                sess.run(init)
                start_time = time.time()

                # Best candidate data point so far
                y_best_i = np.argmax(Y_train, axis=0)
                x_best = X_train[y_best_i, :]
                y_best = np.max(Y_train)

                n_data_list = []     # Keep track of size of training dataset
                y_best_list = []    # Keep track of the best value of objective function

                for i in range(n_train):
                    if i % iter_restart_training == 0:
                        model.reset(sess)  # Retrain the model from scratch after we collect each point
                        epochs_i = n_epochs
                        cycle_i = lr_cycle_base
                    else:   # Continue training
                        cycle_i = lr_cycle
                        epochs_i = n_epochs_continue
                    # Training step
                    loss_final = model.train(sess, epochs_i, dm, X_val=X_nn_train, Y_val=Y_train, print_loss=False,
                                             save_model=False, cycle=cycle_i)

                    # Random set of unlabelled x points - we will use Bayesian optimization to choose which one to label
                    x_sample = params.sample_x(int(af_m))
                    x_nn_sample = params.to_nn_input(x_sample)
                    # x_new, x_nn_new = bo.ei_direct_im(x_sample, x_nn_sample, f, y_best)
                    if nn_args['uncertainty'] == 'bbb' or nn_args['uncertainty'] == 'mnf' or \
                            nn_args['uncertainty'] == 'ensemble':
                        i_new, x_nn_new = bo.ei_mc(x_nn_sample, f, y_best, batch_size=int(2**13))
                        i_new = [i_new]     # Temporary hack to adjust for dimension
                    else:
                        i_new, x_nn_new = bo.ei_direct(x_nn_sample, f, y_best)
                    x_new = x_sample[i_new]

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
                    n_data_list.append(dm.n)
                    y_best_list.append(y_best)
                    print("Trained with %d data points. Best value=%f" % (dm.n, y_best))
                    np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_list, best_y=y_best_list, best_x=x_best,
                             best_x_nn=x_nn_best)
                time_tot = time.time() - start_time
                print("Took %f seconds" % time_tot)
                np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_list, best_y=y_best_list,
                         time_tot=time_tot, best_x=x_best, best_x_nn=x_nn_best)
                print(x_best)
                print(y_best)
    elif opt == "nn2":
        # Bayesian optimization using Bayesian neural networks with continued training
        # The NN predicts the spectra, not the objective function. So we have an intermediate variable
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        import tensorflow as tf

        n_units = [n_units] * n_layers

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for _ in range(trials):
                results_dir_i, _ = get_trial_dir(os.path.join(results_dir, 'trial%d'), i0=trial_i)

                # Random initialization of initial dataset
                X_train = params.sample_x(n_start)
                X_nn_train = params.to_nn_input(X_train)
                Z_train = prob_fun(X_train)  # Calculate Mie scattering
                Y_train = obj_fun(Z_train)

                dm = data_manager.DataManager(X_nn_train, Z_train, n_batch)
                model = nn.choose_model(**nn_args, dm=dm, results_dir=results_dir_i,
                                        print_loss=False, opt_name="adam",
                                        n_units=n_units)

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

                n_data_list = []
                y_best_list = []
                t_list = []

                for i in range(n_train):
                    t0 = time.time()

                    if i % iter_restart_training == 0:
                        model.reset(sess)  # Retrain the model from scratch after we collect each point
                        epochs_i = n_epochs
                        cycle_i = lr_cycle_base
                    else:   # Continue training
                        epochs_i = n_epochs_continue
                        cycle_i = lr_cycle

                    # Training step
                    _ = model.train(sess, epochs_i, dm, print_loss=False,
                                    save_model=False, cycle=cycle_i)

                    # Random set of unlabelled x points - we will use Bayesian optimization to choose which one to label
                    x_sample = params.sample_x(int(af_m))
                    x_nn_sample = params.to_nn_input(x_sample)
                    i_new, x_nn_new = bo.ei_mc(x_nn_sample, f, y_best, obj_fun, batch_size=int(2**13))  # 8192
                    x_new = x_sample[[i_new]]

                    t1 = time.time()
                    z_new = prob_fun(x_new)
                    t2 = time.time()
                    # y_new = obj_fun(z_new)    # We don't need to calculate this

                    # Add the labelled data point to our training data set
                    X_train = np.vstack((X_train, x_new))
                    X_nn_train = np.vstack((X_nn_train, x_nn_new))
                    Z_train = np.vstack((Z_train, z_new))
                    dm.add_data(x_nn_new, z_new)

                    # Update the best data point so far
                    i_best = np.argmax(obj_fun(Z_train))
                    x_best = X_train[i_best]
                    x_nn_best = X_nn_train[i_best]
                    y_best = obj_fun(Z_train)[i_best]
                    z_best = Z_train[i_best]

                    t3 = time.time()
                    ti = t3 - t0 - (t2 - t1)
                    t_list.append(ti)

                    # Save results to a file
                    n_data_list.append(dm.n)
                    y_best_list.append(y_best)
                    print("Trained with %d data points. Best value=%f" % (dm.n, y_best))
                    np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_list, best_y=y_best_list,
                             best_x=x_best, best_x_nn=x_nn_best, z_best=z_best, t_list=t_list)
                    # np.savez(os.path.join(results_dir_i, f'loss_n{i}'), train_loss=train_loss, val_loss=val_loss,)
                time_tot = time.time() - start_time
                print("Took %f seconds" % time_tot)
                np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_list, best_y=y_best_list,
                         time_tot=time_tot, best_x=x_best, best_x_nn=x_nn_best, z_best=z_best,
                         t_list=t_list)
                print(x_best)
                print(y_best)
                # saver = tf.train.Saver()
                # saver.save(sess, os.path.join(results_dir_i, 'model'))
                np.savez(os.path.join(results_dir_i, 'data'), n_data=n_data_list, X_train=X_train,
                         X_nn_train=X_nn_train, Z_train=Z_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize nanoparticle scattering")
    parser.add_argument("--results-dir", type=str, default='results/opt/test')
    parser.add_argument("--n-start", type=int, default=5, help='Size of initial dataset')
    parser.add_argument("--n-batch", type=int, default=10)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-epochs-continue", type=int, default=10)
    parser.add_argument("--iter-restart-training", type=int, default=100)
    parser.add_argument("--n_train", type=int, default=1000)
    # Weighted training for nn when adding new data point
    parser.add_argument('--lr-cycle', dest='lr_cycle', action='store_true')
    parser.add_argument('--no-lr-cycle', dest='lr_cycle', action='store_false')
    parser.set_defaults(lr_cycle=False)
    parser.add_argument('--lr-cycle-base', dest='lr_cycle', action='store_true')
    parser.add_argument('--no-lr-cycle-base', dest='lr_cycle', action='store_false')
    parser.set_defaults(lr_cycle_base=False)
    # Optional arguments for size of neural network
    parser.add_argument("--n-units", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    # Optimization
    parser.add_argument("--opt", type=str, default="random",
                        choices=["random", "gp", 'gp2', "nn", "dlib", 'nlopt', "nn2", 'cma'],
                        help="Model for optimization")
    parser.add_argument('--af-n', type=int, default=30, help='Number of times to sample Bayesian model')
    parser.add_argument("--acquisition", type=str, default="EI",
                        choices=["EI", "MPI", "LCB"],
                        help="Acquisition function to label a new point")
    parser.add_argument('--af-m', type=int, default=int(1e5), help='Number of data points to sample for acquisition')
    parser.add_argument("--objective", type=str, default="orange")
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
