"""
Collection of acquisition functions for Bayesian optimization to find what is the next point that should be labeled.
"""
import numpy as np
from scipy.stats import norm
import warnings


def ei(x, f, y_best, obj_fun=lambda x: x[:, 0, :]):
    """Expected improvement (EI)"""
    y_pred = obj_fun(f(x))
    y_mean = np.mean(y_pred, axis=1)
    y_std = np.std(y_pred, axis=1)
    Z = (y_mean - y_best)/y_std
    prob = (y_mean - y_best) * norm.cdf(Z) + y_std * norm.pdf(Z)
    i_max = [np.argmax(prob)]
    return x[i_max, :]


def ei_direct_batch(x, f_surrogate, y_best):
    """Expected improvement (EI), where the surrogate model predicts mean and standard deviation"""
    y_mean, y_std = f_surrogate(x)
    y_mean = y_mean[:, 0]
    y_std = y_std[:, 0]
    prob = (y_mean - y_best) * norm.cdf((y_mean - y_best) / y_std) + \
           y_std * norm.pdf((y_mean - y_best) / y_std)
    i_max = np.nanargmax(prob)
    return i_max, x[i_max], prob[i_max]


def ei_direct(x, f_surrogate, y_best, batch_size=32):
    """Expected improvement (EI), where the surrogate model predicts mean and standard deviation. Inference is split up
    into batches to avoid overflowing memory

    f_surrogate is a function that returns mean and standard deviation of predictions as (n, 1) arrays"""
    n = x.shape[0]
    batch_ind = range(0, n, batch_size)

    i_max_list = []
    x_max_list = []
    y_max_list = []
    for j_batch in batch_ind:
        i_max_j, x_max_j, y_max_j = ei_direct_batch(x[j_batch:j_batch + batch_size], f_surrogate, y_best)
        i_max_list.append(i_max_j)
        x_max_list.append(x_max_j)
        y_max_list.append(y_max_j)
    j_max = np.argmax(y_max_list)
    return j_max * batch_size + i_max_list[j_max], x_max_list[j_max]


def ei_mc_batch(x, f, y_best, obj_fun=lambda x: x[:, 0, :]):
    """Expected improvement (EI) acquisition function, using a Monte Carlo approximation.
    Arguments:
        x: input points to sample, shape (n_batch, x_dim)
        f: surrogate function that takes x and produces probabilistic predictions, shape (n_batch, y_dim, n_sample)
        y_best: scalar value of the best objective functon value so far
        obj_fun: function that converts model predictions to the objective function, produces shape (n_batch, n_sample)
    Return:
        i_max: index in x that maximizes the objective function
        x_new: x, shape (1, x_dim)
        y_new: new y, shape(1)?
    """
    y_pred = obj_fun(f(x))
    assert y_pred.ndim == 2     # (n_data, n_sample)
    scores = np.sum(np.maximum(y_pred - y_best, 0), axis=1)
    i_max = np.argmax(scores)
    return i_max, x[[i_max]], scores[i_max]


def ei_mc(x, f, y_best, obj_fun=lambda x: x[:, 0, :], batch_size=1024):
    """Expected improvement (EI) acquisition function, using a Monte Carlo approximation
        Arguments:
            x: input points to sample, shape (n_batch, x_dim)
            f: surrogate function that takes x and returns probabilistic predictions, shape (n_batch, y_dim, n_sample)
            y_best: scalar value of the best objective function value so far
            obj_fun: function that takes y and returns objective function, shape (n_batch, n_sample)
            batch_size: size of batch, useful so that memory is not overrun
        Return:
            i_max: index in x that maximizes the objective function
            x_new: x, shape (1, x_dim)
        """
    n = x.shape[0]
    batch_ind = range(0, n, batch_size)
    i_max_list = []
    x_max_list = []
    y_max_list = []
    for j_batch in batch_ind:
        i_max_j, x_max_j, y_max_j = ei_mc_batch(x[j_batch:j_batch + batch_size], f, y_best, obj_fun)
        i_max_list.append(i_max_j)
        x_max_list.append(x_max_j)
        y_max_list.append(y_max_j)
    j_max = np.argmax(y_max_list)
    return j_max * batch_size + i_max_list[j_max], x_max_list[j_max]


def ei_mc_chem_batch(x, a, e, f, y_best, obj_fun=lambda x: x[:, 0, :]):
    """Expected improvement (EI) acquisition function, using a Monte Carlo approximation. Single batch
    Arguments:
        x: input points to sample, shape (n_batch, x_dim)
        f: surrogate function that takes x and produces probabilistic predictions, shape (n_batch, y_dim, n_sample)
        y_best: scalar value of the best objective functon value so far
        obj_fun: function that converts model predictions to the objective function, produces shape (n_batch, n_sample)
    Return:
        i_max: index in x that maximizes the objective function
        x_new: x, shape (1, x_dim)
        y_new: new y, scalar
    """
    y_pred = obj_fun(f(x, a, e))
    assert y_pred.ndim == 2     # (n_data, n_sample)
    scores = np.sum(np.maximum(y_pred - y_best, 0), axis=1)     # shape (n_data)
    i_max = np.argmax(scores)
    return i_max, x[[i_max]], scores[i_max]   # x has shape (1, x_dim)


def ei_mc_chem(x, a, e, f, y_best, obj_fun=lambda x: x[:, 0, :], batch_size=32):
    """Expected improvement (EI) acquisition function, using a Monte Carlo approximation
        Arguments:
            x: input points to sample, shape (n_batch, x_dim)
            f: surrogate function that takes x and returns probabilistic predictions, shape (n_batch, y_dim, n_sample)
            y_best: scalar value of the best objective function value so far
            obj_fun: function that takes y and returns objective function, shape (n_batch, n_sample)
            batch_size: size of batch, useful so that memory is not overrun
        Return:
            i_max: index in x that maximizes the objective function
            x_new: x, shape (1, x_dim)
        """
    n = x.shape[0]
    batch_ind = range(0, n, batch_size)
    i_max_list = []
    x_max_list = []
    y_max_list = []
    for j_batch in batch_ind:
        i_max_j, x_max_j, y_max_j = ei_mc_chem_batch(x[j_batch:j_batch + batch_size], a[j_batch:j_batch + batch_size],
                                                     e[j_batch:j_batch + batch_size], f, y_best, obj_fun)
        i_max_list.append(i_max_j)
        x_max_list.append(x_max_j)
        y_max_list.append(y_max_j)
    j_max = np.argmax(y_max_list)
    return j_max * batch_size + i_max_list[j_max], x_max_list[j_max]


def ei_direct_chem_batch(x, a, e, f, y_best):
    """Expected improvement (EI) acquisition function, analytical. Single batch
    Arguments:
        x: input points to sample, shape (n_batch, x_dim)
        f: surrogate function that takes x and produces probabilistic predictions, shape (n_batch, y_dim, n_sample)
        y_best: scalar value of the best objective functon value so far
        obj_fun: function that converts model predictions to the objective function, produces shape (n_batch, n_sample)
    Return:
        i_max: index in x that maximizes the objective function
        x_new: x, shape (1, x_dim)
        y_new: new y, scalar
    """
    y_mean, y_std = f(x, a, e)
    y_mean = y_mean[:, 0]
    y_std = y_std[:, 0]
    prob = (y_mean - y_best) * norm.cdf((y_mean - y_best) / y_std) + \
           y_std * norm.pdf((y_mean - y_best) / y_std)

    i_max = np.argmax(prob)
    return i_max, x[[i_max]], prob[i_max]   # x has shape (1, x_dim)


def ei_direct_chem(x, a, e, f, y_best, batch_size=32):
    """Expected improvement (EI) acquisition function, analytical
        Arguments:
            x: input points to sample, shape (n_batch, x_dim)
            f: surrogate function that takes x and returns probabilistic predictions, shape (n_batch, y_dim, n_sample)
            y_best: scalar value of the best objective function value so far
            obj_fun: function that takes y and returns objective function, shape (n_batch, n_sample)
            batch_size: size of batch, useful so that memory is not overrun
        Return:
            i_max: index in x that maximizes the objective function
            x_new: x, shape (1, x_dim)
        """
    n = x.shape[0]
    batch_ind = range(0, n, batch_size)
    i_max_list = []
    x_max_list = []
    y_max_list = []
    for j_batch in batch_ind:
        i_max_j, x_max_j, y_max_j = ei_direct_chem_batch(x[j_batch:j_batch + batch_size], a[j_batch:j_batch + batch_size],
                                                     e[j_batch:j_batch + batch_size], f, y_best)
        i_max_list.append(i_max_j)
        x_max_list.append(x_max_j)
        y_max_list.append(y_max_j)
    j_max = np.argmax(y_max_list)
    return j_max * batch_size + i_max_list[j_max], x_max_list[j_max]


if __name__ == '__main__':
    x_sample = np.random.rand(50, 1)
    print(x_sample)

    def f_noise(x):
        return x[:, :, np.newaxis] + np.random.normal(0, 0.1, size=(x.shape[0], 1, 5))

    y_best = 0.5

    def obj_fun(x):
        return x[:, 0, :]

    print(ei(x_sample, f_noise, y_best, obj_fun))
