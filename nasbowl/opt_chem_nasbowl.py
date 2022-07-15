import argparse
import os
import pickle
import time
import pandas as pd
import tabulate

import bayesopt
import kernels
from bayesopt.generate_test_graphs import *
# from benchmarks import NAS101Cifar10, NAS201
from kernels import *

from spektral.datasets import qm9
# from spektral.utils import label_to_one_hot
from norm_chem import get_objective_y
# from lib import data_manager
import json

import numpy as np

parser = argparse.ArgumentParser(description='NAS-BOWL')
parser.add_argument('--dataset', default='qm9', help='The benchmark dataset to run the experiments. '
                                                             'options = ["nasbench101", "nasbench201"].')
parser.add_argument('--task', default=['cifar10-valid'],
                    nargs="+", help='the benchmark task *for nasbench201 only*.')
parser.add_argument("--use_12_epochs_result", action='store_true',
                    help='Whether to use the statistics at the end of the 12th epoch, instead of using the final '
                         'statistics *for nasbench201 only*')
parser.add_argument('--n_repeat', type=int, default=1, help='number of repeats of experiments')
parser.add_argument("--data_path", default='data/')
parser.add_argument('--n_init', type=int, default=5, help='number of initialising points')
parser.add_argument("--max_iters", type=int, default=100, help='number of maximum iterations')
parser.add_argument('--pool_size', type=int, default=10000, help='number of candidates generated at each iteration')
parser.add_argument('--mutate_size', type=int, help='number of mutation candidates. By default, half of the pool_size '
                                                    'is generated from mutation.')
parser.add_argument('--pool_strategy', default='random', help='the pool generation strategy. Options: random,'
                                                              'mutate')
parser.add_argument('--save_path', default='results/', help='path to save log file')
parser.add_argument('-s', '--strategy', default='gbo', help='optimisation strategy: option: gbo (graph bo), '
                                                            'random (random search)')
parser.add_argument('-a', "--acquisition", default='EI', help='the acquisition function for the BO algorithm. option: '
                                                              'UCB, EI, AEI')
parser.add_argument('-k', '--kernels', default=['wl'],
                    nargs="+",
                    help='graph kernel to use. This can take multiple input arguments, and '
                         'the weights between the kernels will be automatically determined'
                         ' during optimisation (weights will be deemed as additional '
                         'hyper-parameters.')
parser.add_argument('-p', '--plot', action='store_true', help='whether to plot the procedure each iteration.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of samples to evaluate')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--cuda', action='store_true', help='Whether to use GPU acceleration')
parser.add_argument('--fixed_query_seed', type=int, default=None,
                    help='Whether to use deterministic objective function as NAS-Bench-101 has 3 different seeds for '
                         'validation and test accuracies. Options in [None, 0, 1, 2]. If None the query will be '
                         'random.')
parser.add_argument('--load_from_cache', action='store_true', help='Whether to load the pickle of the dataset. ')
parser.add_argument('--mutate_unpruned_archs', action='store_true',
                    help='Whether to mutate on the unpruned archs. This option is only valid if mutate '
                         'is specified as the pool_strategy')
parser.add_argument('--no_isomorphism', action='store_true', help='Whether to allow mutation to return'
                                                                  'isomorphic architectures')
parser.add_argument('--maximum_noise', default=0.01, type=float, help='The maximum amount of GP jitter noise variance')

parser.add_argument("--results-dir", type=str, default='results/opt/test')
parser.add_argument("--objective", type=str, default="gap",
                        choices=['gap', 'cv', 'alpha', 'mingap', 'mincv', 'minalpha', 'gap0.1', 'gapalpha',
                                 'mingapalpha'])
parser.add_argument('--trial-i', type=int, default=0)

args = parser.parse_args()
options = vars(args)
print(options)

if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

if args.cuda and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

assert args.strategy in ['random', 'gbo']
assert args.pool_strategy in ['random', 'mutate', ]

# Initialise the objective function. Negative ensures a maximisation task that is assumed by the acquisition function.

# Persistent data structure...
cache_path = 'data/' + args.dataset + '.pickle'

o = None
if args.load_from_cache:
    if os.path.exists(cache_path):
        try:
            o = pickle.load(open(cache_path, 'rb'))
            o.seed = args.fixed_query_seed
            if args.dataset == 'nasbench201':
                o.task = args.task[0]
                o.use_12_epochs_result = args.use_12_epochs_result
        except:
            pass


# Load and process data
x_data, y = qm9.load_data(return_type='networkx', amount=None)  # Set to None to train on whole dataset
z, obj_fun = get_objective_y(y, args.objective)
y_data = obj_fun(z)[:, np.newaxis]
ind = np.arange(y_data.shape[0])  # Indexing the original data - convenient IDs

if not os.path.exists(args.results_dir):
    try:
        os.makedirs(args.results_dir)
    except FileExistsError:
        pass
meta = open(os.path.join(args.results_dir, 'meta.txt'), 'a')
meta.write(json.dumps(args.results_dir))
meta.close()


def get_trial_dir(dir_format, i0=0):
    i = i0
    while True:
        results_dir_i = dir_format % i
        if os.path.isdir(results_dir_i):
            i += 1
        else:
            os.makedirs(results_dir_i)
            break
    return results_dir_i, i


# print(x_pool[10].nodes(data=True))
# print(x_pool[10].edges(data=True))
#
# quit()

"""[(0, {'index': 0, 'coords': array([-0.0029,  1.5099,  0.0087]), 'atomic_num': 6, 'iso': 0, 'charge': 0, 'info': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}), (1, {'index': 1, 'coords': array([ 0.0261,  0.0033, -0.0375]), 'atomic_num': 6, 'iso': 0, 'charge': 0, 'info': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}), (2, {'index': 2, 'coords': array([ 0.9423, -0.6551, -0.4568]), 'atomic_num': 8, 'iso': 0, 'charge': 0, 'info': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])})]
[(0, 1, {'start_atom': 0, 'end_atom': 1, 'type': 1, 'stereo': 0, 'info': array([0, 0, 0])}), (1, 2, {'start_atom': 1, 'end_atom': 2, 'type': 2, 'stereo': 0, 'info': array([0, 0, 0])})]
"""

all_data = []
for j in range(args.n_repeat):
    results_dir_i, _ = get_trial_dir(os.path.join(args.results_dir, 'trial%d'), i0=args.trial_i + j)
    n_data_arr = []
    best_y_arr = []
    time_arr = []

    start_time = time.time()
    best_tests = []
    best_vals = []
    # 2. Take n_init_point random samples from the candidate points to warm_start the Bayesian Optimisation
    ind_train = np.random.choice(range(y.shape[0]), size=args.n_init, replace=False)
    x = [x_data[i] for i in ind_train]
    y = y_data[ind_train]

    ind_pool = np.delete(ind, ind_train, 0)
    pool = [x_data[i] for i in ind_pool]
    y_pool = [y_data[i] for i in ind_pool]

    # x, x_config, x_unpruned = random_sampling(args.n_init, benchmark=args.dataset, save_config=True,
    #                                           return_unpruned_archs=True)
    # y_np_list = [o.eval(x_) for x_ in x]
    y = torch.tensor([yi[0] for yi in y]).float()
    # train_details = [y[1] for y in y_np_list]

    # Initialise the GP surrogate and the acquisition function
    kern = []

    for k in args.kernels:
        # Graph kernels
        if k == 'wl':
            k = WeisfilerLehman(h=2, oa=args.dataset != 'nasbench201', node_label='atomic_num', edge_label='type')
        elif k == 'mlk':
            k = MultiscaleLaplacian(n=1)
        elif k == 'vh':
            k = WeisfilerLehman(h=0, oa=args.dataset != 'nasbench201',)
        else:
            try:
                k = getattr(kernels, k)
                k = k()
            except AttributeError:
                logging.warning('Kernel type ' + str(k) + ' is not understood. Skipped.')
                continue
        kern.append(k)
    if kern is None:
        raise ValueError("None of the kernels entered is valid. Quitting.")
    if args.strategy != 'random':
        gp = bayesopt.GraphGP(x, y, kern, verbose=args.verbose)
        gp.fit(wl_subtree_candidates=(0,) if args.kernels[0] == 'vh' else tuple(range(1, 4)),
               optimize_lik=args.fixed_query_seed is None,
               max_lik=args.maximum_noise)
    else:
        gp = None

    # 3. Main optimisation loop
    columns = ['Iteration', 'Last func val', 'Best func val', 'TrainTime']

    res = pd.DataFrame(np.nan, index=range(args.max_iters), columns=columns)
    sampled_idx = []

    for i in range(args.max_iters):
        # Generate a pool of candidates from a pre-specified strategy
        # if args.pool_strategy == 'random':
        #     pool, _, unpruned_pool = random_sampling(args.pool_size, benchmark=args.dataset, return_unpruned_archs=True)
        # else:
        #     pass

        time1 = time.time()

        i_temp = None
        if args.pool_size > 0:
            i_temp = np.random.choice(range(len(ind_pool)), args.pool_size, replace=False)
            ind_subpool = ind_pool[i_temp]
            subpool = [pool[ii] for ii in i_temp]
            y_subpool = [y_pool[ii] for ii in i_temp]
        else:
            ind_subpool = ind_pool
            subpool = pool
            y_subpool = y_pool

        if args.strategy != 'random':
            if args.acquisition == 'UCB':
                a = bayesopt.GraphUpperConfidentBound(gp)
            elif args.acquisition == 'EI':
                a = bayesopt.GraphExpectedImprovement(gp, in_fill='best', augmented_ei=False)
            elif args.acquisition == 'AEI':
                # Uses the augmented EI heuristic and changed the in-fill criterion to the best test location with
                # the highest *posterior mean*, which are preferred when the optimisation is noisy.
                a = bayesopt.GraphExpectedImprovement(gp, in_fill='posterior', augmented_ei=True)
            else:
                raise ValueError("Acquisition function" + str(args.acquisition) + ' is not understood!')
        else:
            a = None

        # Ask for a location proposal from the acquisition function
        if args.strategy == 'random':
            next_x = random.sample(subpool, args.batch_size)
            sampled_idx.append(next_x)
            next_x_unpruned = None
        else:
            next_x, eis, indices = a.propose_location(top_n=args.batch_size, candidates=subpool)
            # next_x_unpruned = [unpruned_pool[i] for i in indices]
        # Evaluate this location from the objective function
        y_new = [y_subpool[k] for k in indices]
        next_y = [y_new[0]]

        if args.pool_size == 0:
            ind_pool = np.delete(ind_pool, indices[0], 0)
        else:
            ind_pool = np.delete(ind_pool, i_temp[indices[0]], 0)
        pool = [x_data[k] for k in ind_pool]
        y_pool = [y_data[k] for k in ind_pool]

        # Evaluate all candidates in the pool to obtain the regret (i.e. true best *in the pool* compared to the one
        # returned by the Bayesian optimiser proposal)
        # pool_vals = [o.eval(x_)[0] for x_ in pool]
        # if gp is not None:
        #     pool_preds = gp.predict(pool,)
        #     pool_preds = [p.detach().cpu().numpy() for p in pool_preds]
        #     pool.extend(next_x)

        # Update the GP Surrogate
        x.extend(next_x)
        # if args.pool_strategy in ['mutate']:
        #     x_unpruned.extend(next_x_unpruned)
        y = torch.cat((y, torch.tensor(next_y).view(-1))).float()
        # test = torch.cat((test, torch.tensor(next_test).view(-1))).float()

        if args.strategy != 'random':
            gp.reset_XY(x, y)
            gp.fit(wl_subtree_candidates=(0,) if args.kernels[0] == 'vh' else tuple(range(1, 4)),
                   optimize_lik=args.fixed_query_seed is None,
                   max_lik=args.maximum_noise
                   )

            # Compute the GP posterior distribution on the trainning inputs
            train_preds = gp.predict(x,)
            train_preds = [t.detach().cpu().numpy() for t in train_preds]

        # Updating evaluation metrics
        best_val = torch.max(y)
        best_y_arr.append(best_val.item())

        end_time = time.time()
        # Compute the cumulative training time.
        values = [str(i), str(np.max(next_y)), best_val.item(),
                  str(end_time - start_time),]
        table = tabulate.tabulate([values], headers=columns, tablefmt='simple', floatfmt='8.4f')
        best_vals.append(best_val)
        n_data_arr.append(i+1+args.n_init)

        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        if args.plot and args.strategy != 'random':
            import matplotlib.pyplot as plt

            plt.subplot(221)
            # True validation error vs GP posterior
            plt.title('Val')
            plt.plot(pool_vals, pool_vals, '.')
            plt.errorbar(pool_vals, pool_preds[0],
                         fmt='.', yerr=np.sqrt(np.diag(pool_preds[1])),
                         capsize=2, color='b', alpha=0.2)
            plt.grid(True)
            plt.subplot(222)
            # Acquisition function
            plt.title('Acquisition')
            plt.plot(pool_vals, eis, 'b+')
            plt.xlim([2.5, None])
            plt.subplot(223)
            plt.title('Train')

            y1, y2 = y[:-args.batch_size], y[-args.batch_size:]
            plt.plot(y, y, ".")
            plt.plot(y1, train_preds[0][:-args.batch_size], 'b+')
            plt.plot(y2, train_preds[0][-args.batch_size:], 'r+')

            if args.verbose:
                from perf_metrics import *
                print('Spearman: ', spearman(pool_vals, pool_preds[0]))
            plt.subplot(224)
            # Best metrics so far
            xaxis = np.arange(len(best_tests))
            plt.plot(xaxis, best_tests, "-.", c='C1', label='Best test so far')
            plt.plot(xaxis, best_vals, "-.", c='C2', label='Best validation so far')
            plt.legend()
            plt.show()

        time_arr.append(end_time - time1)

        res.iloc[i, :] = values
        np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=best_y_arr,
                 time_tot=end_time - start_time, ind_train=ind_train, time_list=time_arr)
    all_data.append(res)

    np.savez(os.path.join(results_dir_i, 'best'), n_data=n_data_arr, best_y=best_y_arr,
             time_tot=time.time() - start_time, ind_train=ind_train, time_list=time_arr)

if args.save_path is not None:
    import datetime
    time_string = datetime.datetime.now()
    time_string = time_string.strftime('%Y%m%d_%H%M%S')
    args.save_path = os.path.join(args.save_path, time_string)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    pickle.dump(all_data, open(args.save_path + '/data.pickle', 'wb'))
    option_file = open(args.save_path + "/command.txt", "w+")
    option_file.write(str(options))
    option_file.close()
