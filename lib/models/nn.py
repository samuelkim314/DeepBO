"""Collection of neural network models for estimating prediction uncertainty"""
import argparse
from lib.models import ensemble, bbb, nn_base
import tensorflow as tf


def choose_model(uncertainty, dm, results_dir=None, drop_rate=0.0, lr=1e-4, lr_decay=0.9, n_ensemble=10,
                 print_loss=False, opt_name="adam", activation=None,
                 bbb_sigma1=3.6, bbb_sigma2=0.1, bbb_pi=0.25, bbb_sigma=0.001,
                 n_units=None, n_channels=None, periodic=True, ub=None, obj_fun=lambda x: x):
    """Return an initialized model corresponding to the named Bayesian neural network along with its hyper-parameters"""
    uncertainty = uncertainty.lower()
    optimizer, lr_ph = get_optimizer(opt_name, lr, lr_decay)
    activation = get_activation(uncertainty, activation)

    # Model for predicting uncertainty
    if uncertainty == "dropout":
        model = nn_base.Dropout(dm=dm, results_dir=results_dir, drop_rate=drop_rate, print_loss=print_loss,
                                optimizer=optimizer, hidden_units=n_units)
    elif uncertainty == "dropout_conv":
        model = nn_base.DropoutConv(dm=dm, results_dir=results_dir,
                                    drop_rate=drop_rate, print_loss=print_loss,
                                    hidden_units=n_units, n_channels=n_channels, optimizer=optimizer)
    elif uncertainty == "ensemble":
        model = ensemble.Ensemble(dm=dm, results_dir=results_dir, n_networks=n_ensemble, print_loss=print_loss,
                                  optimizer=optimizer, hidden_units=n_units, lr_ph=lr_ph, lr=lr)
    elif uncertainty == "conv_ensemble":
        model = ensemble.ConvEnsemble(dm=dm, results_dir=results_dir, n_networks=n_ensemble, print_loss=print_loss,
                                      optimizer=optimizer, n_channels=n_channels, hidden_units=n_units,
                                      lr_ph=lr_ph, lr=lr)
    elif uncertainty == "bbb":
        model = bbb.BayesByBackprop(dm=dm, results_dir=results_dir, print_loss=print_loss,
                                    activation=activation,
                                    prior_sigma1=bbb_sigma1, prior_sigma2=bbb_sigma2, prior_pi=bbb_pi,
                                    hidden_units=n_units, optimizer=optimizer, ub=ub, lr_ph=lr_ph, lr=lr)
    elif uncertainty == "bbb_conv":
        model = bbb.BayesByBackpropConv(dm=dm, results_dir=results_dir, print_loss=print_loss, optimizer=optimizer,
                                        prior_sigma1=bbb_sigma1, prior_sigma2=bbb_sigma2, prior_pi=bbb_pi,
                                        data_sigma=bbb_sigma, hidden_units=n_units, n_channels=n_channels,
                                        periodic=periodic, lr_ph=lr_ph, lr=lr)
    elif uncertainty == "neurallinear":
        model = nn_base.NeuralLinear(dm=dm, results_dir=results_dir, print_loss=print_loss, optimizer=optimizer,
                                     hidden_units=n_units, lr_ph=lr_ph, lr=lr)
    elif uncertainty == "conv_neurallinear":
        model = nn_base.ConvNeuralLinear(dm=dm, results_dir=results_dir, print_loss=print_loss, optimizer=optimizer,
                                     hidden_units=n_units, n_channels=n_channels, lr_ph=lr_ph, lr=lr)
    elif uncertainty == 'single':
        model = nn_base.Single(dm=dm, results_dir=results_dir, print_loss=print_loss, optimizer=optimizer,
                               hidden_units=n_units, obj_fun=obj_fun)
    else:
        raise ValueError("Could not find a model with that name.")
    return model


def get_optimizer(opt_name, lr, lr_decay):
    lr = tf.placeholder_with_default(lr, [])

    if opt_name == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=lr_decay)
    elif opt_name == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=lr_decay)
    elif opt_name == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    else:
        raise ValueError('Invalid optimizer name')
    return optimizer, lr


def get_activation(uncertainty, activation_name):
    if activation_name == 'relu':
        return tf.nn.relu
    elif activation_name == 'tanh':
        return tf.nn.tanh
    elif activation_name is None and uncertainty == 'neurallinear':     # Default for neurallinear is tanh
        return tf.nn.tanh
    else:   # Default for everything else is relu
        return tf.nn.relu


def is_mc(model):
    analytical_models = ['neurallinear']
    if model == 'neurallinear':
        return False
    else:
        return True


def add_args(parser):
    """Add arguments controlling hyper-parameters of Bayesian neural network models"""
    parser.add_argument("--uncertainty", type=str, default="dropout",
                        choices=["dropout", "ensemble", "bbb", "neurallinear", "bbb_conv", 'dropout_conv',
                                 'mnf', 'gnn_ensemble', 'single', 'gnn_bbb', 'conv_ensemble', 'conv_neurallinear',
                                 'graph_neurallinear'],
                        help="Model architecture to estimate prediction uncertainty")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr-decay", type=float, default=.9, help="Learning rate decay")
    # parser.add_argument("--n-layers", type=int, default=12, help="Number of hidden layers")
    # parser.add_argument("--n-neurons", type=int, default=100)
    parser.add_argument("--drop-rate", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--n-ensemble", type=int, default=10, help="Size of ensemble")
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh'])
    # Bayes by Backprop (BBB) hyperparameters
    parser.add_argument("--bbb-sigma1", type=float, default=3.6)
    parser.add_argument("--bbb-sigma2", type=float, default=0.12)
    parser.add_argument("--bbb-pi", type=float, default=0.25)
    parser.add_argument("--bbb-sigma", type=float, default=1e-3)
    return parser


def process_args(kwargs):
    nn_keys = ['uncertainty', 'lr', 'lr_decay', 'drop_rate', 'n_ensemble', 'activation',
               'bbb_sigma1', 'bbb_sigma2', 'bbb_pi', 'bbb_sigma', 'anneal']
    nn_args = {}
    args = {}
    for k, v in kwargs.items():
        if k in nn_keys:
            nn_args[k] = v
        else:
            args[k] = v
    return args, nn_args


