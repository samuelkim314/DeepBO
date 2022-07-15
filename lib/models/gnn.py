"""Implement BNNs using GNNs provided by spektral library"""
from keras import activations, initializers, regularizers, constraints, backend as K
from lib import data_manager
from lib.models import bbb
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense
from spektral.layers import EdgeConditionedConv, GlobalAvgPool, ops
import sklearn.linear_model


class BaseGNN:
    """
    Base class for Bayesian graph neural network
    """
    def __init__(self, dm, optimizer=None, lr_ph=None, lr=1e-4):
        """
        Initialize and build the network.
        Note that if you want to extend this class, this initializer builds the network. So define all your variables
        used in build() before calling this constructor.
        """
        # Model definition
        self.X_in = Input(shape=(dm.N, dm.F))
        self.A_in = Input(shape=(dm.N, dm.N))
        self.E_in = Input(shape=(dm.N, dm.N, dm.S))
        self.y = tf.placeholder("float", shape=[None, dm.y_dim], name="y")
        self.y_dim = dm.y_dim

        temp = set(tf.global_variables())
        self.yhat = self.build()
        self.vars = set(tf.global_variables()) - temp

        self.mse = tf.reduce_mean(tf.square(self.y - self.yhat))
        self.loss = tf.reduce_mean(self.mse)
        self.optimizer = optimizer
        self.opt_vars = self.optimizer.variables()
        self.minimizer = self.optimizer.minimize(self.loss)

        self.epoch = tf.placeholder_with_default(0, [])  # dummy variable
        self.lr = lr
        self.lr_ph = lr_ph

    def build(self):
        """Build the graph neural network - 4 graph convolutions layers, 4 fully-connected layers"""
        gc1 = EdgeConditionedConv(32, activation='relu')([self.X_in, self.A_in, self.E_in])
        gc = EdgeConditionedConv(32, activation='relu')([gc1, self.A_in, self.E_in])
        gc = EdgeConditionedConv(32, activation='relu')([gc, self.A_in, self.E_in])
        gc = EdgeConditionedConv(32, activation='relu')([gc, self.A_in, self.E_in])
        pool = GlobalAvgPool()(gc)
        h = Dense(32, activation=tf.nn.relu)(pool)
        h = Dense(32, activation=tf.nn.relu)(h)
        h = Dense(32, activation=tf.nn.relu)(h)
        h = Dense(32, activation=tf.nn.relu)(h)
        h = Dense(self.y_dim)(h)

        return h

    def calc_mse(self, sess, dm):
        """Calculate MSE by breaking up dataset into batches so we don't overrun GPU memory."""
        loss = []  # List of loss for each mini-batch
        weight = []  # Size of each mini-batch, used for calculating weighted average

        batcher = dm.batch_iter()
        steps = dm.steps
        for step in range(steps):
            x_batch, a_batch, e_batch, y_batch = next(batcher)
            mse_i = sess.run(self.mse, feed_dict={self.X_in: x_batch, self.A_in: a_batch, self.E_in: e_batch,
                                                  self.y: y_batch})
            loss.append(mse_i)
            weight.append(len(x_batch))
        return np.average(loss, weights=weight)

    def train_epoch(self, sess, batcher, steps, N=None, epoch=-1, lr=None):
        """Train a single epoch"""
        for step in range(steps):
            x_batch, a_batch, e_batch, y_batch = next(batcher)
            if lr is None:
                sess.run(self.minimizer, feed_dict={self.X_in: x_batch, self.A_in: a_batch, self.E_in: e_batch,
                                                    self.y: y_batch, self.epoch: epoch})
            else:
                sess.run(self.minimizer, feed_dict={self.X_in: x_batch, self.A_in: a_batch, self.E_in: e_batch,
                                                    self.y: y_batch, self.epoch: epoch, self.lr_ph: lr})

    def train(self, sess, epochs, dm, early_stopping=False, dm_new=None, anneal=False, cycle=False):
        """Train the neural network

        Args:
            sess : tensorflow session
            epochs : (int) number of epochs to train
            dm : data_manager object containing the dataset
            early_stopping : (bool) use early stopping - this can save time and prevent overfitting"""

        loss_best = 0  # Best validation loss so far
        patience = 10  # Hyper-parameter for early stopping - number of time the validation loss can increase before we
        # quit training
        worse_count = 0  # Counter for how many times validation loss increases in a row
        batcher = dm.batch_iter()  # Controller for mini-batches

        batcher_new = None
        if dm_new is not None:
            batcher_new = dm_new.batch_iter()

        for epoch in range(epochs):
            steps = dm.steps
            if anneal:
                epoch_i = epoch
            else:
                epoch_i = 1000

            if cycle:
                lr = self.lr / 2 * (np.cos((np.pi * ((epoch - 1) % epochs)) / epochs) + 1)
            else:
                lr = 1e-4

            if batcher_new is not None:
                self.train_epoch(sess, batcher_new, 1, dm.n, epoch=epoch_i, lr=lr)
            self.train_epoch(sess, batcher, steps, dm.n, epoch=epoch_i, lr=lr)  # Train a single epoch
            # if epoch % 100 == 99:
            if epoch % 10 == 0:
                # Calculate the training/validation loss
                train_loss = self.calc_mse(sess, dm)

                if early_stopping:
                    # Early stopping
                    if epoch < 10 or train_loss < loss_best:
                        loss_best = train_loss
                        worse_count = 0
                    else:
                        worse_count += 1
                    if worse_count > patience:
                        break

        return self.calc_mse(sess, dm)

    def predict(self, sess, X, A, E, sample=True):
        """Predict target value for X

        Args:
            sess: TensorFlow session
            X: matrix of x values
            sample:  bool to draw a stochastic sample from the posterior"""
        return sess.run(self.yhat, feed_dict={self.X_in: X, self.A_in: A, self.E_in: E})

    def predict_posterior(self, sess, X, A, E, dm, n=None):
        return self.predict(sess, X, A, E)[:, :, np.newaxis]


class Ensemble(BaseGNN):
    # Ensemble of fully-connected networks
    def __init__(self, dm, n_networks=10, optimizer=None, lr_ph=None, lr=1e-4):
        self.y_hat_list = []
        self.n_networks = n_networks

        super().__init__(dm, optimizer, lr_ph, lr)

        # Redefine mse and loss to average over the ensemble
        self.mse_list = [tf.reduce_mean(tf.square(self.y - yhat)) for yhat in self.y_hat_list]
        self.loss = tf.reduce_mean(self.mse_list)
        self.mse = self.loss
        self.minimizer = self.optimizer.minimize(self.loss)

    def build(self):
        y_hat = []
        for i in range(self.n_networks):
            y_hat.append(super().build())
        self.y_hat_list = y_hat
        return tf.reduce_mean(y_hat)

    def predict(self, sess, X, A, E, sample=True):
        """Predict target value for X

        Args:
            sess: TensorFlow session
            X: matrix of x values
            sample:  bool to draw a stochastic sample from the posterior"""
        if sample:
            i = np.random.randint(low=0, high=self.n_networks)
            return sess.run(self.y_hat_list[i], feed_dict={self.X_in: X, self.A_in: A, self.E_in: E})
        else:
            return sess.run(self.yhat, feed_dict={self.X_in: X, self.A_in: A, self.E_in: E})

    def predict_posterior(self, sess, X, A, E, dm, n=None):
        num_data = X.shape[0]
        preds = np.zeros([num_data, dm.y_dim, self.n_networks])

        for i in range(self.n_networks):
            preds[:, :, i] = sess.run(self.y_hat_list[i], feed_dict={self.X_in: X, self.A_in: A, self.E_in: E})

        return preds


class BBBFC(BaseGNN):
    """GNN with BBB only on fully-connected layers"""
    def __init__(self, dm, data_sigma=0.01, optimizer=None, lr_ph=None, lr=1e-4):
        self.data_sigma = data_sigma

        self.activation = tf.nn.relu
        self.conv_layers = []
        for i in range(4):
            # Switch this out between MAP and BBB versions
            # self.conv_layers.append(EdgeConditionedConvBBB(32, activation='relu'))
            self.conv_layers.append(EdgeConditionedConv(32, activation='relu'))
        self.n_units = [32, 32]
        self.layers = []
        for i in range(3):
            self.layers.append(bbb.BBBLayer())

        self.N = tf.placeholder_with_default(float(dm.n), shape=(), name="num_batches")

        super().__init__(dm, optimizer, lr_ph, lr)

        # MAP estimate. MLE is already built in super().__init__
        temp = set(tf.global_variables())
        self.yhat_map = self.build(sample=False, build_layer=False)
        self.vars.update(set(tf.global_variables()) - temp)

        self.mse = tf.reduce_mean(tf.square(self.y - self.yhat))
        self.loss = self.kl_loss(self.y, self.yhat, self.N)
        self.minimizer = self.optimizer.minimize(self.loss)

    def build(self, sample=True, build_layer=True):
        gc = self.X_in
        for layer in self.conv_layers:
            gc = layer([gc, self.A_in, self.E_in])

        pool = GlobalAvgPool()(gc)

        h = pool

        # Iterate over the hidden layers
        for i, (layer, out_dim) in enumerate(zip(self.layers[:-1], self.n_units)):
            in_dim = h.shape[1].value
            if build_layer:
                layer.build(in_dim, out_dim)
            h = layer(h, sample=sample)
            h = self.activation(h)
        # Output layer
        if build_layer:
            self.layers[-1].build(self.n_units[-1], self.y_dim)
        h = self.layers[-1](h, sample=sample)

        # model = Model(inputs=[self.X_in, self.A_in, self.E_in], outputs=h)

        return h

    def log_likelihood(self, y, yhat):
        """Log likelihood of prediction assuming a normal distribution and a given variance"""
        if isinstance(self.data_sigma, float) or isinstance(self.data_sigma, int):
            gaussian = tf.distributions.Normal(yhat, self.data_sigma)
        elif callable(self.data_sigma):
            gaussian = tf.distributions.Normal(yhat, self.data_sigma(self.epoch))
        else:
            raise ValueError('data_sigma needs to be either float or callable function')
        return tf.reduce_sum(gaussian.log_prob(y))

    def log_variational_posterior(self):
        """Log posterior using a Gaussian and reparameterization trick for a single layer"""
        log_prob = 0
        for layer in self.layers:
            log_prob += layer.log_variational_posterior()
        return log_prob

    def log_mixture(self):
        """Log scale mixture of two Gaussian densities for the prior for a single layer"""
        log_prob = 0
        for layer in self.layers:
            log_prob += layer.log_mixture_prior()
        return log_prob

    def kl_loss(self, y, yhat, N):
        """KL loss (ELBO) that we optimize"""
        # loss = (self.log_variational_posterior() - self.log_mixture())/steps - self.log_likelihood(y, yhat)
        loss = (self.log_variational_posterior() - self.log_mixture()) * tf.cast(tf.shape(y)[0], tf.float32) / N - \
               self.log_likelihood(y, yhat)
        return loss

    def predict(self, sess, X, A, E, sample=True):
        """Predict target value for X

        Args:
            sess: TensorFlow session
            X: matrix of x values
            sample:  bool to draw a stochastic sample from the posterior"""
        if sample:
            return sess.run(self.yhat, feed_dict={self.X_in: X, self.A_in: A, self.E_in: E})
        else:
            return sess.run(self.yhat_map, feed_dict={self.X_in: X, self.A_in: A, self.E_in: E})

    def predict_posterior(self, sess, X, A, E, dm, n=None):
        num_data = X.shape[0]
        preds = np.zeros([num_data, dm.y_dim, n])

        for i in range(n):
            preds[:, :, i] = self.predict(sess, X, A, E, sample=True)

        return preds


class BBB(BaseGNN):
    # Ensemble of fully-connected networks
    def __init__(self, dm, data_sigma=0.01, optimizer=None, lr_ph=None, lr=1e-4):
        self.data_sigma = data_sigma

        self.activation = tf.nn.relu
        self.conv_layers = []
        for i in range(4):
            # Switch this out between MAP and BBB versions
            self.conv_layers.append(EdgeConditionedConvBBB(32, activation='relu'))
            # self.conv_layers.append(EdgeConditionedConv(32, activation='relu'))
        self.n_units = [32, 32]
        self.layers = []
        for i in range(3):
            self.layers.append(bbb.BBBLayer())

        self.N = tf.placeholder_with_default(float(dm.n), shape=(), name="num_batches")

        super().__init__(dm, optimizer, lr_ph, lr)

        # MAP estimate. MLE is already built in super().__init__
        temp = set(tf.global_variables())
        self.yhat_map = self.build(sample=False, build_layer=False)
        self.vars.update(set(tf.global_variables()) - temp)

        self.mse = tf.reduce_mean(tf.square(self.y - self.yhat))
        self.loss = self.kl_loss(self.y, self.yhat, self.N)
        self.minimizer = self.optimizer.minimize(self.loss)

    def build(self, sample=True, build_layer=True):
        gc = self.X_in
        for layer in self.conv_layers:
            gc = layer([gc, self.A_in, self.E_in])

        pool = GlobalAvgPool()(gc)

        h = pool

        # Iterate over the hidden layers
        for i, (layer, out_dim) in enumerate(zip(self.layers[:-1], self.n_units)):
            in_dim = h.shape[1].value
            if build_layer:
                layer.build(in_dim, out_dim)
            h = layer(h, sample=sample)
            h = self.activation(h)
        # Output layer
        if build_layer:
            self.layers[-1].build(self.n_units[-1], self.y_dim)
        h = self.layers[-1](h, sample=sample)

        # model = Model(inputs=[self.X_in, self.A_in, self.E_in], outputs=h)

        return h

    def log_likelihood(self, y, yhat):
        """Log likelihood of prediction assuming a normal distribution and a given variance"""
        if isinstance(self.data_sigma, float) or isinstance(self.data_sigma, int):
            gaussian = tf.distributions.Normal(yhat, self.data_sigma)
        elif callable(self.data_sigma):
            gaussian = tf.distributions.Normal(yhat, self.data_sigma(self.epoch))
        else:
            raise ValueError('data_sigma needs to be either float or callable function')
        return tf.reduce_sum(gaussian.log_prob(y))

    def log_variational_posterior(self):
        """Log posterior using a Gaussian and reparameterization trick for a single layer"""
        log_prob = 0
        for layer in self.conv_layers:
            log_prob += layer.log_variational_posterior()
        for layer in self.layers:
            log_prob += layer.log_variational_posterior()
        return log_prob

    def log_mixture(self):
        """Log scale mixture of two Gaussian densities for the prior for a single layer"""
        log_prob = 0
        for layer in self.conv_layers:
            log_prob += layer.log_mixture_prior()
        for layer in self.layers:
            log_prob += layer.log_mixture_prior()
        return log_prob

    def kl_loss(self, y, yhat, N):
        """KL loss (ELBO) that we optimize"""
        # loss = (self.log_variational_posterior() - self.log_mixture())/steps - self.log_likelihood(y, yhat)
        loss = (self.log_variational_posterior() - self.log_mixture()) * tf.cast(tf.shape(y)[0], tf.float32) / N - \
               self.log_likelihood(y, yhat)
        return loss

    def predict(self, sess, X, A, E, sample=True):
        """Predict target value for X

        Args:
            sess: TensorFlow session
            X: matrix of x values
            sample:  bool to draw a stochastic sample from the posterior"""
        if sample:
            return sess.run(self.yhat, feed_dict={self.X_in: X, self.A_in: A, self.E_in: E})
        else:
            return sess.run(self.yhat_map, feed_dict={self.X_in: X, self.A_in: A, self.E_in: E})

    def predict_posterior(self, sess, X, A, E, dm, n=None):
        num_data = X.shape[0]
        preds = np.zeros([num_data, dm.y_dim, n])

        for i in range(n):
            preds[:, :, i] = self.predict(sess, X, A, E, sample=True)

        return preds


class EdgeConditionedConvBBB(EdgeConditionedConv):
    """
    An edge-conditioned convolutional layer as presented by [Simonovsky and
    Komodakis (2017)](https://arxiv.org/abs/1704.02901).

    **Mode**: single, batch.

    **This layer expects dense inputs.**

    For each node \(i\), this layer computes:
    $$
        Z_i =  \\frac{1}{\\mathcal{N}(i)} \\sum\\limits_{j \\in \\mathcal{N}(i)} F(E_{ji}) X_{j} + b
    $$
    where \(\\mathcal{N}(i)\) represents the one-step neighbourhood of node \(i\),
     \(F\) is a neural network that outputs the convolution kernel as a
    function of edge attributes, \(E\) is the edge attributes matrix, and \(b\)
    is a bias vector.

    **Input**

    - node features of shape `(n_nodes, n_node_features)` (with optional `batch`
    dimension);
    - binary adjacency matrices with self-loops, of shape `(n_nodes, num_nodes)`
    (with optional `batch` dimension);
    - edge features of shape `(n_nodes, n_nodes, n_edge_features)` (with
    optional `batch` dimension);

    **Output**

    - node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `kernel_network`: a list of integers describing the hidden structure of
    the kernel-generating network (i.e., the ReLU layers before the linear
    output);
    - `activation`: activation function to use;
    - `use_bias`: boolean, whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.

    **Usage**
    ```py
    X_in = Input(shape=(N, F))
    A_in = Input(shape=(N, N))
    E_in = Input(shape=(N, N, S))
    output = EdgeConditionedConv(channels)([X_in, A_in, E_in])
    ```
    """
    def __init__(self,
                 channels,
                 kernel_network=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 prior_sigma1=2.0, prior_sigma2=0.1, prior_pi=0.25,
                 **kwargs):
        super().__init__(channels, **kwargs)
        self.channels = channels
        self.kernel_network = kernel_network
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

        # Hyper-parameters
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.prior_pi = prior_pi

        self.dense_kernel_rho = None
        self.dense_kernel_mu = None
        self.dense_bias_rho = None
        self.dense_bias_mu = None
        self.bias_rho = None
        self.bias_mu = None
        self.dense_kernel = None
        self.dense_bias = None
        self.bias = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.use_bias:
            self.bias_mu = self.add_weight(shape=(self.channels,),
                                        initializer=initializers.RandomUniform(-0.2, 0.2),
                                        name='bias_mu',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias_rho = self.add_weight(shape=(self.channels,),
                                        initializer=initializers.RandomUniform(-5, 4),
                                        name='bias_rho',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias = self.sample_weight(self.bias_mu, self.bias_rho)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # (batch_size, N, F)
        A = inputs[1]  # (batch_size, N, N)
        E = inputs[2]  # (batch_size, N, N, S)

        mode = ops.autodetect_mode(A, X)

        # Parameters
        N = K.shape(X)[-2]
        F = K.int_shape(X)[-1]
        F_ = self.channels

        # Normalize adjacency matrix
        A = ops.normalize_A(A)

        # Filter network
        kernel_network = E
        if self.kernel_network is not None:
            for i, l in enumerate(self.kernel_network):
                kernel_network = self.dense_layer(kernel_network, l,
                                                  'FGN_{}'.format(i),
                                                  activation='relu',
                                                  use_bias=self.use_bias,
                                                  kernel_initializer=self.kernel_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  bias_regularizer=self.bias_regularizer,
                                                  kernel_constraint=self.kernel_constraint,
                                                  bias_constraint=self.bias_constraint)
        kernel_network = self.dense_layer(kernel_network, F_ * F, 'FGN_out')

        # Convolution
        target_shape = (-1, N, N, F_, F) if mode == ops.modes['B'] else (N, N, F_, F)
        kernel = K.reshape(kernel_network, target_shape)
        output = kernel * A[..., None, None]

        if mode == ops.modes['B']:
            output = tf.einsum('abicf,aif->abc', output, X)
        else:
            output = tf.einsum('bicf,if->bc', output, X)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {
            'channels': self.channels,
            'kernel_network': self.kernel_network,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def sample_weight(mu, rho):
        """Monte Carlo sample of parameter matrix"""
        epsilon = tf.random.normal(tf.shape(rho))
        z = tf.add(mu, tf.multiply(tf.math.log1p(tf.exp(rho)), epsilon))
        return z

    def dense_layer(self,
                    x,
                    units,
                    name,
                    activation=None,
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None):
        input_dim = K.int_shape(x)[-1]
        self.dense_kernel_mu = self.add_weight(shape=(input_dim, units),
                                    name=name + '_kernel_mu',
                                    initializer=initializers.RandomUniform(-0.2, 0.2),
                                    regularizer=kernel_regularizer,
                                    constraint=kernel_constraint)
        self.dense_kernel_rho = self.add_weight(shape=(input_dim, units),
                                    name=name + '_kernel_rho',
                                    initializer=initializers.RandomUniform(-5, 4),
                                    regularizer=kernel_regularizer,
                                    constraint=kernel_constraint)
        self.dense_bias_mu = self.add_weight(shape=(units,),
                                       name=name + '_bias_mu',
                                       initializer=initializers.RandomUniform(-0.2, 0.2),
                                       regularizer=bias_regularizer,
                                       constraint=bias_constraint)
        self.dense_bias_rho = self.add_weight(shape=(units,),
                                        name=name + '_bias_rho',
                                        initializer=initializers.RandomUniform(-5, 4),
                                        regularizer=bias_regularizer,
                                        constraint=bias_constraint)

        self.dense_kernel = self.sample_weight(self.dense_kernel_mu, self.dense_kernel_rho)
        self.dense_bias = self.sample_weight(self.dense_bias_mu, self.dense_bias_rho)

        act = activations.get(activation)
        output = K.dot(x, self.dense_kernel)
        if use_bias:
            output = K.bias_add(output, self.dense_bias_mu)
        output = act(output)
        return output

    def log_variational_posterior(self):
        """Log posterior using a Gaussian and reparameterization trick"""
        sigma = tf.math.log1p(tf.exp(self.dense_kernel_rho))
        gaussian = tf.distributions.Normal(self.dense_kernel_mu, sigma)
        log_prob_W = tf.reduce_sum(gaussian.log_prob(self.dense_kernel))

        sigma = tf.math.log1p(tf.exp(self.dense_bias_rho))
        gaussian = tf.distributions.Normal(self.dense_bias_mu, sigma)
        log_prob_dense_b = tf.reduce_sum(gaussian.log_prob(self.dense_bias))

        sigma = tf.math.log1p(tf.exp(self.bias_rho))
        gaussian = tf.distributions.Normal(self.bias_mu, sigma)
        log_prob_b = tf.reduce_sum(gaussian.log_prob(self.bias))
        return log_prob_W + log_prob_dense_b + log_prob_b

    def log_mixture_prior(self):
        """Log scale mixture of two Gaussian densities for the prior"""
        gaussian1 = tf.distributions.Normal(0., self.prior_sigma1)
        gaussian2 = tf.distributions.Normal(0., self.prior_sigma2)
        log_prob_b = tf.reduce_sum(tf.math.log(self.prior_pi * gaussian1.prob(self.bias) +
                                               (1 - self.prior_pi) * gaussian2.prob(self.bias)))
        log_prob_dense_b = tf.reduce_sum(tf.math.log(self.prior_pi * gaussian1.prob(self.dense_bias) +
                                               (1 - self.prior_pi) * gaussian2.prob(self.dense_bias)))
        log_prob_W = tf.reduce_sum(tf.math.log(self.prior_pi * gaussian1.prob(self.dense_kernel) +
                                               (1 - self.prior_pi) * gaussian2.prob(self.dense_kernel)))

        return log_prob_b + log_prob_dense_b + log_prob_W


class GraphNeuralLinear(BaseGNN):
    # Convolutional neural linear - CNN with Bayesian on the last layer
    def __init__(self, dm, optimizer=None, lr_ph=None, lr=1e-4):
        # Output of last hidden layer
        self.h = None

        super().__init__(dm, optimizer, lr_ph, lr)

        # Model for doing Bayesian linear regression
        self.clf = sklearn.linear_model.BayesianRidge()

    def build(self):
        """Same as base CNN, but saving the output of the last hidden layer"""
        gc1 = EdgeConditionedConv(32, activation='relu')([self.X_in, self.A_in, self.E_in])
        gc = EdgeConditionedConv(32, activation='relu')([gc1, self.A_in, self.E_in])
        gc = EdgeConditionedConv(32, activation='relu')([gc, self.A_in, self.E_in])
        gc = EdgeConditionedConv(32, activation='relu')([gc, self.A_in, self.E_in])
        pool = GlobalAvgPool()(gc)
        h = Dense(64, activation=tf.nn.relu)(pool)
        h = Dense(64, activation=tf.nn.relu)(h)
        h = Dense(64, activation=tf.nn.relu)(h)
        h = Dense(64, activation=tf.nn.relu)(h)
        self.h = h
        h = Dense(self.y_dim)(h)

        # model = Model(inputs=[self.X_in, self.A_in, self.E_in], outputs=h)

        return h

    def predict_posterior(self, sess, X, A, E, dm, n=30):
        """Predict distribution of target value for X

        Returns:
            pred_mean: (n_data, n_features)
            pred_std: (n_data, n_features)
        """
        ydim = dm.y_dim
        n_data = X.shape[0]
        pred_mean = np.zeros([n_data, ydim])
        pred_std = np.zeros([n_data, ydim])

        for i in range(ydim):
        #     print(np.shape(dm.X))
        #     print(np.shape(dm.A))
        #     print(np.shape(dm.E))
            self.clf.fit(sess.run(self.h, feed_dict={self.X_in: dm.X, self.A_in: dm.A, self.E_in: dm.E}),
                         dm.Y[:, i].ravel())
            pred_mean_i, pred_std_i = self.clf.predict(sess.run(self.h, feed_dict={self.X_in: X, self.A_in: A,
                                                                                   self.E_in: E}), return_std=True)
            pred_mean[:, i] = pred_mean_i
            pred_std[:, i] = pred_std_i

        return pred_mean, pred_std

