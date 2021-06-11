# from .nn import BaseNetwork
from lib.models import nn_base
import tensorflow as tf
import numpy as np


class BBBLayer:
    # Fully-connected network with weight uncertainty using Bayes by Backprop
    def __init__(self, prior_sigma1=2.0, prior_sigma2=0.1, prior_pi=0.25):
        # Hyper-parameters
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.prior_pi = prior_pi

        # Parameters
        self.b = None
        self.W = None
        self.b_mu = None
        self.b_rho = None
        self.W_mu = None
        self.W_rho = None

        # If we want to set a prior other than 0
        self.use_prior = False
        self.W_prior = None
        self.b_prior = None
        self.W_prior_sigma = None
        self.b_prior_sigma = None

    @staticmethod
    def init_mu(shape):
        """Initialize mu, mean of the parameter

        Args:
            shape: 1-D integer Tensor or Python array of the tensor shape"""
        return tf.Variable(tf.random.uniform(shape, minval=-0.2, maxval=0.2))

    @staticmethod
    def init_rho(shape):
        """Initialize rho, reparameterization of standard deviation

        Args:
            shape: 1-D integer Tensor or Python array of the tensor shape"""
        return tf.Variable(tf.random.uniform(shape, minval=-5, maxval=-4))

    def build(self, in_dim, out_dim):
        """Initialize mu and rho parameters for the weights and biases of a single layer

        Args:
            in_dim: integer of the input dimension
            out_dim: integer of the output dimension
        """
        self.b_mu = self.init_mu([out_dim])
        self.b_rho = self.init_rho([out_dim])
        self.W_mu = self.init_mu([in_dim, out_dim])
        self.W_rho = self.init_rho([in_dim, out_dim])

        return self.b_mu, self.b_rho, self.W_mu, self.W_rho

    def log_variational_posterior(self):
        """Log posterior using a Gaussian and reparameterization trick"""
        sigma = tf.math.log1p(tf.exp(self.W_rho))
        gaussian = tf.distributions.Normal(self.W_mu, sigma)
        log_prob_W = tf.reduce_sum(gaussian.log_prob(self.W))

        sigma = tf.math.log1p(tf.exp(self.b_rho))
        gaussian = tf.distributions.Normal(self.b_mu, sigma)
        log_prob_b = tf.reduce_sum(gaussian.log_prob(self.b))
        return log_prob_W + log_prob_b

    def log_mixture_prior(self):
        """Log scale mixture of two Gaussian densities for the prior"""
        if not self.use_prior:
            gaussian1 = tf.distributions.Normal(0., self.prior_sigma1)
            gaussian2 = tf.distributions.Normal(0., self.prior_sigma2)
            log_prob_b = tf.reduce_sum(tf.math.log(self.prior_pi * gaussian1.prob(self.b) +
                                                   (1 - self.prior_pi) * gaussian2.prob(self.b)))
            log_prob_W = tf.reduce_sum(tf.math.log(self.prior_pi * gaussian1.prob(self.W) +
                                                   (1 - self.prior_pi) * gaussian2.prob(self.W)))
        else:
            gaussian_w = tf.distributions.Normal(self.W_prior, self.W_prior_sigma)
            gaussian_b = tf.distributions.Normal(self.b_prior, self.b_prior_sigma)
            log_prob_b = tf.reduce_sum(tf.math.log(gaussian_b.prob(self.b)))
            log_prob_W = tf.reduce_sum(tf.math.log(gaussian_w.prob(self.W)))

        return log_prob_b + log_prob_W

    @staticmethod
    def sample_weight(mu, rho):
        """Monte Carlo sample of parameter matrix"""
        epsilon = tf.random.normal(tf.shape(rho))
        z = tf.add(mu, tf.multiply(tf.math.log1p(tf.exp(rho)), epsilon))
        return z

    def __call__(self, x, sample=True):
        """Multiply x by weight matrix and add bias. Will reuse weights"""
        # Initialize variables if they haven't been already
        if any(v is None for v in [self.b_mu, self.b_rho, self.W_mu, self.W_rho]):
            self.build(x.shape[1].value, x.shape[1].value)

        if sample:
            self.W = self.sample_weight(self.W_mu, self.W_rho)
            self.b = self.sample_weight(self.b_mu, self.b_rho)
        else:
            self.W = self.W_mu
            self.b = self.b_mu

        h = tf.matmul(x, self.W)
        h = h + self.b

        return h

    def set_prior(self, sess):
        self.use_prior = True
        self.W_prior = sess.run(self.W_mu)
        self.b_prior = sess.run(self.b_mu)
        self.W_prior_sigma = sess.run(tf.log1p(tf.exp(self.W_rho)))
        self.b_prior_sigma = sess.run(tf.log1p(tf.exp(self.b_rho)))


class BBBConvLayer(BBBLayer):
    # Convolutional layer with periodic boundary conditions and weight uncertainty using Bayes by Backprop
    def __init__(self, prior_sigma1=2.0, prior_sigma2=0.1, prior_pi=0.25, filter_size=3,
                 periodic=True):
        super().__init__(prior_sigma1, prior_sigma2, prior_pi)
        self.filter_size = filter_size
        self.periodic = periodic

    def build(self, in_channels, out_channels):
        """Initialize mu and rho parameters for the weights and biases of a single layer

        Args:
            in_channels: integer of the input dimension
            out_channels: integer of the output dimension
        """
        self.W_mu = self.init_mu([self.filter_size, self.filter_size, in_channels, out_channels])
        self.W_rho = self.init_rho([self.filter_size, self.filter_size, in_channels, out_channels])
        return self.W_mu, self.W_rho

    def __call__(self, x, sample=True):
        """Multiply x by weight matrix and add bias."""
        # Initialize variables if they haven't been already
        if any(v is None for v in [self.W_mu, self.W_rho]):
            self.build(x.shape[3].value, x.shape[3].value)

        if sample:
            self.W = self.sample_weight(self.W_mu, self.W_rho)
        else:
            self.W = self.W_mu

        # Pad for periodic boundary conditions
        if self.periodic:
            padding = self.filter_size - 1
            x = tf.concat((x, x[:, :padding, :, :]), axis=1)
            x = tf.concat((x, x[:, :, :padding, :]), axis=2)
            h = tf.nn.conv2d(x, self.W, strides=[1, 1, 1, 1], padding='VALID')
        else:
            h = tf.nn.conv2d(x, self.W, strides=[1, 1, 1, 1], padding='SAME')

        return h

    def log_variational_posterior(self):
        """Log posterior using a Gaussian and reparameterization trick"""
        sigma = tf.log1p(tf.exp(self.W_rho))
        gaussian = tf.distributions.Normal(self.W_mu, sigma)
        log_prob_W = tf.reduce_sum(gaussian.log_prob(self.W))

        return log_prob_W

    def log_mixture_prior(self):
        """Log scale mixture of two Gaussian densities for the prior"""
        gaussian1 = tf.distributions.Normal(0., self.prior_sigma1)
        gaussian2 = tf.distributions.Normal(0., self.prior_sigma2)
        log_prob_W = tf.reduce_sum(tf.log(self.prior_pi * gaussian1.prob(self.W) +
                                          (1 - self.prior_pi) * gaussian2.prob(self.W)))
        return log_prob_W


class BayesByBackprop(nn_base.BaseNetwork):
    """Fully-connected network with weight uncertainty using Bayes by Backprop"""
    def __init__(self, dm, activation=tf.nn.relu, prior_sigma1=2.0, prior_sigma2=0.1, prior_pi=0.25, data_sigma=0.01,
                 hidden_units=None, optimizer=None, lr_ph=None, lr=1e-4):
        """Note that hidden_units corresponds to just the hidden layers"""
        # Hyper-parameters
        self.data_sigma = data_sigma
        self.activation = activation

        if hidden_units is None:
            self.hidden_units = [64, 128, 256, 512, 256, 128, 64]
        else:
            self.hidden_units = hidden_units

        # self.layers = []
        # for i in range(len(self.hidden_units) + 1):  # +1 to add the output layer
        #     self.layers.append(BBBLayer(prior_sigma1, prior_sigma2, prior_pi))
        self.layers = []
        self.init_layers(prior_sigma1, prior_sigma2, prior_pi)

        super().__init__(dm, optimizer=optimizer,
                         hidden_units=self.hidden_units,
                         lr_ph=lr_ph, lr=lr)

        self.N = tf.placeholder_with_default(float(dm.n), shape=(), name="num_batches")

        temp = set(tf.global_variables())
        self.yhat_map = self.build(sample=False, build_layer=False)
        self.vars.update(set(tf.global_variables()) - temp)
        self.loss_map = tf.reduce_mean(tf.square(self.y - self.yhat_map))   # Non-sampling loss

        loss = self.kl_loss(self.y, self.yhat, self.N)
        self.minimizer = self.optimizer.minimize(loss)
        self.opt_vars = self.optimizer.variables()

    def init_layers(self, prior_sigma1, prior_sigma2, prior_pi):
        # Initialize all the layers with hyper-parameters
        # +1 to add the output layer
        for i in range(len(self.hidden_units) + 1):
            self.layers.append(BBBLayer(prior_sigma1, prior_sigma2, prior_pi))

    def build(self, sample=True, build_layer=True):
        h = self.X
        # Iterate over the hidden layers
        for i, (layer, out_dim) in enumerate(zip(self.layers[:-1], self.hidden_units)):
            in_dim = h.shape[1].value
            if build_layer:
                layer.build(in_dim, out_dim)
            h = layer(h, sample=sample)
            h = self.activation(h)
        # Output layer
        if build_layer:
            self.layers[-1].build(self.hidden_units[-1], self.y_dim)
        h = self.layers[-1](h, sample=sample)
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

    def kl_loss_bound(self, yhat, N):
        """KL loss (ELBO) that we optimize"""
        # loss = (self.log_variational_posterior() - self.log_mixture())/steps - self.log_likelihood(y, yhat)
        loss = (self.log_variational_posterior() - self.log_mixture()) * tf.cast(tf.shape(yhat)[0], tf.float32) / N + \
               tf.nn.relu(self.yhat - self.ub)
        return loss

    def train_epoch(self, sess, batcher, steps, N=None, augment=False, augment_sg11=False, lr=None):
        for step in range(steps):
            x_batch, y_batch = next(batcher)
            if lr is None:
                if N is None:
                    sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, })
                else:
                    sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.N: N})
            else:
                if N is None:
                    sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.lr_ph: lr})
                else:
                    sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.N: N, self.lr_ph: lr})

    def calc_loss(self, sess, x, y):
        """Calculate loss without sampling"""
        return sess.run(self.loss_map, feed_dict={self.X: x, self.y: y})

    def predict(self, sess, X, sample=True):
        if sample:
            return sess.run(self.yhat, feed_dict={self.X: X})
        else:
            return sess.run(self.yhat_map, feed_dict={self.X: X})

    def set_prior(self, sess):
        for layer in self.layers:
            layer.set_prior(sess)
        loss = self.kl_loss(self.y, self.yhat, self.N)
        self.minimizer = self.optimizer.minimize(loss)
        self.opt_vars = self.optimizer.variables()


class BayesByBackpropConv(BayesByBackprop):
    # Convolutional network with weight uncertainty using Bayes by Backprop
    def __init__(self, dm, prior_sigma1=2.0, prior_sigma2=0.1, prior_pi=0.25, data_sigma=0.01,
                 n_channels=None, hidden_units=None, optimizer=None, periodic=True, lr_ph=None, lr=1e-4):
        # Hyper-parameters
        self.data_sigma = data_sigma
        self.periodic = periodic

        if n_channels is None and hidden_units is None:
            self.n_channels = [8, 8, 16, 16, 32]
            self.n_units = [64, 32, 8, 4, 2]
        else:
            self.n_channels = n_channels
            self.n_units = hidden_units

        self.conv_layers = None
        self.dense_layers = None

        super().__init__(dm, hidden_units=hidden_units, optimizer=optimizer,
                         prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, prior_pi=prior_pi, data_sigma=data_sigma,
                         lr_ph=lr_ph, lr=lr)

    def init_layers(self, prior_sigma1, prior_sigma2, prior_pi):
        # Initialize all the layers with hyper-parameters
        self.conv_layers = []
        for i in range(len(self.n_channels)):
            self.conv_layers.append(BBBConvLayer(prior_sigma1, prior_sigma2, prior_pi, periodic=self.periodic))
        self.dense_layers = []
        # +1 to add the output layer
        for i in range(len(self.n_units) + 1):
            self.dense_layers.append(BBBLayer(prior_sigma1, prior_sigma2, prior_pi))
        self.layers = self.conv_layers + self.dense_layers

    def build(self, sample=True, build_layer=True):

        h = self.X
        for i, (layer, out_channels) in enumerate(zip(self.conv_layers, self.n_channels)):
            in_channels = h.shape[3].value
            if build_layer:
                layer.build(in_channels, out_channels)
            h = layer(h, sample=sample)
            h = tf.nn.relu(h)
            # if i % 2 == 0:
            #     h = tf.keras.layers.AvgPool2D(strides=1)(h)
            # else:
            #     h = tf.keras.layers.MaxPool2D()(h)
            h = tf.keras.layers.MaxPool2D()(h)
        h = tf.keras.layers.Flatten()(h)
        for i, (layer, out_dim) in enumerate(zip(self.dense_layers[:-1], self.n_units)):
            in_dim = h.shape[1].value
            if build_layer:
                layer.build(in_dim, out_dim)
            h = layer(h, sample=sample)
            h = tf.nn.relu(h)
        # Output layer
        if build_layer:
            self.dense_layers[-1].build(self.n_units[-1], self.y_dim)
        h = self.dense_layers[-1](h, sample=sample)
        return h

    def train_epoch(self, sess, batcher, steps, N=None, augment=False, augment_sg11=False, lr=None):
        """Override so that we can include random translation, rotation, and flips for data augmentation"""
        for step in range(steps):
            x_batch, y_batch = next(batcher)

            if augment:
                randx = np.random.randint(0, 32)
                x_batch = np.concatenate((x_batch[:, randx:], x_batch[:, :randx]), axis=1)
                randy = np.random.randint(0, 32)
                x_batch = np.concatenate((x_batch[:, :, randy:], x_batch[:, :, :randy]), axis=2)

            if augment_sg11:
                irot = np.random.randint(4)
                x_batch = np.rot90(x_batch, k=irot, axes=(1, 2))
                if np.random.randint(2):
                    x_batch = np.fliplr(x_batch)

            if lr is None:
                if N is None:
                    sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch,})
                else:
                    sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.N: N})
            else:
                if N is None:
                    sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.lr_ph: lr})
                else:
                    sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.N: N, self.lr_ph: lr})

