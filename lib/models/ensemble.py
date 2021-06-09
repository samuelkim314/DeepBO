import numpy as np
import tensorflow as tf

from lib.models.nn_base import BaseNetwork, CNN


class Ensemble(BaseNetwork):
    # Ensemble of fully-connected networks
    def __init__(self, dm, results_dir, n_networks=10, print_loss=False, optimizer=None,
                 hidden_units=None, lr_ph=None, lr=1e-4):

        self.y_hat_list = []
        self.n_networks = n_networks

        super().__init__(dm, results_dir, print_loss=print_loss, optimizer=optimizer, hidden_units=hidden_units,
                         lr_ph=lr_ph, lr=lr)

        self.N = tf.placeholder_with_default(float(dm.n), shape=(), name="num_batches")

        self.mse_list = [tf.reduce_mean(tf.square(self.y - yhat)) for yhat in self.y_hat_list]
        self.loss = tf.reduce_mean(self.mse_list) * tf.cast(tf.shape(self.y)[0], tf.float32) / self.N
        self.minimizer = self.optimizer.minimize(self.loss)

    def build(self):
        y_hat = []
        for i in range(self.n_networks):
            y_hat.append(super().build())
        self.y_hat_list = y_hat
        return tf.reduce_mean(y_hat)

    def predict(self, sess, X, sample=True):
        """Predict target value for X

        Args:
            sess: TensorFlow session
            X: matrix of x values
            sample:  bool to draw a stochastic sample from the posterior"""
        if sample:
            i = np.random.randint(low=0, high=self.n_networks)
            return sess.run(self.y_hat_list[i], feed_dict={self.X: X})
        else:
            return sess.run(self.yhat, feed_dict={self.X: X})

    def predict_posterior(self, sess, X, dm, n=30):
        num_data = X.shape[0]
        preds = np.zeros([num_data, dm.y_dim, self.n_networks])

        for i in range(self.n_networks):
            preds[:, :, i] = sess.run(self.y_hat_list[i], feed_dict={self.X: X})

        return preds

    def train_epoch(self, sess, batcher, steps, N=None, augment=False, augment_sg11=False, epoch=-1, lr=None):
        """Train a single epoch"""
        for step in range(steps):
            x_batch, y_batch = next(batcher)
            if lr is None:
                sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.epoch: epoch, self.N: N})
            else:
                sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.epoch: epoch, self.N: N,
                                                    self.lr_ph: lr})


class ConvEnsemble(CNN, Ensemble):
    # Ensemble of convolutional networks
    def __init__(self, dm, results_dir, n_networks=10, print_loss=False, optimizer=None,
                 n_channels=None, hidden_units=None, lr_ph=None, lr=1e-4):

        self.y_hat_list = []
        self.n_networks = n_networks

        super().__init__(dm, results_dir, print_loss=print_loss, optimizer=optimizer,
                         n_channels=n_channels, hidden_units=hidden_units, lr_ph=lr_ph, lr=lr)

        self.N = tf.placeholder_with_default(float(dm.n), shape=(), name="num_batches")

        self.mse_list = [tf.reduce_mean(tf.square(self.y - yhat)) for yhat in self.y_hat_list]
        self.loss = tf.reduce_mean(self.mse_list) * tf.cast(tf.shape(self.y)[0], tf.float32) / self.N
        self.minimizer = self.optimizer.minimize(self.loss)

    def build(self):
        y_hat = []
        for i in range(self.n_networks):
            y_hat.append(super().build())
        self.y_hat_list = y_hat
        return tf.reduce_mean(y_hat)

    def train_epoch(self, sess, batcher, steps, N=None, augment=False, augment_sg11=False, epoch=0, lr=None):
        """Override so that we can include random translation for data augmentation"""
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
                sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.N: N})
            else:
                sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch, self.N: N,
                                                    self.lr_ph: lr})

            sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch})
