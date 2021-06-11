import tensorflow as tf
import numpy as np
import sklearn.linear_model


class BaseNetwork:
    """Abstract class for Bayesian neural networks"""
    def __init__(self, dm, optimizer=None, hidden_units=None, lr_ph=None, lr=None):
        # shape of self.X matches that of dm.X other than first axis, which size batch_size
        # This dynamic shape allows extension ot multiple data types (e.g. images)
        self.X = tf.placeholder("float", shape=[None, *dm.X.shape[1:]], name="X")
        self.y = tf.placeholder("float", shape=[None, dm.y_dim], name="y")
        self.y_dim = dm.y_dim

        self.hidden_units = hidden_units

        # Build architecture graph while keeping track of all TensorFlow variables associated with this graph
        temp = set(tf.global_variables())   # Set (unordered collection) of all variables
        self.yhat = self.build()
        self.vars = set(tf.global_variables()) - temp   # Subtracting out previous set to get just variables for the
                                                        # architecture

        self.mse = tf.reduce_mean(tf.square(self.y - self.yhat))
        self.loss = self.mse
        self.optimizer = optimizer
        self.opt_vars = self.optimizer.variables()
        self.minimizer = self.optimizer.minimize(self.loss)

        # dummy variable used for epoch-dependent operations such as KL annealing
        self.lr = lr
        self.lr_ph = lr_ph

        # self.saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

    def build(self):
        assert len(self.X.shape) == 2
        h = self.X
        for units_i in self.hidden_units:
            h = tf.keras.layers.Dense(units_i, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(self.y_dim, activation=None)(h)
        return h

    def train_epoch(self, sess, batcher, steps, N=None, augment=False, augment_sg11=False, lr=None):
        """Train a single epoch"""
        for step in range(steps):
            x_batch, y_batch = next(batcher)
            sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch})

    def calc_loss(self, sess, x, y):
        """Calculate loss"""
        return sess.run(self.loss, feed_dict={self.X: x, self.y: y})

    def calc_mse_batch(self, sess, X, Y, batch_size):
        """Calculate validation loss by breaking up validation dataset into batches so we don't overrun GPU memory."""
        n = Y.shape[0]
        batch_ind = range(0, n, batch_size)
        loss = []       # List of loss for each mini-batch
        weight = []     # Size of each mini-batch, used for calculating weighted average
        for i_batch in batch_ind:
            X_batch = X[i_batch:i_batch + batch_size]
            Y_batch = Y[i_batch:i_batch + batch_size]
            loss.append(sess.run(self.mse, feed_dict={self.X: X_batch, self.y: Y_batch}))
            weight.append(len(X_batch))
        return np.average(loss, weights=weight)

    def calc_loss_batch(self, sess, X, Y, batch_size):
        """Calculate validation loss by breaking up validation dataset into batches so we don't overrun GPU memory."""
        n = Y.shape[0]
        batch_ind = range(0, n, batch_size)
        loss = []       # List of loss for each mini-batch
        weight = []     # Size of each mini-batch, used for calculating weighted average
        for i_batch in batch_ind:
            X_batch = X[i_batch:i_batch + batch_size]
            Y_batch = Y[i_batch:i_batch + batch_size]
            loss.append(self.calc_loss(sess, X_batch, Y_batch))
            weight.append(len(X_batch))
        return np.average(loss, weights=weight)

    def train(self, sess, epochs, dm, X_val=None, Y_val=None, save_model=False,
              augment=False, augment_sg11=False, cycle=False):
        """Train the neural network

        Args:
            sess : tensorflow session
            epochs : (int) number of epochs to train
            dm : data_manager object containing the dataset
            X_val, Y_val : (optional) validation dataset
            save_model : (bool) save the trained model (and checkpoints) to a Tensorflow file
            augment:    (bool) flag to use data augmentation using periodic translation
            augment_sg11:   (bool) flag for data augmentation using space group 11 (flips and rotations)
        """

        batcher = dm.batch_iter()

        train_loss_list = []
        val_loss_list = []

        for epoch in range(epochs):
            steps = dm.steps

            if cycle:
                lr = self.lr / 2 * (np.cos((np.pi * ((epoch - 1) % epochs)) / epochs) + 1)
            else:
                lr = 1e-4

            self.train_epoch(sess, batcher, steps, dm.n, augment, augment_sg11, lr=lr)
            if X_val is not None:
                # Calculate the training/validation loss
                train_loss = self.calc_mse_batch(sess, dm.X, dm.Y, dm.batch_size)
                val_loss = self.calc_mse_batch(sess, X_val, Y_val, dm.batch_size)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)

                # # Save the model
                # if save_model and epoch % 100 == 99:
                #     # Can specify a new results directory for the model
                #     if results_dir is None:
                #         self.saver.save(sess, os.path.join(self.results_dir, 'model'), global_step=epoch)
                #     else:
                #         self.saver.save(sess, os.path.join(results_dir, 'model'), global_step=epoch)
        if X_val is not None:
            # Return the final validation error
            # return self.calc_mse_batch(sess, X_val, Y_val, dm.batch_size)
            return train_loss_list, val_loss_list
        else:
            return self.calc_mse_batch(sess, dm.X, dm.Y, dm.batch_size)

    def reset(self, sess):
        """Initialize or reset all model variables"""
        sess.run(tf.variables_initializer(self.vars))
        self.reset_optimizer(sess)

    def reset_optimizer(self, sess):
        """Initialize or reset model optimizer variables"""
        sess.run(tf.variables_initializer(self.opt_vars))

    def predict(self, sess, X, sample=True):
        """Predict target value for X

        Args:
            sample:  bool to draw a stochastic sample from the posterior"""
        return sess.run(self.yhat, feed_dict={self.X: X})

    def predict_posterior(self, sess, X, dm, n=30):
        """Predict distribution of target value for X

        Returns:
            preds: (n_data, n_features, n_sample) array
        """
        num_data = X.shape[0]
        preds = np.zeros([num_data, dm.y_dim, n])

        for i in range(n):
            preds[:, :, i] = self.predict(sess, X, sample=True)

        return preds


class CNN(BaseNetwork):
    # Convolutional network
    def __init__(self, dm, n_channels=None, hidden_units=None, optimizer=None, lr_ph=None, lr=None):
        # self.X = tf.placeholder("float", shape=[None, dm.width, dm.height, dm.n_channels], name="X")
        # self.y = tf.placeholder("float", shape=[None, dm.y_dim], name="y")
        # self.y_dim = dm.y_dim

        if n_channels is None and hidden_units is None:
            self.n_channels = [8, 8, 16, 16, 32]
            self.hidden_units = [64, 32, 8, 4, 2]
        else:
            self.n_channels = n_channels
            self.hidden_units = hidden_units

        super().__init__(dm, optimizer=optimizer, hidden_units=self.hidden_units,
                         lr_ph=lr_ph, lr=lr)

    def build(self):
        assert len(self.X.shape) == 4
        h = self.X
        filter_size = 3
        padding = filter_size - 1
        for i, channels_i in enumerate(self.n_channels):
            # Padding for periodic boundary conditions
            h = tf.concat((h, h[:, :padding, :, :]), axis=1)
            h = tf.concat((h, h[:, :, :padding, :]), axis=2)
            h = tf.keras.layers.Conv2D(filters=channels_i, kernel_size=filter_size, strides=(1, 1), padding='VALID',
                                       activation=tf.nn.relu)(h)
            # if i % 2 == 0:
            #     h = tf.keras.layers.AvgPool2D(strides=1)(h)
            # else:
            #     h = tf.keras.layers.MaxPool2D()(h)
            h = tf.keras.layers.MaxPool2D()(h)
        h = tf.keras.layers.Flatten()(h)
        for neurons_i in self.hidden_units:
            h = tf.keras.layers.Dense(neurons_i, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(self.y_dim, activation=None)(h)
        return h

    def train_epoch(self, sess, batcher, steps, N=None, augment=False, augment_sg11=False, lr=None):
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

            sess.run(self.minimizer, feed_dict={self.X: x_batch, self.y: y_batch})


class Dropout(BaseNetwork):
    # Fully-connect network with dropout
    def __init__(self, dm, drop_rate=0.5, optimizer=None, hidden_units=None):
        self.rate = drop_rate
        self.rate_ph = tf.placeholder_with_default(drop_rate, shape=(), name="rate")  # dropout probability

        super().__init__(dm, optimizer=optimizer, hidden_units=hidden_units)

    def build(self):
        h = self.X
        for units_i in self.hidden_units:
            h = tf.keras.layers.Dense(units_i, activation=tf.nn.relu)(h)
            h = tf.keras.layers.Dropout(self.rate_ph)(h)
        h = tf.keras.layers.Dense(self.y_dim)(h)
        return h

    def calc_loss(self, sess, x, y):
        """Calculate loss"""
        return sess.run(self.loss, feed_dict={self.X: x, self.y: y, self.rate_ph: 0})

    def predict(self, sess, X, sample=True):
        if sample:
            prob = self.rate
        else:
            prob = 0
        return sess.run(self.yhat, feed_dict={self.X: X, self.rate_ph: prob})


class DropoutConv(CNN, BaseNetwork):
    # Convolutional network with dropout
    def __init__(self, dm, drop_rate=0.5, n_channels=None, hidden_units=None, optimizer=None):
        self.rate = drop_rate
        self.rate_ph = tf.placeholder_with_default(drop_rate, shape=(), name="rate")  # dropout probability

        if n_channels is None and hidden_units is None:
            self.n_channels = [8, 8, 16, 16, 32]
            self.n_units = [64, 32, 32, 32, self.y_dim]
        else:
            self.n_channels = n_channels
            self.n_units = hidden_units

        super().__init__(dm, n_channels=n_channels, hidden_units=hidden_units, optimizer=optimizer)

    def build(self):
        h = self.X
        filter_size = 3
        for i, channels_i in enumerate(self.n_channels):
            padding = filter_size - 1
            h = tf.concat((h, h[:, :padding, :, :]), axis=1)
            h = tf.concat((h, h[:, :, :padding, :]), axis=2)
            h = tf.keras.layers.Conv2D(filters=channels_i, kernel_size=filter_size, strides=1, padding='valid',
                                       activation=tf.nn.relu)(h)
            if i % 2 == 0:
                h = tf.keras.layers.AvgPool2D(strides=1)(h)
            else:
                h = tf.keras.layers.MaxPool2D()(h)
        h = tf.keras.layers.Flatten()(h)
        for i, units_i in enumerate(self.n_units[:-1]):
            h = tf.keras.layers.Dense(units_i, activation=tf.nn.relu)(h)
            h = tf.keras.layers.Dropout(self.rate_ph)(h)
        h = tf.keras.layers.Dense(self.n_units[-1])(h)
        return h


class Single(BaseNetwork):
    # Fully-connected network with the ability to optimize the input
    def __init__(self, dm, optimizer=None, hidden_units=None, obj_fun=lambda x: x[:, 0]):

        self.training = tf.placeholder_with_default(True, shape=[])

        self.X = tf.placeholder_with_default([[0.0]*dm.x_dim], shape=[None, dm.x_dim], name="X")
        self.y = tf.placeholder("float", shape=[None, dm.y_dim], name="y")
        self.y_dim = dm.y_dim
        self.hidden_units = hidden_units

        input_opt_dim = 1024

        self.X_var_ph = tf.placeholder(tf.float32, shape=[input_opt_dim, dm.x_dim])
        self.X_var = tf.Variable(tf.zeros((input_opt_dim, dm.x_dim)))
        self.set_var_op = self.X_var.assign(self.X_var_ph)
        self.X_var2 = tf.clip_by_value(self.X_var, 0, 1)

        temp = set(tf.global_variables())
        self.yhat = self.build()
        self.vars = set(tf.global_variables()) - temp

        self.mse = tf.reduce_mean(tf.square(self.y - self.yhat))
        self.loss = self.mse
        self.optimizer = optimizer
        self.opt_vars = self.optimizer.variables()
        self.minimizer = self.optimizer.minimize(self.loss)

        self.obj_fun = obj_fun
        self.optimizer2 = optimizer
        self.minimizer_input = self.optimizer2.minimize(-obj_fun(self.yhat), var_list=[self.X_var])

        self.epoch = tf.placeholder_with_default(0, [])  # dummy variable

        # self.saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

    def build(self):
        h = tf.cond(self.training, lambda: self.X, lambda: self.X_var2)
        for units_i in self.hidden_units:
            h = tf.keras.layers.Dense(units_i, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(self.y_dim, activation=None)(h)
        return h

    def max_input(self, sess, X_0):
        sess.run(self.set_var_op, feed_dict={self.X_var_ph: X_0, self.training: False})
        for i in range(1000):
            sess.run(self.minimizer_input, feed_dict={self.training: False})
            # if i % 100 == 0:
            #     print(sess.run(self.X_var2[:5], feed_dict={self.training: False}))
            #     print(sess.run(self.obj_fun(self.yhat)[:5], feed_dict={self.training: False}))
        X_opt, y_opt = sess.run([self.X_var2, self.obj_fun(self.yhat)], feed_dict={self.training: False})
        return X_opt, y_opt


class NeuralLinear(BaseNetwork):
    # Neural linear - standard network with Bayesian on the last layer
    def __init__(self, dm, optimizer=None, hidden_units=None, lr_ph=None, lr=1e-4):

        self.h = None
        if hidden_units is None:
            self.hidden_units = [64, 128, 256, 512, 256, 128, 64]
        else:
            self.hidden_units = hidden_units

        super().__init__(dm, optimizer=optimizer, hidden_units=hidden_units,
                         lr_ph=lr_ph, lr=lr)
        self.clf = sklearn.linear_model.BayesianRidge()

    def build(self):
        h = self.X
        for neurons_i in self.hidden_units:
            # h = tf.keras.layers.Dense(neurons_i, activation=tf.nn.tanh)(h)
            h = tf.keras.layers.Dense(neurons_i, activation=tf.nn.relu)(h)
        self.h = h
        h = tf.keras.layers.Dense(self.y_dim, activation=None)(h)
        return h

    def predict(self, sess, X, sample=True):
        return sess.run(self.yhat, feed_dict={self.X: X})

    def predict_posterior(self, sess, X, dm, n=30):
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
            self.clf.fit(sess.run(self.h, feed_dict={self.X: dm.X}), dm.Y[:, i].ravel())
            pred_mean_i, pred_std_i = self.clf.predict(sess.run(self.h, feed_dict={self.X: X}), return_std=True)
            pred_mean[:, i] = pred_mean_i
            pred_std[:, i] = pred_std_i

        return pred_mean, pred_std


class ConvNeuralLinear(CNN, BaseNetwork):
    # Convolutional neural linear - CNN with Bayesian on the last layer
    def __init__(self, dm, optimizer=None, n_channels=None, hidden_units=None,
                 lr_ph=None, lr=1e-4):

        self.h = None

        super().__init__(dm, optimizer=optimizer,
                         n_channels=n_channels, hidden_units=hidden_units, lr_ph=lr_ph, lr=lr)
        self.clf = sklearn.linear_model.BayesianRidge()

    def build(self):
        """Same as base CNN, but saving the output of the last hidden layer"""
        assert len(self.X.shape) == 4
        h = self.X
        filter_size = 3
        padding = filter_size - 1
        for i, channels_i in enumerate(self.n_channels):
            # Padding for periodic boundary conditions
            h = tf.concat((h, h[:, :padding, :, :]), axis=1)
            h = tf.concat((h, h[:, :, :padding, :]), axis=2)
            h = tf.keras.layers.Conv2D(filters=channels_i, kernel_size=filter_size, strides=(1, 1), padding='VALID',
                                       activation=tf.nn.relu)(h)
            # if i % 2 == 0:
            #     h = tf.keras.layers.AvgPool2D(strides=1)(h)
            # else:
            #     h = tf.keras.layers.MaxPool2D()(h)
            h = tf.keras.layers.MaxPool2D()(h)
        h = tf.keras.layers.Flatten()(h)
        for neurons_i in self.hidden_units:
            h = tf.keras.layers.Dense(neurons_i, activation=tf.nn.relu)(h)
        self.h = h
        h = tf.keras.layers.Dense(self.y_dim, activation=None)(h)
        return h

    def predict(self, sess, X, sample=True):
        return sess.run(self.yhat, feed_dict={self.X: X})

    def predict_posterior(self, sess, X, dm, n=30):
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
            self.clf.fit(sess.run(self.h, feed_dict={self.X: dm.X}), dm.Y[:, i].ravel())
            pred_mean_i, pred_std_i = self.clf.predict(sess.run(self.h, feed_dict={self.X: X}), return_std=True)
            pred_mean[:, i] = pred_mean_i
            pred_std[:, i] = pred_std_i

        return pred_mean, pred_std
