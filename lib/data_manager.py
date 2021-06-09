import numpy as np


class DataManager:
    """Manages data for training by splitting it into mini-batches

    Attributes:
        X, Y            feature and label data
        batch_size      mini-batch size
        x_dim, y_dim    dimensionality of X, Y
        steps           number of batches in the dataset (rounded up)"""
    def __init__(self, X, Y, batch_size, Z=None):
        # Ensure that Y and Z have 2 dimensions
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        if Z is not None and Z.ndim == 1:
            Z = Z[:, np.newaxis]

        self.X = X
        self.Y = Y
        self.Z = Z  # shape (n, Z_dim)
        self.batch_size = batch_size
        self.x_dim = X.shape[1]
        self.Y_dim = Y.shape[1]

        # y_dim is the feature size of data that is fed to model
        if Z is not None:
            self.Z_dim = Z.shape[1]
            self.y_dim = self.Z_dim
        else:
            self.y_dim = self.Y_dim

        # The following 3 variables are defined in self.update()
        self.n = None       # Number of data points
        self.batch_ind = None
        self.steps = None
        self.update()

    def batch_iter(self, shuffle=True):
        """Generator for batches

        Args:
            shuffle (bool):     flag to shuffle data in each epoch
        """
        batch_size = self.batch_size
        X = self.X
        if self.Z is not None:
            Y = self.Z
        else:
            Y = self.Y
        while True:
            # shuffle X and Y
            i_arr = np.arange(0, self.n)
            if shuffle:
                np.random.shuffle(i_arr)
            X_shuff = X[i_arr]
            Y_shuff = Y[i_arr]
            for i_batch in self.batch_ind:
                X_batch = X_shuff[i_batch:i_batch+batch_size]
                Y_batch = Y_shuff[i_batch:i_batch+batch_size]
                yield X_batch, Y_batch

    def get_data(self, i):
        """Get rows from index i. i can be a scalar, list, or numpy array. Even if i is a scalar, will return an array
        with the first dimension being the batch_size"""
        if not (isinstance(i, list) or isinstance(i, np.ndarray)):
            i = [i]
        if self.Z is None:
            return self.X[i], self.Y[i]
        else:
            return self.X[i], self.Y[i], self.Z[i]

    def add_data(self, X, Y):
        """Add data"""
        assert X.shape[1] == self.x_dim
        assert Y.shape[1] == self.y_dim
        self.X = np.vstack((self.X, X))
        self.Y = np.vstack((self.Y, Y))
        self.update()

    def replace_data(self, X, Y):
        """Remove old data and add new data"""
        assert X.shape[1] == self.x_dim
        assert Y.shape[1] == self.y_dim
        self.X = X
        self.Y = Y
        self.update()

    def remove_data(self, i):
        self.X = np.delete(self.X, i, axis=0)
        self.Y = np.delete(self.Y, i, axis=0)
        if self.Z is not None:
            self.Z = np.delete(self.Z, i, axis=0)
        self.update()

    def get_best(self):
        i_best = np.argmax(self.Y)
        return self.X[i_best], self.Y[i_best]

    def get_arg_best(self):
        i_best = np.argmax(self.Y)
        return i_best

    def update(self):
        self.n = self.X.shape[0]
        self.batch_ind = range(0, self.n, self.batch_size)
        self.steps = len(self.batch_ind)


class ImageDataManager(DataManager):
    def __init__(self, X, Y, batch_size):
        super().__init__(X, Y, batch_size)
        self.width = X.shape[1]
        self.height = X.shape[2]
        self.n_channels = X.shape[3]


class ChemDataManager:
    """Manages chemistry graph data for training by splitting it into mini-batches

        Attributes:
            X, A, E, Y      feature and label data
            Z               label data that is used for training the model. if this is not None, then this is fed to the
                            model during training, and Y is the objective function data.
            batch_size      mini-batch size
            x_dim, y_dim    dimensionality of X, Y
            steps           number of batches in the dataset"""
    def __init__(self, X, A, E, Y, batch_size=32, Z=None):
        # Ensure that Y and Z have 2 dimensions
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        if Z is not None and Z.ndim == 1:
            Z = Z[:, np.newaxis]

        self.X = X  # shape (n, N, F)
        self.Y = Y  # shape (n, Y_dim)
        self.A = A  # shape (n, N, N)
        self.E = E  # shape (n, N, N, S)
        self.Z = Z  # shape (n, Z_dim)

        self.N = X.shape[-2]    # Number of nodes in the graphs
        self.F = X.shape[-1]    # Node features dimensionality
        self.S = E.shape[-1]    # Edge features dimensionality
        self.Y_dim = Y.shape[1]

        self.batch_size = batch_size

        # y_dim is the feature size of data that is fed to model
        if Z is not None:
            self.Z_dim = Z.shape[1]
            self.y_dim = self.Z_dim
        else:
            self.y_dim = self.Y_dim

        # The following 3 variables are defined in self.update()
        self.n = None
        self.batch_ind = None
        self.steps = None
        self.update()

    def batch_iter(self, shuffle=True):
        """Generator for batches

        Args:
            shuffle (bool):     flag to shuffle data in each epoch
        """
        batch_size = self.batch_size
        X = self.X
        A = self.A
        E = self.E
        if self.Z is not None:
            Y = self.Z
        else:
            Y = self.Y

        while True:
            # shuffle X and Y
            i_arr = np.arange(0, self.n)
            if shuffle:
                np.random.shuffle(i_arr)
            X_shuff = X[i_arr]
            A_shuff = A[i_arr]
            E_shuff = E[i_arr]
            Y_shuff = Y[i_arr]
            for i_batch in self.batch_ind:
                X_batch = X_shuff[i_batch:i_batch+batch_size]
                A_batch = A_shuff[i_batch:i_batch+batch_size]
                E_batch = E_shuff[i_batch:i_batch+batch_size]
                Y_batch = Y_shuff[i_batch:i_batch+batch_size]
                yield X_batch, A_batch, E_batch, Y_batch

    def get_data(self, i):
        """Get rows from index i. i can be a scalar, list, or numpy array. Even if i is a scalar, will return an array
        with the first dimension being the batch_size"""
        if not (isinstance(i, list) or isinstance(i, np.ndarray)):
            i = [i]
        if self.Z is None:
            return self.X[i], self.A[i], self.E[i], self.Y[i]
        else:
            return self.X[i], self.A[i], self.E[i], self.Y[i], self.Z[i]

    def add_data(self, X, A, E, Y, Z=None):
        """Add data"""
        assert X.shape[-2] == self.N
        assert X.shape[-1] == self.F
        assert E.shape[-1] == self.S
        assert Y.shape[1] == self.Y_dim
        self.X = np.vstack((self.X, X))
        self.A = np.vstack((self.A, A))
        self.E = np.vstack((self.E, E))
        self.Y = np.vstack((self.Y, Y))
        if Z is not None:
            assert Z.shape[1] == self.Z_dim
            self.Z = np.vstack((self.Z, Z))
        self.update()

    def replace_data(self, X, A, E, Y, Z=None):
        """Remove old data and add new data"""
        assert X.shape[-2] == self.N
        assert X.shape[-1] == self.F
        assert E.shape[-1] == self.S
        assert Y.shape[1] == self.Y_dim
        self.X = X
        self.A = A
        self.E = E
        self.Y = Y
        if Z is not None:
            assert Z.shape[1] == self.Z_dim
            self.Z = Z
        self.update()

    def remove_data(self, i):
        self.X = np.delete(self.X, i, axis=0)
        self.A = np.delete(self.A, i, axis=0)
        self.E = np.delete(self.E, i, axis=0)
        self.Y = np.delete(self.Y, i, axis=0)
        if self.Z is not None:
            self.Z = np.delete(self.Z, i, axis=0)
        self.update()

    def get_best(self):
        i_best = np.argmax(self.Y)
        return self.X[i_best], self.A[i_best], self.E[i_best], self.Y[i_best]

    def update(self):
        self.n = self.X.shape[0]
        self.batch_ind = range(0, self.n, self.batch_size)
        self.steps = len(self.batch_ind)
