import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class FourierLevelSet:
    def __init__(self, uc_level=0.2, nf=2, eps_in=1.0, eps_out=11.4, res=32):
        self.uc_level = uc_level
        self.nf = nf    # Number of Fourier components. (-n, -n+1, ..., 0, ..., n)
        self.eps_in = eps_in
        self.eps_out = eps_out

        # ((nf^2+1)=25, 2) array of G vectors in k-space
        self.uc_gvecs = []
        for xi in np.linspace(-nf, nf, nf * 2 + 1):
            for yi in np.linspace(-nf, nf, nf * 2 + 1):
                self.uc_gvecs.append([xi, yi])
        self.uc_gvecs = np.array(self.uc_gvecs)

        # Position vectors
        x_plot = np.linspace(-0.5, 0.5, res)
        y_plot = np.linspace(-0.5, 0.5, res)
        # Shape: (res, res)
        xv, yv = np.meshgrid(x_plot, y_plot)

        # For non-TensorFlow calculation
        # Dot product of uc_gvecs and r
        q = self.uc_gvecs[:, 0, np.newaxis, np.newaxis] * xv[np.newaxis] + \
            self.uc_gvecs[:, 1, np.newaxis, np.newaxis] * yv[np.newaxis]
        self.q = 2 * np.pi * q  # shape (len(uc_gvecs), res, res)

    def calc_data(self, x, sample_level=True, level_set=True, sample_full=False):
        """

        Args:
            x: parameterization of geometry, shape = (batch_dim, n_coefs) where N is batch size
            sample_level: if x[:, 0] is the level
            level_set: return level_set or raw data

        Return:
            images, shape = (batch_dim, res, res). Note, you need to add a 4th dimension to use with TensorFlow
            convolution
        """

        if not sample_level:
            n_coefs = x.shape[1] // 2
            if sample_full:
                x = x * 2 - 1
            uc_coefs_re = x[:, :n_coefs, np.newaxis, np.newaxis]
            uc_coefs_im = x[:, n_coefs:, np.newaxis, np.newaxis]

            q = self.q[np.newaxis, :, :, :]
            level = np.sum(-uc_coefs_re * np.sin(q) + uc_coefs_im * np.cos(q), axis=1)

            if level_set:
                images = np.where(level < self.uc_level, self.eps_in, self.eps_out)
            else:
                images = level
        else:
            uc_level = x[:, 0] * 6 - 3
            x = x[:, 1:]
            if sample_full:
                x = x * 2 - 1
            n_coefs = x.shape[1] // 2
            uc_coefs_re = x[:, :n_coefs, np.newaxis, np.newaxis]
            uc_coefs_im = x[:, n_coefs:, np.newaxis, np.newaxis]

            q = self.q[np.newaxis, :, :, :]
            level = np.sum(-uc_coefs_re * np.sin(q) + uc_coefs_im * np.cos(q), axis=1)

            if level_set:
                images = np.where(np.less(level, uc_level[:, np.newaxis, np.newaxis]),
                                  self.eps_in, self.eps_out)
            else:
                images = level
        return images


def plot_level():
    # Plot samples of random unit cells as a function of uc_level
    f, axes = plt.subplots(5, 10)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        for i, level_i in enumerate(np.linspace(-2, 2, 10)):
            leveller = FourierLevelSet(uc_level=level_i)
            x = np.random.rand(5, 50)
            images = leveller.calc_data(x, sess)
            for j in range(5):
                axes[j, i].imshow(images[j, :, :, 0], cmap='gray', origin='lower')
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([])
        plt.show()


def plot_samples():
    """Plot samples of randomly generated unit cells"""
    f, axes = plt.subplots(2, 5)

    leveller = FourierLevelSet()
    x = np.random.rand(10, 51)
    images = leveller.calc_data(x, sample_level=True, sample_full=True)
    for i in range(10):
        axis_i = axes.ravel()[i]
        axis_i.imshow(images[i, :, :], cmap='gray', origin='lower')
        axis_i.set_xticks([])
        axis_i.set_yticks([])
    plt.show()


if __name__ == "__main__":
    # plot_level()
    plot_samples()
