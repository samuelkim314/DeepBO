import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
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


class FourierLevelSetTF:
    """Same as FourierLevelSet, but using Tensorflow for GPU acceleration"""
    def __init__(self, uc_level=0.2, nf=2, eps_in=1.0, eps_out=10.0):
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
        x_plot = np.linspace(-0.5, 0.5, 32)
        y_plot = np.linspace(-0.5, 0.5, 32)
        # Shape: (32, 32)
        xv, yv = np.meshgrid(x_plot, y_plot)

        r_arr = np.stack((xv, yv), axis=2)  # Shape (32, 32, 2)
        q = 2 * np.pi * np.einsum('ijk,lk->lij', r_arr, self.uc_gvecs)  # Shape (len(uc_gvecs), 32, 32)
        q = tf.cast(q, tf.float32)

        self.coefs_real = tf.placeholder(tf.float32, (None, 25))     # Shape (batch_dim, len(uc_gvecs))
        self.coefs_imag = tf.placeholder(tf.float32, (None, 25))
        self.level = tf.einsum('ij,jlm->ilm', -self.coefs_real, tf.sin(q)) + \
                     tf.einsum('ij,jlm->ilm', self.coefs_imag, tf.cos(q))

    def calc_data(self, c, sess, sample_level=False, sample_eps=False, level_set=True):
        """Calculate level-set image given Fourier coefficients x using TensorFlow
        Uses einsum, so it is significantly faster and less memory-intensive than multiplying large matrices.

        :param c: complex Fourier coefficients. First half are real components, second half imaginary.
        :param sess:
        :return:
        """
        if not sample_eps and not sample_level:
            n_coefs = c.shape[1] // 2

            cr_i = c[:, :n_coefs]
            ci_i = c[:, n_coefs:]

            image = sess.run(self.level, feed_dict={self.coefs_real: cr_i, self.coefs_imag: ci_i})
            if level_set:
                image = np.where(image < self.uc_level, self.eps_in, self.eps_out)
        elif sample_level and not sample_eps:
            uc_level = c[:, 0] * 4 - 2
            c = c[:, 1:]

            n_coefs = c.shape[1] // 2
            cr_i = c[:, :n_coefs]
            ci_i = c[:, n_coefs:]

            image = sess.run(self.level, feed_dict={self.coefs_real: cr_i, self.coefs_imag: ci_i})
            if level_set:
                image = np.where(np.less(image, uc_level[:, np.newaxis, np.newaxis]),
                                 self.eps_in, self.eps_out)
        else:
            uc_level = c[:, 0] * 4 - 2
            eps_out = c[:, 1] * 9 + 1
            c = c[:, 2:]

            n_coefs = c.shape[1] // 2
            cr_i = c[:, :n_coefs]
            ci_i = c[:, n_coefs:]

            image = sess.run(self.level, feed_dict={self.coefs_real: cr_i, self.coefs_imag: ci_i})
            if level_set:
                image = np.where(np.less(image, uc_level[:, np.newaxis, np.newaxis]),
                                 self.eps_in, eps_out[:, np.newaxis, np.newaxis])
        return image[:, :, :, np.newaxis]


class FourierLevelSetOld:
    """Calculate Fourier level set. These are using deprecated methods and are kept just for comparison. These may also
    be easier to read, since the optimized version uses einsum."""
    def __init__(self, uc_level=0.2, nf=2, eps_in=1.0, eps_out=10.0):
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

        self.x_plot = np.linspace(-0.5, 0.5, 32)
        self.y_plot = np.linspace(-0.5, 0.5, 32)
        # Shape: (32, 32)
        self.xv, self.yv = np.meshgrid(self.x_plot, self.y_plot)

        # Dot product of uc_gvecs and r
        q = self.uc_gvecs[:, 0, np.newaxis, np.newaxis] * self.xv[np.newaxis] + \
            self.uc_gvecs[:, 1, np.newaxis, np.newaxis] * self.yv[np.newaxis]
        self.q = 2 * np.pi * q  # shape (len(uc_gvecs), 32, 32)

        # Setting up TensorFlow variables
        q = np.float32(self.q[np.newaxis, :, :, :])     # Shape (1, len(uc_gvecs), 32, 32)
        self.xr = tf.placeholder(tf.float32, (None, 25, 1, 1))
        self.xi = tf.placeholder(tf.float32, (None, 25, 1, 1))
        series = -self.xr * tf.sin(q) + self.xi * tf.cos(q)
        self.level = tf.reduce_sum(series, axis=1)

        r_arr = np.stack((self.xv, self.yv), axis=2)  # Shape (32, 32, 2)
        q = 2 * np.pi * np.einsum('ijk,lk->lij', r_arr, self.uc_gvecs)  # Shape (len(uc_gvecs), 32, 32)
        q = tf.cast(q, tf.float32)

        self.coefs_real = tf.placeholder(tf.float32, (None, 25))     # Shape (batch_dim, len(uc_gvecs))
        self.coefs_imag = tf.placeholder(tf.float32, (None, 25))
        self.level2 = tf.einsum('ij,jlm->ilm', -self.coefs_real, tf.sin(q)) + \
                      tf.einsum('ij,jlm->ilm', self.coefs_imag, tf.cos(q))

    def level_set(self, uc_coefs, r):
        """Calculate epsilon profile based on level-set of the Fourier sum"""
        # Calculate real part of the Fourier sum
        q = 2 * np.pi * np.dot(self.uc_gvecs, r)
        fourier_sum = np.sum(-np.real(uc_coefs) * np.sin(q) + np.imag(uc_coefs) * np.cos(q))
        # Level set
        return self.eps_in if fourier_sum < self.uc_level else self.eps_out

    def image_iter(self, uc_coefs):
        """Calculate the level set for an image, iterating pixel by pixel"""
        eps_arr = np.zeros((32, 32))
        for ix in range(32):
            for iy in range(32):
                eps_arr[iy, ix] = self.level_set(uc_coefs, [self.x_plot[ix], self.y_plot[iy]])
        return eps_arr

    def calc_data(self, x):
        """x: parameterization of geometry, shape = (batch_dim, n_coefs) where N is batch size"""
        n_coefs = x.shape[1] // 2
        uc_coefs_arr = x[:, :n_coefs] + 1j * x[:, n_coefs:]

        uc_coefs = uc_coefs_arr[:, :, np.newaxis, np.newaxis]
        q = self.q[np.newaxis, :, :, :]
        level = np.sum(-np.real(uc_coefs) * np.sin(q) + np.imag(uc_coefs) * np.cos(q), axis=1)
        images = np.where(level < self.uc_level, self.eps_in, self.eps_out)

        return images[:, :, :, np.newaxis]

    def calc_data_old(self, x):
        """x: parameterization of geometry, shape = (batch_dim, n_coefs) where N is batch size

        Note this function is deprecated. It is very slow but easy to read."""
        n_coefs = x.shape[1] // 2
        uc_coefs_arr = x[:, :n_coefs] + 1j * x[:, n_coefs:]
        images = []
        for uc_coef in uc_coefs_arr:
            images.append(self.image_iter(uc_coef))
        return np.array(images)[:, :, :, np.newaxis]

    def calc_data_tf_old(self, x, sess):
        n_coefs = x.shape[1] // 2

        xr_i = x[:, :n_coefs, np.newaxis, np.newaxis]
        xi_i = x[:, n_coefs:, np.newaxis, np.newaxis]

        # max ~ 4e4?

        image = sess.run(self.level, feed_dict={self.xr: xr_i, self.xi: xi_i})
        image = np.where(image < self.uc_level, self.eps_in, self.eps_out)
        return image[:, :, :, np.newaxis]

    def calc_data_tf(self, c, sess):
        """Calculate level-set image given Fourier coefficients x
        Uses einsum, so it is significantly faster and less memory-intensive than multiplying large matrices.

        :param c: complex Fourier coefficients. First half are real components, second half imaginary.
        :param sess:
        :return:
        """
        n_coefs = c.shape[1] // 2

        cr_i = c[:, :n_coefs]
        ci_i = c[:, n_coefs:]

        image = sess.run(self.level2, feed_dict={self.coefs_real: cr_i, self.coefs_imag: ci_i})
        image = np.where(image < self.uc_level, self.eps_in, self.eps_out)
        return image[:, :, :, np.newaxis]


def compare_methods():
    leveller = FourierLevelSet(uc_level=0.2)
    # uc_coefs = np.random.rand(2, 25) + 1j * np.random.rand(2, 25)
    x = np.random.rand(int(4e4), 50)
    # image1 = leveller.calc_data(x)

    # t1 = time.time()
    # leveller.calc_data(x)
    # print(time.time() - t1)
    #
    # t1 = time.time()
    # leveller.calc_data_old(x)
    # print(time.time() - t1)

    # def fun():
    #     leveller.calc_data(x)
    # cProfile.run('fun()')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        image1 = leveller.calc_data(x, sess)
        #
        # t1 = time.time()
        # leveller.calc_data_tf(x, sess)
        # print(time.time() - t1)
        #
        # input()

        # t1 = time.time()
        # leveller.calc_data_tf(x, sess)
        # print(time.time() - t1)
        #
        # input()

        # def fun():
        #     leveller.calc_data_tf(x, sess)
        # cProfile.run('fun()')

    f, axes = plt.subplots(2, 3)
    axes[0, 0].imshow(image1[0, :, :, 0], cmap='gray', origin='lower')
    axes[1, 0].imshow(image1[1, :, :, 0], cmap='gray', origin='lower')
    # axes[0, 1].imshow(image2[0, :, :, 0], cmap='gray', origin='lower')
    # axes[1, 1].imshow(image2[1, :, :, 0], cmap='gray', origin='lower')
    # axes[0, 2].imshow(image3[0, :, :, 0], cmap='gray', origin='lower')
    # axes[1, 2].imshow(image3[1, :, :, 0], cmap='gray', origin='lower')
    plt.show()
    plt.savefig('fig')


def plot_level():
    # Plot as a function of uc_level
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


def test_eps():
    f, axes = plt.subplots(1, 20)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        leveller = FourierLevelSet()
        x = np.random.rand(20, 52)
        images = leveller.calc_data(x, sess, sample_eps=True)
        for i in range(20):
            axes[i].imshow(images[i, :, :, 0], cmap='gray', origin='lower')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        plt.show()


def plot_level2():
    f, axes = plt.subplots(2, 5)

    leveller = FourierLevelSet()
    x = np.random.rand(10, 51) * 2 - 1
    images = leveller.calc_data(x, sample_level=True)
    for i in range(10):
        axis_i = axes.ravel()[i]
        axis_i.imshow(images[i, :, :], cmap='gray', origin='lower')
        axis_i.set_xticks([])
        axis_i.set_yticks([])
    plt.show()


def plot_level_full():
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


def plot_level3():
    f, axes = plt.subplots(2, 5)

    leveller = FourierLevelSet(res=32+2*5)
    x = np.random.rand(10, 50)
    images = leveller.calc_data(x, sample_level=False, level_set=False)
    for i in range(10):
        axis_i = axes.ravel()[i]
        axis_i.imshow(images[i, :, :], cmap='gray', origin='lower')
        axis_i.set_xticks([])
        axis_i.set_yticks([])
        # axis_i.colorbar()
    plt.show()


if __name__ == "__main__":
    # plot_level()
    plot_level_full()
