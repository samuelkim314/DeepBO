import scipy.special as sp
import numpy as np


def besselj_d(m, x):
    return (sp.jv(m-1, x) - sp.jv(m+1, x))/2


def bessely_d(m, x):
    return (sp.yv(m-1, x) - sp.yv(m+1, x))/2


def sbesselj(m, x):
    # TODO: replace with sp.spherical_jn?
    return np.divide(sp.jv(m + 1/2, x), np.sqrt(x))


def sbessely(m, x):
    return np.divide(sp.yv(m + 1/2, x), np.sqrt(x))


def sbesselj_d(m, x):
    return (m*sbesselj(m-1, x) - (m+1)*sbesselj(m+1, x))/(2*m+1)


def sbessely_d(m, x):
    return (m * sbessely(m - 1, x) - (m + 1) * sbessely(m + 1, x)) / (2 * m + 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.linspace(0.1, 10)
    plt.plot(x, sbesselj(1, x)*np.sqrt(np.pi/2))
    plt.plot(x, sp.spherical_jn(1, x))
    plt.show()

