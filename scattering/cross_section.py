import numpy as np
from . import bessel as bs


def total_cs(a, omega, eps, order=3):
    """Total cross-section of a spherical multi-layer particle

    Returns an N-by-2 matrix containing teh total scattering cross-section in the first column and total absorption
    cross-section in the second column, where 'total' here means summing over TE and TM for l = 1, 2, 3.

    Input arguments:
        a: 1-by-K row vector specifying teh thickness for each layer of the particle, starting from the inner-most
        layer. a(1) is the radius of the core, a(2) is the thickness of the first coating (NOT the radius), etc.

        omega: N-by-1 column vector specifying the frequencies at which to evaluate the cross-sections.

        eps: N-by-(K+1) matrix specifying the relative permittivity, such that eps[:, 0] are for the core at the
        frequencies given by omega, eps[:, 1] for the first coating, etc, and eps[:, K+1] for the medium where the
        particle sits in.

    Unit convention: suppose the input is in in unit of nm, then the returned cross-sections are in unit of nm^2, and
    the input omega is in unit of 2*pi/lambda, where lambda is free-space wavelength in units of nm. The same goes when
    a is in some other unit of length.

    2012 Wenjun Qiu @ MIT
    Ported to Python by Sam Kim 2017 @ MIT"""

    sigma = 0
    for l in range(order):
        sigma = sigma + spherical_cs(1, l+1, a, omega, eps) + spherical_cs(2, l+1, a, omega, eps)
    return sigma


def product(a, b):
    a = np.transpose(a)
    b = np.transpose(b)
    return np.vstack((a[:, 0]*b[:, 0] + a[:, 2]*b[:, 1], a[:, 1]*b[:, 0] + a[:, 3]*b[:, 1],
                     a[:, 0]*b[:, 2] + a[:, 2]*b[:, 3], a[:, 1]*b[:, 2] + a[:, 3]*b[:, 3]))


def spherical_TM1(k, l, a, omega, eps1, eps2):
    k1 = omega * np.sqrt(eps1)
    k2 = omega * np.sqrt(eps2)
    x1 = k1 * a
    x2 = k2 * a

    j1 = bs.sbesselj(l, x1)
    j1_d = bs.sbesselj_d(l, x1)*x1 + j1
    y1 = bs.sbessely(l, x1)
    y1_d = bs.sbessely_d(l, x1)*x1 + y1
    j2 = bs.sbesselj(l, x2)
    j2_d = bs.sbesselj_d(l, x2) * x2 + j2
    y2 = bs.sbessely(l, x2)
    y2_d = bs.sbessely_d(l, x2) * x2 + y2

    if k == 1:
        M = product(np.vstack((y2_d, -j2_d, -y2, j2)), np.vstack((j1, j1_d, y1, y1_d)))
    else:
        M = product(np.vstack((eps1*y2_d, -eps1*j2_d, -y2, j2)), np.vstack((j1, eps2*j1_d, y1, eps2*y1_d)))

    return M


def spherical_TM2(k, l, a, omega, eps):
    # print(a)
    a = np.cumsum(a)    # TODO: Check if flattened array is alright or if it needs to be along axis
    # print(a)
    (K, N) = eps.shape
    K = K - 1   # Number of layers
    M = np.transpose(np.hstack((np.ones((N, 1)), np.zeros((N, 2)), np.ones((N, 1)))))

    for i in range(K):
        tmp = spherical_TM1(k, l, a[i], omega, eps[i, :], eps[i+1, :])
        M = product(tmp, M)
    return M


def spherical_cs(k, l, a, omega, eps):
    M = spherical_TM2(k, l, a, omega, eps)
    tmp = M[0, :]/M[1, :]
    R = (tmp - 1j)/(tmp + 1j)
    coef = (2*l + 1)*np.pi/(2*omega**2*eps[-1, :])
    sigma = np.tile(coef, (2, 1)) * np.vstack((abs(1-R)**2, 1-abs(R)**2))

    return sigma