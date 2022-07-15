import numpy as np
import scattering.cross_section as cs
import argparse
from joblib import Parallel, delayed


class MieScattering:
    def __init__(self, n_layers=4, lam_min=350, lam_max=750, n_lam=201, th_min=30, th_max=70, n_mat=4, sample_mat=False,
                 n_cores=1):
        self.n_layers = n_layers
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.n_lam = n_lam
        self.th_min = th_min
        self.th_max = th_max
        self.n_mat = n_mat  # Number of materials to choose from
        self.sample_mat = sample_mat
        self.lam = np.linspace(lam_min, lam_max, n_lam)
        self.omega = 2*np.pi/self.lam
        self.n_cores = n_cores

        # Material properties
        self.eps_silica = 2.04   # core material
        self.eps_tio2 = 5.913 + 0.2441 / (self.lam ** 2 * 1e-6 - 0.0803)
        self.eps_water = 1.77   # particle is surrounded by water

        # Yang, Honghua U., et al. "Optical dielectric function of silver." Physical Review B 91.23 (2015): 235137.
        self.eps_ag = 5 - (8.9 / 6.58e-16) ** 2 / ((2 * np.pi * 3e8 / (self.lam * 1e-9)) ** 2 + (1 / 17e-15) ** 2)
        # tau = 1/Tau = 17 fs
        # hbar*wp = h * fp = 8.9  eV
        # h = 4.14e-15 eV*s, hbar = 6.58e-16 eV*s
        # eps_inf = 5

        # Gold: Olmon, Robert L., et al. "Optical dielectric function of gold." Physical Review B 86.23 (2012): 235147.
        self.eps_au = 1 - (8.5 / 6.58e-16) ** 2 / ((2 * np.pi * 3e8 / (self.lam * 1e-9)) ** 2 + (1 / 14e-15) ** 2)

        self.eps_zno = 8.5      # Zinc oxide ZnO
        self.eps_feo = 14.2     # iron oxide Fe2O3

        # ITO
        # self.eps_ito = 4 -

    def sample_x(self, n_data, onehot=False):
        """Generate random dataset features (X) of size n_data and feature size n_layers
        Used mostly for active learning purposes. If you want both X and Y, use gen_data"""

        x_cont = np.random.rand(n_data, self.n_layers)
        x_cont = x_cont * (self.th_max - self.th_min) + self.th_min
        if self.sample_mat:
            x_cat = np.random.randint(0, self.n_mat, (n_data, self.n_layers))
            if onehot:
                x_cat = self.toonehot(x_cat)
            x = np.concatenate((x_cat, x_cont), axis=1)
        else:
            x = x_cont
        return x

    def toonehot(self, x):
        """Convert NxM matrix t from categorical to one-hot encoding, where all features are categorical"""
        x_onehot = []
        for x_i in x:
            x_onehot.append(np.eye(self.n_mat)[x_i.astype(int)].ravel())
        return np.array(x_onehot)

    def to_nn_input(self, x):
        """Convert NxM matrix (generated from sample_x) into a form suitable for feeding into a neural network"""
        if self.sample_mat:
            x_norm = self.norm(x[:, self.n_mat:])  # Normalizing continuous variables
            x_onehot = self.toonehot(x[:, :self.n_mat])
            return np.concatenate((x_onehot, x_norm), axis=1)
        else:
            x_norm = self.norm(x)
            return x_norm

    def gen_data(self, n_data):
        """Generate dataset (X and Y) of size n_data"""
        th = self.sample_x(n_data)

        # # Version without joblib
        # spect = np.empty((n_data, self.n_lam))
        # for i in range(n_data):
        #     spect[i, :] = calc_spectrum(th[i, :])[0, :]
        # return th,spect

        # # Version using joblib
        # spect = Parallel(n_jobs=self.n_cores, verbose=1)(delayed(calc_spectrum)(i, self) for i in th)
        # return th, np.squeeze(np.array(spect)[:, 0, :])

        return th, self.calc_data(th)

    def calc_data(self, points, verbose=1):
        """Calculate spectra (data labels) for given thicknesses (data features)"""
        if np.ndim(points) == 2:
            # (n_data, n_layers) = np.shape(points)
            # spect = np.empty((n_data, n))
            # for i in range(n_data):
            #     spect[i, :] = calc_spectrum(points[i, :])[0, :]
            # return spect

            # Parallelize for larger queries
            if np.shape(points)[0] > 1000:
                spect = Parallel(n_jobs=self.n_cores, verbose=verbose)(delayed(self.calc_spectrum)(i) for i in points)
                spect_re = np.squeeze(np.array(spect)[:, 0, :])
                if np.size(points, 0) == 1 and spect_re.ndim == 1:  # if it's 1D, reshape back into a 2D array
                    spect_re = spect_re[np.newaxis, ...]
            else:
                (n_data, n_layers) = np.shape(points)
                spect_re = np.empty((n_data, self.n_lam))
                for i in range(n_data):
                    spect_re[i, :] = self.calc_spectrum(points[i, :])[0, :]
            return spect_re
        elif np.ndim(points) == 1:
            return self.calc_spectrum(points)[0, :]

    # def norm(self, X):
    #     """Normalize the data features to [-1, 1]"""
    #     return (X-(self.th_min+self.th_max)/2)/(self.th_max-self.th_min)*2
    #
    # def undo_norm(self, X):
    #     """Undo normalization"""
    #     return X*(self.th_max-self.th_min)/2 + (self.th_min+self.th_max)/2

    def norm(self, X):
        """Normalize the data features to [0, 1]"""
        return (X-self.th_min)/(self.th_max-self.th_min)

    def calc_spectrum(self, x):
        """Calculate spectra given list of thicknesses (one datapoint at a time)"""
        m = self.n_layers  # number of layers

        # permittivity
        eps = np.empty((m + 1, self.n_lam))
        if self.sample_mat:
            for i in range(m):
                if int(x[i]) == 0:
                    eps[i, :] = self.eps_silica
                elif int(x[i]) == 1:
                    eps[i, :] = self.eps_tio2
                elif int(x[i]) == 2:
                    # eps[i, :] = self.eps_ag
                    eps[i, :] = self.eps_zno
                elif int(x[i]) == 3:
                    eps[i, :] = self.eps_feo
                else:
                    raise ValueError('Material index must be integer 0-3')
        else:
            for i in range(m):
                if i % 2 == 0:
                    eps[i, :] = self.eps_silica
                else:
                    eps[i, :] = self.eps_tio2

        eps[-1, :] = self.eps_water

        if m <= 3:
            order = 4
        elif m <= 5:
            order = 9
        elif m <= 7:
            order = 12
        elif m <= 9:
            order = 15
        elif m <= 11:
            order = 18
        else:
            order = 25

        if self.sample_mat:
            th = x[self.n_layers:]
        else:
            th = x

        return cs.total_cs(th, self.omega, eps, order) / (np.pi * sum(th) ** 2)


default_params = MieScattering()


def single_test():
    """Calculate single spectrum. Use this function for testing purposes."""
    # r = np.random.randint(th_min*10, th_max*10, 2)/10.0
    r = [50, 65]
    # scatter = MieScattering(n_layers=2)
    # spect = calc_spectrum(r)[0, :]
    #
    # return r, spect


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scattering data")
    parser.add_argument("--mode", type=str, choices=['test', 'gen', 'grid', 'time'], default="test")
    parser.add_argument("-n", type=int, default=50000)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default='data')
    parser.add_argument("--lam_min", type=int, default=350)
    parser.add_argument("--lam_max", type=int, default=750)
    parser.add_argument("--n_lam", type=int, default=201)
    parser.add_argument("--th_min", type=int, default=30)
    parser.add_argument("--th_max", type=int, default=70)

    args = parser.parse_args()
    print(args)

    scatter = MieScattering(n_layers=6)
    x = scatter.sample_x(2)
    print(x)
    #
    # print(scatter.toonehot(x[:, :6]))
    print(scatter.calc_data(x))
    data = scatter.calc_data(x)

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
    plt.figure(figsize=(4,3))
    plt.plot(scatter.lam, data[0, :], color='k')
    plt.xlabel('Wavelength [nm]')
    plt.xlim([350, 750])
    plt.ylabel('Scattering cross-section $\sigma/\pi r^2$')
    plt.tight_layout()
    plt.show()

    # if args.mode == 'test':
    #     single_test()
    # elif args.mode == 'time':
    #     scatter = MieScattering(n_layers=args.n_layers, lam_min=args.lam_min, lam_max=args.lam_max,
    #                             n_lam=args.n_lam, th_min=args.th_min, th_max=args.th_max, n_cores=args.n_cores)
    #     import time
    #     t0 = time.time()
    #     scatter.gen_data(args.n)
    #     tf = time.time()
    #     print(tf-t0)
    #
    # else:
    #     scatter = MieScattering(n_layers=args.n_layers, lam_min=args.lam_min, lam_max=args.lam_max,
    #                             n_lam=args.n_lam, th_min=args.th_min, th_max=args.th_max, n_cores=args.n_cores)
    #     # start = time.time()
    #     th, spect = scatter.gen_data(args.n)
    #     # end = time.time()
    #     # print(end - start)
    #     print(np.shape(spect))
