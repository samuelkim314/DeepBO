#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:26:08 2020

@author: charlotte
"""

import meep as mp
from meep import mpb
import numpy as np
from scipy.ndimage import gaussian_filter1d as gaussfil


def converttovec3(klist, dim):
    """Converts list to a meep Vector3 object"""
    if dim == 2:
        kvec3 = []
        for i in range(len(klist)):
            kpoint = mp.Vector3(klist[i][0], klist[i][1])
            kvec3.append(kpoint)
    elif dim == 3:
        kvec3 = []
        for i in range(len(klist)):
            kpoint = mp.Vector3(klist[i][0], klist[i][1], klist[i][2])
            kvec3.append(kpoint)
    else:
        raise ValueError('Dimension must be 2 or 3')
    return kvec3


def convertfromvec3(vector3):
    """Convert Vector3 object to numpy array"""
    return np.array([vector3.x, vector3.y, vector3.z])


def runmpb(run_type="tm", dim=2, res=32, kvecs=None, nbands=10, rvecs=None, eps_mat=None):
    """
    Keyword arguments:
    run_type: string "te" or "tm" that describe which solver to use
    dim:    dimensionality of the problem (2 or 3)
    res:    resolution
    kvecs:  list of k-vectors to evaluate at
    nbands: number of bands to compute
    rvecs:  (dim, dim) array of lattice basis vectors
    """

    if dim == 2:
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1), basis1=rvecs[0], basis2=rvecs[1])
    elif dim == 3:
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1, 1), basis1=rvecs[0], basis2=rvecs[1], basis3=rvecs[2])
    else:
        raise ValueError('Missing argument for dimension')

    grpvel_k = []

    def grpvel(ms):
        gv3 = ms.compute_group_velocities()
        gv = []
        for gvband in gv3:
            gv.append(list(convertfromvec3(gvband)))
        grpvel_k.append(gv)

    # To define the epsilon profile, we define a pixel at every point with a specified epsilon
    pixel_list = []
    eps_n = eps_mat.shape[0]
    for (i, j), eps_i in np.ndenumerate(eps_mat):
        pixel_list.append(mp.Block(center=mp.Vector3(i / eps_n - 0.5, j / eps_n - 0.5),
                                   size=mp.Vector3(1 / eps_n, 1 / eps_n),
                                   material=mp.Medium(epsilon=eps_i)))

    ms = mpb.ModeSolver(num_bands=nbands,
                        k_points=kvecs,
                        geometry_lattice=geometry_lattice,
                        resolution=res,)
    ms.geometry = pixel_list
    if run_type == 'tm':
        ms.run_tm(grpvel)
    elif run_type == 'te':
        ms.run_te(grpvel)
    else:
        raise ValueError('Please specify polarization')

    efreq_k = ms.all_freqs
    gap_k = ms.gap_list
    eps = ms.get_epsilon()

    return efreq_k, gap_k, grpvel_k, eps


def calc_DOS(dataw, datav_original, kin=25, N_w=20000):
    # denotes the resolution of frequency : dw = (w_max - w_min) / N_w

    # Settings for DOS
    ceilDOS = 1  # impose ceiling on DOS array (1) or not (0)
    maxDOS = 40
    w_max_custom = -1  # the range of frequency, '-1' denotes default settings
    w_min_custom = 0

    # Input parameters here
    N_band = 10

    # reciprocal vectors for square lattice
    reciprocal_vector1 = [1, 0, 0]
    reciprocal_vector2 = [0, 1, 0]
    reciprocal_vector3 = [0, 0, 0]

    num_kpoints = [kin, kin, 1]

    # Initialization and import data
    # --------------------
    # the reciprocal vectors initialization
    vectorb1 = reciprocal_vector1
    vectorb2 = reciprocal_vector2
    vectorb3 = reciprocal_vector3
    vectorsb = [vectorb1, vectorb2, vectorb3]

    # Nx,Ny,Nz is the number of k points along the x,y,z axis. N_kpoints is the
    # total number of k points in Brillouin zone
    n_kpoints = np.prod(num_kpoints)
    N_kpoints = N_band*n_kpoints

    # import data
    # the two importing txt files are arranged as matrix of N*1 and N*3
    datav = np.transpose(np.dot(vectorsb, np.transpose(datav_original)))     # the transformed group velocities

    if w_max_custom == -1:
        w_max = 1.05*max(dataw)   # the maximum of frequency should be larger than max(dataw) a little
    else:
        w_max = w_max_custom

    if w_min_custom == -1:
        w_min = 0
    else:
        w_min = w_min_custom

    itmd_v = np.sort(abs(datav), axis=1)[:, ::-1]   # intermediate velocity: v1 >= v2 >= v3

    # N_w=20*N_kpoints       # divide the frequency region into N_w part
    step_w = (w_max-w_min)/N_w      # the resolution of frequency
    hside = 1/num_kpoints[1]/2    # half of side length of one transformed cube
    DOSarray = np.zeros(N_w+1)       # initialize the density of states array

    w1 = hside*abs(itmd_v[:, 0]-itmd_v[:, 1]-itmd_v[:, 2])
    w2 = hside*(itmd_v[:, 0]-itmd_v[:, 1]+itmd_v[:, 2])
    w3 = hside*(itmd_v[:, 0]+itmd_v[:, 1]-itmd_v[:, 2])
    w4 = hside*(itmd_v[:, 0]+itmd_v[:, 1]+itmd_v[:, 2])

    # DOS calculation ----------------------
    # principle of calculation process can be found in our article
    # "Generalized Gilat-Raubenheimer Method for Density-of-States Calculation
    # in Photonic Crystals"

    # Ignore divide by 0 warnings. These result in infinity values, which are then clipped out
    with np.errstate(divide='ignore', invalid='ignore'):
        for num_k in range(N_kpoints):
            n_w_kcenter = round((dataw[num_k]-w_min)/step_w)
            v = np.linalg.norm(datav[num_k, :])
            v1 = itmd_v[num_k, 0]
            v2 = itmd_v[num_k, 1]
            v3 = itmd_v[num_k, 2]

            flag_delta_n_w = 0       # first time compute delta_n_w = 1
            for vdirection in range(2):      # two velocity directions denote w-w_k0 > 0 and <0
                delta_n_w_arr = np.arange(N_w)
                n_tmpt = n_w_kcenter + (-1) ** vdirection * delta_n_w_arr
                delta_w = abs(dataw[num_k] - (n_tmpt * step_w + w_min))

                DOScontribution = np.where(delta_w <= w1[num_k],
                                           np.where(v1 >= v2+v3,
                                                    4*hside**2/v1,
                                                    (2*hside**2*(v1*v2+v2*v3+v3*v1) - (delta_w**2+(hside*v)**2))/v1/v2/v3),
                                           np.where(delta_w < w2[num_k],
                                                    (hside**2*(v1*v2+3*v2*v3+v3*v1) -
                                                     hside*delta_w*(-v1+v2+v3)-(delta_w**2+hside**2*v**2)/2)/v1/v2/v3,
                                                    np.where(delta_w < w3[num_k],
                                                             2*(hside**2*(v1+v2)-hside*delta_w)/v1/v2,
                                                             np.where(delta_w < w4[num_k],
                                                                      (hside*(v1+v2+v3)-delta_w)**2/v1/v2/v3/2,
                                                                      0)
                                                             )
                                                    )
                                           )

                DOScontribution = np.clip(DOScontribution, None, 8*hside**3/step_w)
                n_tmpt_ind = n_tmpt.astype(int)

                added = False
                if flag_delta_n_w == 0:
                    DOSarray[n_tmpt_ind[0]] += DOScontribution[0]
                    flag_delta_n_w = 1
                    added = True

                break_flag = np.logical_and.reduce([delta_w > w1[num_k], delta_w >= w2[num_k], delta_w >= w3[num_k],
                                                    delta_w >= w4[num_k]])
                i_break = np.argmax(break_flag)
                condition = np.logical_and.reduce([np.arange(N_w) < i_break, np.arange(N_w) != 0,
                                                   1 <= n_tmpt_ind, n_tmpt_ind <= N_w+1])
                DOSarray[n_tmpt_ind[condition]] += DOScontribution[condition]
                if added and condition[n_tmpt_ind[0]]:
                    DOSarray[n_tmpt_ind[0]] -= DOScontribution[0]

    # output DOS data into output.txt
    if num_kpoints[2] == 1:    # the structure is 2 dimension
        DOSarray = DOSarray*num_kpoints[1]

    if num_kpoints[0]*2 == num_kpoints[1]:     # the structure has time-reversal symmetry
        DOSarray = DOSarray*2

    if ceilDOS == 1:
        DOSarray[DOSarray > maxDOS] = maxDOS

    freq = w_min + step_w * np.arange(N_w+1)
    refsignal = DOSarray

    return freq[1:], refsignal[1:]


def normalize(freq32, DOS32, epsavg):
    # Specify inputs here
    sigma = 100     # bandwidth of filter
    wmax = 1.2      # Max frequency to truncate
    nw = 500        # no. of frequency points to interpolate

    fil32 = gaussfil(DOS32, sigma)

    old_w = np.array(freq32)*np.sqrt(epsavg)
    old_DOS = np.array(fil32)
    new_w = np.linspace(0, wmax, nw)
    new_DOS = np.interp(new_w, old_w, old_DOS)

    return new_w, new_DOS


def main(eps_mat, Nk=25):
    res = 32
    pol = 'tm'
    nbands = 10

    # create uniform k-grid for sampling. k points are sampled in the middle of microcell
    dk = 1 / Nk  # assume sample unit cell of unit length, Nkpoints = Nintervals
    kx = np.linspace(-0.5 + dk / 2, 0.5 - dk / 2, Nk)
    ky = np.linspace(-0.5 + dk / 2, 0.5 - dk / 2, Nk)
    gridx, gridy = np.meshgrid(kx, ky)
    gridx = np.ravel(gridx)
    gridy = np.ravel(gridy)
    kvecs = list()
    for i in range(len(gridx)):
        kpoint = np.array((gridx[i], gridy[i]))
        kvecs.append(kpoint)

    D = 2  # dimensionality of system

    rvecs = (np.array([1., 0.]), np.array([0., 1.]))  # square lattice
    kvec3 = converttovec3(kvecs, D)
    rvec3 = converttovec3(rvecs, D)

    efreq, gap, grpvel, mpbeps = runmpb(run_type=pol, dim=D, res=res, kvecs=kvec3, nbands=nbands, rvecs=rvec3, eps_mat=eps_mat)
    dataw = efreq.ravel()
    datav_original = np.reshape(grpvel, (np.shape(grpvel)[0]*np.shape(grpvel)[1], 3))

    freq, DOS = calc_DOS(dataw, datav_original, kin=Nk)
    freq_norm, DOS_norm = normalize(freq, DOS, np.mean(eps_mat.ravel()))

    return freq_norm, DOS_norm


if __name__ == "__main__":
    eps_mat = np.random.rand(32, 32) * 10.4 + 1
    main(eps_mat)

