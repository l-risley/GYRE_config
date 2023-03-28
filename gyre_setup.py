"""
Parameters for the gyre setup and boundary conditions.
Most values have come from NEMO set up:
/projects/jodap/lrisley/GYRE/fcm_make_ocean/preprocess-ocean/src/nemo/src/OCE/USR/usrdef_hgr.F90
/projects/jodap/lrisley/GYRE/fcm_make_ocean/preprocess-ocean/src/nemo/src/OCE/DOM/phycst.F90
"""

import numpy as np

#TODO: CHECK lx, ly, f0, beta

param = { "L_x" : 212e4,      # computational domain in x
          "L_y" : 318e4,      # computational domain in y
          "rday": 86400,     # seconds in a day
          "ra" : 6371229,     # earth radius
          "rad" : np.pi/180,  # conversion from degree to radian
          "phi0" : 15, # lattitude of most southern grid point
          "phi1" : 29,
          "g" : 9.80665,    # gravitational acceleration
          "dx" : 106000/12, # spatial step length
          "dy" : 106000/12
}

def zonal_boundary(u : np.ndarray):
    # Dirichlet Boundary Condition
    # Set u=0 at eastern and western boundaries
    u[:, [0, -1]] = 0
    return u


def meridional_boundary(v : np.ndarray):
    # Dirichlet Boundary Condition
    # Set v=0 at northern and southern boundaries
    v[[0, -1], :] = 0
    return v

def omega_calc():
    """
    Calculate the earth rotation parameter (s^-1).
    """
    rday = param["rday"]
    # seconds in sideral year
    rsiyea = 365.25 * rday * 2 * np.pi/6.283076
    #seconds in sideral day
    rsiday = rday / (1 + rday/rsiyea)
    return 2 * np.pi / rsiday

def f0_calc():
    # coriolis parameter
    # see phys_cst
    rad, phi0 = param['rad'], param['phi0']
    omega = omega_calc()
    return 2 * omega * np.sin(rad*phi0)

def beta_calc():
    # beta plane
    # see phys_cst
    rad, phi1, ra = param['rad'], param['phi1'], param['ra']
    omega = omega_calc()
    return 2 * omega * np.cos(rad * phi1)/ra
