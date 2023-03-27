"""
Parameters for the gyre setup and boundary conditions.
"""

import numpy as np

#TODO: CHECK lx, ly, f0, beta

param = { "L_x" : 1e6,      # computational domain in x
          "L_y" : 1e6,      # computational domain in y
          "f_0" : 1e-4,     # coriolis parameter
          "beta" : 1e-11,   # beta-plane
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