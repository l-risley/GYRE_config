"""
Simple script to find the control variables and save to a NetCDF file.
"""

import numpy as np
from read_nemo_fields import *
from transforms.T_transform import *

def find_cv(eta_diff, u_diff, v_diff, lat):
    dx = 106000/12 # spatial step length
    dy = 106000/12
    alpha = 0

    deta_new = eta_diff[1:-2, 2:-1]
    du_new = u_diff[1:-2, 1:-1]
    dv_new = v_diff[1:-1, 2:-1]

    u_y = lat[1:-2, 1:-1]
    v_y = lat[1:-1, 2:-1]

    # find the control variables from the model increments
    d_eta, sf_u, vp_u, du_mean, dv_mean = T_transform(deta_new, du_new, dv_new, dx, dy, u_y, v_y, alpha)

    dsf_new, dvp_new = sf_u[1:-1, 1:-1], vp_u[1:-1, 1:-1] # DON'T WANT TO SAVE THE WEIRD BOUNDARIES

    return dsf_new, dvp_new


