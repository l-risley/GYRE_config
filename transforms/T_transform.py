"""
T-transform from model variables to control variables
dz = T dx.p
@author: Laura Risley 2023
"""

from transforms.balance import *
from transforms.U_transform import *
from transforms.Tik_regularisation import *

def T_transform(d_eta, du, dv, dx, dy, lat_u, lat_v, alpha):
    """
    T-transform from model variables (elevation, zonal velocity and meridional velocity) to control variables
    (elevation, unbalanced streamfunction and unbalanced velocity
    potential).
    dz = T dx
    x = (eta, u, v) and z = (eta, sf, vp)
    Inputs:  - d_eta, elevation increment
             - du, dv, horizontal velocity matrices (ny, nx+1), (ny+1, nx)
             - vp_u, unbalanced velocity potential increment matrix (ny+2, nx+2)
             - dx, dy, spatial grid length
             - alpha, Tikhonov's regularisation parameter
    Outputs: - d_eta, elevation increment
             - sf_u, unbalanced streamfunction increment matrix (ny+2, nx+2)
             - vp_u, unbalanced velocity potential increment matrix (ny+2, nx+2)
             - du_mean, mean zonal velocity
             - dv_mean, mean meridional velocity
    """
    # number of eta points on the grid
    ny, nx = np.shape(d_eta)
    # Use geostrophic balance to find the balanced velocities
    du_b = geostrophic_balance_D(d_eta, 'u', lat_u, dy)
    dv_b = geostrophic_balance_D(d_eta, 'v', lat_v, dx)

    # Find the unbalanced components of the velocities
    du_u = du - du_b
    dv_u = dv - dv_b

    ### Find the unbalanced components of stream function and velocity potential
    ## Use Tikhonov's regularisation
    # no convergence plots necessary within the T-transform
    conv = None
    sf_u, vp_u = tik_reg(alpha, du_u, dv_u, dx, dy, ny, nx, conv)

    # Store the mean values of u and v
    du_mean = du.mean()
    dv_mean = dv.mean()

    return d_eta, sf_u, vp_u, du_mean, dv_mean
