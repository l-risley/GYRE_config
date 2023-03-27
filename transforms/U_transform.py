"""
U-transform from control variables to model variables,
dx = U dz.
@author: Laura Risley 2023
"""

from general_functions import *
from transforms.balance import *
from gyre_setup import *

def vel_from_helm(sf, vp, dx, dy):
    """
    Transform stream function and velocity potential to horizontal velocity vectors, based on Helmholtz theorem.
    u = - d sf/dy + d vp/dx
    v =  d sf/dx + d vp/dy
    Inputs:  - sf, streamfunction matrix (ny+2, nx+2)
             - vp, velocity potential matrix (ny+2, nx+2)
             - dx, dy, spatial grid length
    Outputs: - u, v, horizontal velocity matrices (ny, nx+1), (ny+1, nx)
    """
    # y-derivative of streamfunction
    u_sf = 0.25 * 1 / dy * (-sf[:-2, :-1] - sf[:-2, 1:] + sf[2:, :-1] + sf[2:, 1:])
    # x-derivative of velocity potential
    u_vp = 1 / dx * (vp[1:-1, 1:] - vp[1:-1, :-1])
    # find u
    u = - u_sf + u_vp

    # x-derivative of streamfunction
    v_sf = 0.25 * 1 / dx * (-sf[:-1, :-2] - sf[1:, :-2] + sf[:-1, 2:] + sf[1:, 2:])
    # y-derivative of velocity potential
    v_vp = 1 / dy * (vp[1:, 1:-1] - vp[:-1, 1:-1])
    # find v
    v = v_sf + v_vp
    u, v = zonal_boundary(u), meridional_boundary(v)
    return u, v

def U_transform(d_eta, sf_u, vp_u, du_mean, dv_mean, dx, dy, lat):
    """
    The U-transform from control variables (elevation, unbalanced streamfunction and unbalanced velocity
    potential) to model variables (elevation, zonal velocity and meridional velocity).
    dx = U dz
    x = (eta, u, v) and z = (eta, sf, vp)
    Inputs:  - d_eta, elevation increment
             - sf_u, unbalanced streamfunction increment matrix (ny+2, nx+2)
             - vp_u, unbalanced velocity potential increment matrix (ny+2, nx+2)
             - du_mean, mean zonal velocity
             - dv_mean, mean meridional velocity
             - dx, dy, spatial grid length
    Outputs: - du, dv, horizontal velocity matrices (ny, nx+1), (ny+1, nx)
    """
    # Find the unbalanced velocities from unbalanced sf and vp
    du_u, dv_u = vel_from_helm(sf_u, vp_u, dx, dy)

    # Use geostrophic balance to find the balanced velocities
    du_b = geostrophic_balance_D(d_eta, 'u', lat, dy)
    dv_b = geostrophic_balance_D(d_eta, 'v', lat, dx)

    # Find the full velocities
    du = du_u + du_b + du_mean
    dv = dv_u + dv_b + dv_mean

    # apply boundary condition
    du, dv = zonal_boundary(du), meridional_boundary(dv)

    return d_eta, du, dv
