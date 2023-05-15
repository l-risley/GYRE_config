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
    return u, v

def U_transform(d_eta, sf_u, vp_u, du_mean, dv_mean, dx, dy, u_lat, v_lat):
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
    du_b = geostrophic_balance_D(d_eta, 'u', u_lat, dy)
    dv_b = geostrophic_balance_D(d_eta, 'v', v_lat, dx)

    # Find the full velocities
    du = du_u + du_b + du_mean
    dv = dv_u + dv_b + dv_mean

    return d_eta, du, dv

def vel_from_helm_gyre(sf, vp, dy, dx, u_mask, v_mask):
    """
    Transform stream function and velocity potential to horizontal velocity vectors, based on Helmholtz theorem.
    u = - d sf/dy + d vp/dx
    v =  d sf/dx + d vp/dy
    Inputs:  - sf, streamfunction matrix (ny+1, nx+1)
             - vp, velocity potential matrix (ny+1, nx+1)
             - dx, dy, spatial grid length
             - u_mask, v_mask, masks applied to the velocities
    Outputs: - u, v, horizontal velocity matrices (ny, nx), (ny, nx)
    """
    # y-derivative of streamfunction
    dsf_dy = 1 / dy * (sf[1:, 1:] - sf[:-1, 1:]) #dzdy(sf, dy)[:, 1:]
    # x-derivative of velocity potential
    dvp_dx = 1 / dx * (vp[:-1, 1:] - vp[:-1, :-1]) #dzdx(vp, dx)[:-1, :]
    # find u
    u = - dsf_dy + dvp_dx

    # x-derivative of streamfunction
    v_sf = 1 / dx * (sf[1:, 1:] - sf[1:, :-1]) #dzdx(sf, dx)[1:, :]
    # y-derivative of velocity potential
    v_vp = 1 / dy * (vp[1:, :-1] - vp[:-1, :-1]) #dzdy(vp, dy)[:, :-1]
    # find v
    v = v_sf + v_vp
    #u[:, -2], v[-2, :] = 0,0
    return ma.array(u, mask=u_mask), ma.array(v, mask=v_mask)

def U_transform_gyre(d_eta, sf_u, vp_u, du_mean, dv_mean, dy, dx, u_lat, v_lat, u_mask, v_mask):
    """
    The U-transform from control variables (elevation, unbalanced streamfunction and unbalanced velocity
    potential) to model variables (elevation, zonal velocity and meridional velocity).
    dx = U dz
    x = (eta, u, v) and z = (eta, sf, vp)
    Inputs:  - d_eta, elevation increment
             - sf_u, unbalanced streamfunction increment matrix (ny+1, nx+1)
             - vp_u, unbalanced velocity potential increment matrix (ny+1, nx+1)
             - du_mean, mean zonal velocity
             - dv_mean, mean meridional velocity
             - dx, dy, spatial grid length
             - u_mask, v_mask, masks applied to the velocities
    Outputs: - du, dv, horizontal velocity matrices (ny, nx), (ny, nx)
    """
    # Find the unbalanced velocities from unbalanced sf and vp
    du_u, dv_u = vel_from_helm_gyre(sf_u, vp_u, dx, dy, u_mask, v_mask)

    # Use geostrophic balance to find the balanced velocities
    du_b = geostrophic_balance(d_eta, 'u', u_lat, dy, u_mask)
    dv_b = geostrophic_balance(d_eta, 'v', v_lat, dx, v_mask)

    # Find the full velocities
    du = du_u + du_b + du_mean
    dv = dv_u + dv_b + dv_mean

    return d_eta, du, dv