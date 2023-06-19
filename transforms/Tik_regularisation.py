"""
Functions to perform Tikhonov's regularisation. This is used to find streamfunction and velocity potential from the
horizontal velocity vectors, Li et al 2005.
@author: Laura Risley 2023
"""

import numpy as np
import numpy.ma as ma
from scipy.optimize import minimize, fmin_cg
from scipy.sparse.linalg import LinearOperator, cg
from numpy.linalg import norm
from transforms.U_transform import *


def min_method(fn, grad, x0: np.ndarray, conv):
    """
    Minimisation function for the cost function in Tikhonov's regularisation.
    Output minimisation value and list of functions at each interation.
    Inputs: fn, function to be minimised
           grad, gradient of the function
           x0, initial input
           conv, None - output only minimised value
                 Convergence - output minimised value, values of fn and grad-norm at each iteration
    """
    print('Began minimisation')
    gtol = 1e-05 * (norm(grad(x0), np.inf).item())
    print(f'Minimisation criteria: gradient norm < {gtol}.')
    method = 'CG'
    if conv is None:
        return minimize(fn, x0, method=method, jac=grad,
                        options={'disp': True, 'gtol': gtol})

    elif conv == 'convergence':
        all_fn = [fn(x0).item()]
        all_grad = [norm(grad(x0), np.inf).item()]

        def store(x):  # callback function
            all_fn.append(fn(x).item())
            all_grad.append(norm(grad(x), np.inf).item())

        ans = minimize(fn, x0, method=method, jac=grad, callback=store,
                       options={'disp': True, 'gtol': gtol})  # , 'maxiter': 200
        print(all_fn)
        print(all_grad)
        return ans, all_fn, all_grad

def split_x_gyre(x, ny_cv, nx_cv):
    """
    Split vector x into two matrices for stream function and velocity potential
    Inputs:  x, vector containing sf (streamfunction) and vp (velocity potential) (2*(ny+2)*(nx+2))
             ny, nx, number of eta points on the grid (ny, nx)
    Outputs: sf, streamfunction (ny+2, nx+2)
             vp, velocity potential (ny+2, nx+2)
    """
    # split x into two equal arrays for sf and vp
    sf, vp = np.split(x, 2)
    # reshape sf and vp into matrices
    sf, vp = np.reshape(sf, (ny_cv, nx_cv)), np.reshape(vp, (ny_cv, nx_cv))
    return sf, vp


def split_b_gyre(b, ny, nx):
    """
    Split vector b into two matrices for zonal and meridional velocity
    Inputs:  b, vector containing the zonal and meridional velocity ((ny)*(nx+1) + (ny+1)*(nx))
             ny, nx, number of eta points on the grid (ny, nx)
    Outputs: u, zonal velocity (ny, nx+1)
             v, meridional velocity  (ny+1, nx)
    """
    # split b into two arrays for u and v
    u, v = np.split(b, 2)
    # reshape sf and vp into matrices
    u, v = np.reshape(u, (ny, nx)), np.reshape(v, (ny, nx))
    return u, v


def A_operator_gyre(x, dy, dx, ny, nx, u_mask, v_mask):
    """
    The linear operator Ax for the transform from stream function and velocity potential (x) to horizontal velocity vectors (b).
    Inputs:  - x, vector containing sf (streamfunction) and vp (velocity potential) (2* ny_cv*nx_cv = 2* ny+2 * nx+2)
             - dy, dx, spatial grid length
             - ny_cv, nx_cv, number of sf and vp points on the grid (ny_cv, nx_cv)
             - u_mask, v_mask, velocity masks
    Outputs: - b, vector containing the horizontal velocities, u and v (ny*(nx+1) + nx*(ny+1))
    """
    # split x into two equal arrays for sf and vp
    sf, vp = split_x_gyre(x, ny+1, nx+1)

    # apply the u-transform
    u, v = vel_from_helm_gyre(sf, vp, dy, dx, u_mask, v_mask)
    # flatten to a vector
    u_vec = u.flatten()
    v_vec = v.flatten()

    # created one vector containing both u and v
    b = np.append(u_vec, v_vec)
    return b


def A_adjoint_gyre(b, dy, dx, ny, nx):
    """
    The adjoint of linear operator Ax.
    Inputs:  - b, vector containing the horizontal velocities, u and v (2*ny*nx)
             - dy, dx, spatial grid length
             - ny, nx, number of eta points on the grid
    Outputs: - x, vector containing sf (streamfunction) and vp (velocity potential) (2 *(ny+1)*(nx+1))
    """
    # split b into two equal arrays for u and v
    u, v = split_b_gyre(b, ny, nx)

    # sizes of sf and vp
    ny_cv, nx_cv = ny + 1, nx + 1

    # initialise sf and vp
    sf, vp = np.zeros((ny_cv, nx_cv)), np.zeros((ny_cv, nx_cv))

    v_dy = 1 / dy * v
    v_dx = 1 / dx * v

    ## adjoint routine begins

    # adjoint of v =  d sf/dx + d vp/dy
    vp[:-1, :-1] += - v_dy
    vp[1:, :-1] += v_dy
    sf[1:, :-1] += - v_dx
    sf[1:, 1:] += v_dx

    v = 0

    u_dy = 1 / dy * u
    u_dx = 1 / dx * u

    # adjoint of u =  -d sf/dy + d vp/dx
    vp[:-1, :-1] += - u_dx
    vp[:-1, 1:] += u_dx
    sf[:-1, 1:] += u_dy
    sf[1:, 1:] += - u_dy

    u = 0

    # flatten to a vector
    sf_vec = sf.flatten()
    vp_vec = vp.flatten()

    # created one vector containing both sf and vp
    x = np.append(sf_vec, vp_vec)
    return x

def tik_reg_gyre(alpha, u, v, dy, dx, ny, nx, u_mask, v_mask, conv=None):
    """
    Tikhonov's regularisation to find the horizontal velocity vectors from streamfunction and velocity potential
    Inputs: alpha, regularisation parameter
            u, v, horizontal velocity vectors
            dy, dx, spatial grid length
            ny, nx, number of eta points on the grid (ny, nx)
            u_mask, v_mask, velocity masks
            conv, None - output only minimised value
                  Convergence - output minimised value, values of fn and grad-norm at each iteration
    Outputs: sf, streamfunction (ny+1, nx+1)
             vp, velocity potential (ny+1, nx+1)
    """

    # put u and v into a vector
    u.set_fill_value(0)
    v.set_fill_value(0)

    b = ma.append(u.flatten(), v.flatten())

    # input x is a vector
    # costfunction
    def tik_fun(x):
        # J_a = 0.5* (b-Ax)^T(b-Ax) + a*0.5*x^Tx
        J_x = b - A_operator_gyre(x, dy, dx, ny, nx, u_mask, v_mask)
        J = ma.dot(J_x, J_x)
        J_reg = alpha * ma.dot(x, x)
        return 0.5 * (J + J_reg)

    # gradient
    def tik_grad(x):
        # grad_J = -A^T(b-Ax) + a*x
        # b-Ax
        J_x = b - A_operator_gyre(x, dy, dx, ny, nx, u_mask, v_mask)
        # adjoint applied to b-Ax
        adj = A_adjoint_gyre(J_x, dy, dx, ny, nx)
        return -adj + alpha * x

    # sizes of sf and vp
    ny_cv, nx_cv = ny + 1, nx + 1

    # initial guess for minimisation
    x_0 = np.zeros(2*ny_cv*nx_cv)
    #x_0 = 300*np.ones(2*ny_cv*nx_cv)
    cf = tik_fun(x_0)
    print(f' Value of the initial cost function: {cf} for x = {x_0}.')
    gcf = tik_grad(x_0)
    print(f' Value of the initial gradient: {gcf} for x = {x_0}.')
    print(f'Value of the initial gradient norm: {norm(gcf)} for x = {x_0}.')

    result = min_method(tik_fun, tik_grad, x_0, conv)  # use pre-defined fn to optimise

    if conv is None:
        x_arr = np.asarray(result.x)
        sf, vp = split_x_gyre(x_arr, ny_cv, nx_cv)
        return sf, vp
    elif conv == 'convergence':
        ans, cf_list, grad_list = result
        x_arr = np.asarray(ans.x)
        sf, vp = split_x_gyre(x_arr, ny_cv, nx_cv)
        cf_array = np.asarray(cf_list)
        grad_array = np.asarray(grad_list)
        return sf, vp, cf_array, grad_array

#######################################################################################################################

def split_x(x, ny_cv, nx_cv):
    """
    Split vector x into two matrices for stream function and velocity potential
    Inputs:  x, vector containing sf (streamfunction) and vp (velocity potential) (2*(ny+2)*(nx+2))
             ny, nx, number of eta points on the grid (ny, nx)
    Outputs: sf, streamfunction (ny+2, nx+2)
             vp, velocity potential (ny+2, nx+2)
    """
    # split x into two equal arrays for sf and vp
    sf, vp = np.split(x, 2)
    # reshape sf and vp into matrices
    sf, vp = np.reshape(sf, (ny_cv, nx_cv)), np.reshape(vp, (ny_cv, nx_cv))
    return sf, vp


def split_b(b, ny, nx):
    """
    Split vector b into two matrices for zonal and meridional velocity
    Inputs:  b, vector containing the zonal and meridional velocity ((ny)*(nx+1) + (ny+1)*(nx))
             ny, nx, number of eta points on the grid (ny, nx)
    Outputs: u, zonal velocity (ny, nx+1)
             v, meridional velocity  (ny+1, nx)
    """
    # split b into two arrays for u and v
    u, v = np.split(b, [ny * (nx + 1)])
    # reshape sf and vp into matrices
    u, v = np.reshape(u, (ny, nx + 1)), np.reshape(v, (ny + 1, nx))
    return u, v


def A_operator(x, dy, dx, ny, nx):
    """
    The linear operator Ax for the transform from stream function and velocity potential (x) to horizontal velocity vectors (b).
    Inputs:  - x, vector containing sf (streamfunction) and vp (velocity potential) (2* ny_cv*nx_cv = 2* ny+2 * nx+2)
             - dy, dx, spatial grid length
             - ny_cv, nx_cv, number of sf and vp points on the grid (ny_cv, nx_cv)
    Outputs: - b, vector containing the horizontal velocities, u and v (ny*(nx+1) + nx*(ny+1))
    """
    # sizes of sf and vp
    ny_cv, nx_cv = ny + 2, nx + 2

    # split x into two equal arrays for sf and vp
    sf, vp = split_x(x, ny_cv, nx_cv)

    # apply the u-transform
    u, v = vel_from_helm(sf, vp, dx, dy)
    # flatten to a vector
    u_vec = u.flatten()
    v_vec = v.flatten()

    # created one vector containing both u and v
    b = np.append(u_vec, v_vec)
    return b


def A_adjoint(b, dy, dx, ny, nx):
    """
    The adjoint of linear operator Ax.
    Inputs:  - b, vector containing the horizontal velocities, u and v (ny*(nx+1) + nx*(ny+1))x
             - dy, dx, spatial grid length
             - ny, nx, number of eta points on the grid (ny, nx)
    Outputs: - x, vector containing sf (streamfunction) and vp (velocity potential) ((ny+2)*(nx+2) + (ny+2)*(nx+2))
    """
    # split b into two equal arrays for u and v
    u, v = split_b(b, ny, nx)

    # sizes of sf and vp
    ny_cv, nx_cv = ny + 2, nx + 2

    # initialise sf and vp
    sf, vp = np.zeros((ny_cv, nx_cv)), np.zeros((ny_cv, nx_cv))

    v_dx = 0.25 * 1 / dx * v
    v_dy = 1 / dy * v

    ## adjoint routine begins

    # adjoint of v =  d sf/dx + d vp/dy
    sf[:-1, :-2] += - v_dx
    sf[1:, :-2] += - v_dx
    sf[:-1, 2:] += v_dx
    sf[1:, 2:] += v_dx

    vp[1:, 1:-1] += v_dy
    vp[:-1, 1:-1] += -v_dy

    v = 0

    u_dy = 0.25 * 1 / dy * u
    u_dx = 1 / dx * u

    # adjoint of u =  -d sf/dy + d vp/dx

    sf[:-2, :-1] += u_dy
    sf[:-2, 1:] += u_dy
    sf[2:, :-1] += - u_dy
    sf[2:, 1:] += - u_dy

    vp[1:-1, 1:] += u_dx
    vp[1:-1, :-1] += -u_dx

    u = 0

    # flatten to a vector
    sf_vec = sf.flatten()
    vp_vec = vp.flatten()

    # created one vector containing both sf and vp
    x = np.append(sf_vec, vp_vec)
    return x

def tik_reg(alpha, u, v, dy, dx, ny, nx, conv=None):
    """
    Tikhonov's regularisation to find the horizontal velocity vectors from streamfunction and velocity potential
    Inputs: alpha, regularisation parameter
            u, v, horizontal velocity vectors
            dy, dx, spatial grid length
            ny, nx, number of eta points on the grid (ny, nx)
            conv, None - output only minimised value
                  Convergence - output minimised value, values of fn and grad-norm at each iteration
    Outputs: sf, streamfunction (ny+1, nx+1)
             vp, velocity potential (ny+2, nx+2)
    """

    # put u and v into a vector
    u.set_fill_value(0)
    v.set_fill_value(0)
    b = ma.append(u.flatten(), v.flatten())
    # input x is a vector
    # costfunction
    def tik_fun(x):
        # J_a = 0.5* (b-Ax)^T(b-Ax) + a*0.5*x^Tx
        J_x = b - A_operator(x, dy, dx, ny, nx)
        J = ma.dot(J_x, J_x)
        J_reg = alpha * ma.dot(x, x)
        return 0.5 * (J + J_reg)

    # gradient
    def tik_grad(x):
        # grad_J = -A^T(b-Ax) + a*x
        # b-Ax
        J_x = b - A_operator(x, dy, dx, ny, nx)
        # adjoint applied to b-Ax
        adj = A_adjoint(J_x, dy, dx, ny, nx)
        return -adj + alpha * x

    # sizes of sf and vp
    ny_cv, nx_cv = ny + 2, nx + 2

    # initial guess for minimisation
    #x_0 = np.zeros(2*ny_cv*nx_cv)
    x_0 = 100*np.ones(2*ny_cv*nx_cv)
    cf = tik_fun(x_0)
    print(cf)
    gcf = tik_grad(x_0)
    print(gcf)

    result = min_method(tik_fun, tik_grad, x_0, conv)  # use pre-defined fn to optimise

    if conv is None:
        x_arr = np.asarray(result.x)
        sf, vp = split_x(x_arr, ny_cv, nx_cv)
        return sf, vp
    elif conv == 'convergence':
        ans, cf_list, grad_list = result
        x_arr = np.asarray(ans.x)
        sf, vp = split_x(x_arr, ny_cv, nx_cv)
        cf_array = np.asarray(cf_list)
        grad_array = np.asarray(grad_list)
        return sf, vp, cf_array, grad_array

#######################################################################################################################
