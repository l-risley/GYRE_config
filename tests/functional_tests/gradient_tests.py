"""

"""
import numpy as np
import numpy.linalg as la

from transforms.U_transform import *
from transforms.T_transform import *
from transforms.Tik_regularisation import *
from read_nemo_fields import *
from gyre_setup import *
from general_functions import *


# costfunction
def tik_fun(x, b, dy, dx, ny, nx, alpha):
    # J_a = 0.5* (b-Ax)^T(b-Ax) + a*0.5*x^Tx
    J_x = b - A_operator_gyre(x, dy, dx, ny, nx)
    J = ma.dot(J_x, J_x)
    J_reg = alpha * ma.dot(x, x)
    return 0.5 * (J + J_reg)


# gradient
def tik_grad(x, b, dy, dx, ny, nx, alpha):
    # grad_J = -A^T(b-Ax) + a*x
    # b-Ax
    J_x = b - A_operator_gyre(x, dy, dx, ny, nx)
    # adjoint applied to b-Ax
    adj = A_adjoint_gyre(J_x, dy, dx, ny, nx)
    return -adj + alpha * x


def gradient_gyre_test():
    """
    Test the gradient of the Tikhonov's regularisation cost function using the following gradient test:
    J(x + ah) = J(x) + ah^T gJ(x) + O(a^2)
    Phi(a) = (J(x+ah) - J(x)) / ah^T gJ(x) = 1 + O(a)
    plot Phi(a) as a tends to zero
    h should be of unit length, h = gJ(x)/norm(g(J(x)), take the 2-norm

    Output: - plot of Phi(a) and 1-Phi(a) as a tends to zero
    """

    time_index = 289

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

    # lon and lat for each grid
    eta_lon, eta_lat, time = read_file_info(eta_input_file)

    # read eta, u and v from files at two times (1 day apart)
    # calculate the increment
    eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)
    eta_1 = read_file(eta_input_file, "sossheig", time_index=time_index + 1)
    eta_diff = eta_1 - eta_0

    u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
    u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)[0]
    u_diff = u_1 - u_0

    v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
    v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
    v_diff = v_1 - v_0
    print(f'Found perturbations of the velocities.')

    dy, dx = param['dy'], param['dx']

    b = np.append(u_diff.flatten(), v_diff.flatten())
    print(f'Found perturbations of the velocities.')

    ## Find streamfunction and velocity potential perturbations
    # set a value of alpha, tikhonov regularisation parameter
    alpha = 0  # 1e-13

    # size of sf and vp will be larger than eta
    ny, nx = np.shape(eta_0)

    # choice of convergence
    conv = None  # 'convergence'
    sf, vp = tik_reg_gyre(alpha, u_diff, v_diff, dy, dx, ny, nx, conv)

    sf_new = sf[1:, 1:]
    vp_new = vp[:-1, :1]
    contour(eta_lon, eta_lat, sf_new, 'Increment', 'SF')
    contour(eta_lon, eta_lat, vp_new, 'Increment', 'VP')

    print(f'Derived the perturbations of streamfunction and velocity potential.')
    # Combine sf and vp into one array
    x = np.append(sf.flatten(), vp.flatten())

    # a will be 10^-p
    p = np.array(range(-11, -1))

    # calculate the cost function
    J = float(tik_fun(x, b, dy, dx, ny, nx, alpha))

    # calculate the gradient of the cost function
    gJ = tik_grad(x, b, dy, dx, ny, nx, alpha)

    # calculate h = gJ/|gJ|
    gJ_norm = la.norm(gJ, 2)
    h = gJ / gJ_norm

    # loop over the different values of p (alpha) and find Phi
    Phi = np.empty_like(p, dtype=float)
    a_array = 10.0 ** p

    j = 0
    for i in p:
        a = 10.0 ** i
        x_a_h = x - a * h
        # J(x+ah)
        J_a_h = float(tik_fun(x_a_h, b, dy, dx, ny, nx, alpha))
        print(f' Jah - J = {J_a_h - J}')
        hgJT = float(np.matmul(h, np.mat(gJ).T))

        # Phi = J(x+ah)- J(x)/ ahTgJ
        Phi[j] = float(abs(J_a_h - J) / (a * hgJT))

        print(f' ahTgJ = {a * hgJT}')
        print(float((J_a_h - J) / (a * hgJT)))
        print(f'Phi is equal to {Phi[j]}')

        j += 1

    # plot values of Phi
    plot_gradient_test(a_array, Phi)
    plt.show()

    plot_minus_gradient_test(a_array, 1 - Phi)
    plt.show()


if __name__ == '__main__':
    gradient_gyre_test()
