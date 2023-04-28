"""
Adjoint tests for the adjoint operator used in:
 - Tikhonov's regularisation for the gyre configuration.
@author: Laura Risley 2023
"""

import numpy as np

from transforms.U_transform import *
from transforms.T_transform import *
from transforms.Tik_regularisation import *
from read_nemo_fields import *
from gyre_setup import *

def adjoint_gyre_test():
    """
    Test the adjoint of Tikhonov's regularisation.
    """

    # what gyre time to look at (this does not matter)
    time_index = 700

    # netcdf file locations
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(u_input_file)

    # read eta, u and v from files at two times (1 day apart)

    u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
    u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)[0]
    u_diff = u_1 - u_0

    v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
    v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
    v_diff = v_1 - v_0
    print(f'Found perturbations of the velocities.')

    contour(lon, lat, u_0, 'Full', 'u')
    contour(lon, lat, v_0, 'Full', 'v')

    dy, dx = param['dy'], param['dx']

    ny, nx = np.shape(u_0)
    # alpha, tikhonov regularisation parameter
    alpha = 0  # 1e-10

    # choice of convergence
    conv = None  # 'convergence'
    sf, vp = tik_reg_gyre(alpha, u_0[2:-2, 2:-2], v_0[2:-2, 2:-2], dy, dx, ny - 4, nx - 4, conv)
    print(np.shape(sf))
    print(f'Derived the perturbations of streamfunction and velocity potential.')
    # Combine sf and vp into one array
    x = np.append(sf.flatten(), vp.flatten())

    # Find Ax
    Ax = A_operator_gyre(x, dy, dx, ny, nx)
    print(f'Found Ax.')

    # Find A^T Ax
    AtAx = A_adjoint_gyre(Ax, dy, dx, ny, nx)
    print(f'Found A^T Ax.')

    # Find the dot products
    prod_1 = np.dot(Ax, Ax)
    prod_2 = np.dot(x, AtAx)
    print(f'Found the dot products.')

    # Find the differences in dot products
    prod_diff = prod_1-prod_2
    print(prod_diff)

def adjoint_old_gyre_test():
    """
    Test the adjoint of Tikhonov's regularisation.
    """
    time_index = 300

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

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

    ## Find streamfunction and velocity potential perturbations
    # set a value of alpha
    alpha = 0 #1e-13

    # size of sf and vp will be larger than eta
    #ny, nx = np.shape(eta_0)
    #ny, nx = ny - 5, nx - 5
    # choice of convergence
    conv = None  # 'convergence'
    #input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/output_diff_sf_vp_combined.nc"
    #sf = read_file(input_file, "sfd_arr", time_index=time_index)
    #vp = read_file(input_file, "vpd_arr", time_index=time_index)
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

    u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
    u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)[0]
    u_diff = u_1 - u_0

    v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
    v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
    v_diff = v_1 - v_0
    ny, nx = np.shape(u_0[1:-2, 2:-1])
    sf, vp = tik_reg(alpha, u_diff[1:-2, 1:-1], v_diff[1:-1, 2:-1], dy, dx, ny, nx, conv)
    print(np.shape(sf))
    #lon, lat, time = read_file_info(input_file)


    print(f'Derived the perturbations of streamfunction and velocity potential.')
    # Combine sf and vp into one array
    x = np.append(sf.flatten(), vp.flatten())

    # Find Ax
    Ax = A_operator(x, dy, dx, ny, nx)
    print(f'Found Ax.')

    # Find A^T Ax
    AtAx = A_adjoint(Ax, dy, dx, ny, nx)
    print(f'Found A^T Ax.')

    # Find the dot products
    prod_1 = np.dot(Ax, Ax)
    prod_2 = np.dot(x, AtAx)
    print(f'Found the dot products.')

    # Find the differences in dot products
    prod_diff = prod_1-prod_2
    print(prod_diff)

if __name__ == '__main__':
    adjoint_gyre_test()
    #adjoint_old_gyre_test()