"""

"""
import numpy as np

from read_nemo_fields import *
from gyre_setup import *
from transforms.T_transform import *
from transforms.U_transform import *
from transforms.Tik_regularisation import *

def vel_from_helm_test():

    time_index = 700

    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

    # read eta, u and v from files at two times (1 day apart)
    # calculate the increment
    eta = read_file(eta_input_file, "sossheig", time_index=time_index)
    u = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
    v = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
    eta_mask, u_mask, v_mask = eta.mask, u.mask, v.mask

    input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/output_sf_vp_combined.nc"
    sf = read_file(input_file, "sf_arr", time_index=time_index)
    vp = read_file(input_file, "vp_arr", time_index=time_index)

    cv_lon, cv_lat, time = read_file_info(input_file)
    lon, lat, time = read_file_info(eta_input_file)

    print(f' The shape of sf is {np.shape(sf)}.')
    print(f' The shape of vp is {np.shape(vp)}.')
    print(f' The shape of u is {np.shape(u)}.')
    print(f' The shape of v is {np.shape(v)}.')
    print(f'lat ={lat}, lon = {lon}')

    contour(cv_lon, cv_lat, sf, 'Full', 'SF')
    contour(cv_lon, cv_lat, vp, 'Full', 'VP')

    dx, dy = param['dx'], param['dy']

    #Find u and v from Helmholtx theorem
    u, v = vel_from_helm_gyre(sf, vp, dy, dx, u_mask, v_mask)
    print(f' The shape of u is {np.shape(u)}.')
    print(f' The shape of v is {np.shape(v)}.')

    #contour(lon[1:, 1:], lat[1:, 1:], u, 'Full', 'u')
    #contour(lon[1:, 1:], lat[1:, 1:], v, 'Full', 'v')

    contour(lon, lat, u, 'Full', 'u')
    contour(lon, lat, v, 'Full', 'v')

def tikhonov_test():
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
    """
    u_mask, v_mask = u_0.mask, v_0.mask
    # alpha, tikhonov regularisation parameter
    alpha = 0  # 1e-10

    # choice of convergence
    conv = None  # 'convergence'
    sf, vp = tik_reg_gyre(alpha, u_0, v_0, dy, dx, ny, nx, u_mask, v_mask, conv)
    print(f' The shape of sf is {np.shape(sf)}.')
    print(f' The shape of vp is {np.shape(vp)}.')

    print(f' SF = {sf}.')
    print(f' VP = {vp}.')

    sf_new = sf[1:, 1:]
    vp_new = vp[:-1, :-1]

    print(f' The shape of sf new is {np.shape(sf_new)}.')
    print(f' The shape of vp new is {np.shape(vp_new)}.')

    contour(lon, lat, sf_new, 'Full', 'SF')
    contour(lon, lat, vp_new, 'Full', 'VP')
    """
    u_mask, v_mask = u_0[2:-2, 2:-2].mask, v_0[2:-2, 2:-2].mask

    ny, nx = np.shape(u_0[2:-2, 2:-2])
    # alpha, tikhonov regularisation parameter
    alpha = 0 #1e-10

    # choice of convergence
    conv = None  # 'convergence'
    sf, vp = tik_reg_gyre(alpha, u_0[2:-2, 2:-2], v_0[2:-2, 2:-2], dy, dx, ny, nx, u_mask, v_mask, conv)
    print(f' The shape of sf is {np.shape(sf)}.')
    print(f' The shape of vp is {np.shape(vp)}.')

    print(f' SF = {sf}.')
    print(f' VP = {vp}.')

    sf_new = sf[1:, 1:]
    vp_new = vp[:-1, :-1]

    print(f' The shape of sf new is {np.shape(sf_new)}.')
    print(f' The shape of vp new is {np.shape(vp_new)}.')

    contour(lon[2:-2, 2:-2], lat[2:-2, 2:-2], sf_new, 'Full', 'SF')
    contour(lon[2:-2, 2:-2], lat[2:-2, 2:-2], vp_new, 'Full', 'VP')

if __name__ == '__main__':
    #vel_from_helm_test()
    tikhonov_test()
