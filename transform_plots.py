"""
Plot the model variables, control variables, full sf and vp fields as well.
@author: Laura Risley 223
"""
import numpy as np

from transforms.U_transform import *
from transforms.T_transform import *
from read_nemo_fields import *
from general_functions import *

def transform_plots(alpha, time_index, conv):
    """
    Plots from the gyre configuration for the full model fields, daily increments, stream function and velocity potential.
    Inputs: alpha, Tikhonov's regularisation parameter
            time_index, time of plots
            conv, whether to plot the convergence plots
    """
    dy, dx = param['dy'], param['dx']

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

    # lon and lat for each grid
    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    # number of times in the files
    num_times = np.size(time)

    print(f'Length of elevation lattitude : {np.shape(eta_lat)} and longitude : {np.shape(eta_lon)}.')
    print(f'Length of u lattitude : {np.shape(u_lat)} and longitude : {np.shape(u_lon)}.')
    print(f'Length of v lattitude : {np.shape(v_lat)} and longitude : {np.shape(v_lon)}.')

    # size of eta
    ny, nx = np.shape(eta_lat)[0]-1, np.shape(eta_lon)[1]-1
    print(f'ny = {ny}, nx = {nx}')

    # plot full fields at both times

    contour(eta_lon, eta_lat, eta_0, f'at time {time[time_index]}', 'Elevation')
    contour(u_lon, u_lat, u_0, f'at time {time[time_index]}', 'Zonal Velocity')
    contour(v_lon, v_lat, v_0, f'at time {time[time_index]}', 'Meridional Velocity')

    contour(eta_lon, eta_lat, eta_1, f'at time {time[time_index+1]}', 'Elevation')
    contour(u_lon, u_lat, u_1, f'at time {time[time_index+1]}', 'Zonal Velocity')
    contour(v_lon, v_lat, v_1, f'at time {time[time_index+1]}', 'Meridional Velocity')

    contour(eta_lon, eta_lat, eta_diff, f'increment', 'Elevation')
    contour(u_lon, u_lat, u_diff, f'increment', 'Zonal Velocity')
    contour(v_lon, v_lat, v_diff, f'increment', 'Meridional Velocity')

    print(f'Shape of elevation : {np.shape(eta_0)}')
    print(f' Shape of u : {np.shape(u_0)}')
    print(f' Shape of v : {np.shape(v_0)}')

    #######################################################
    ##### FIND SF AND VP FOR WITHIN THE DOMAIN, NO BOUNDARY
    eta_new = eta_0[1:-2, 2:-1]
    u_new = u_0[1:-2, 1:-1]
    v_new = v_0[1:-1, 2:-1]
    ny, nx = np.shape(eta_new)
    print(f'Shape of new elevation : {np.shape(eta_new)}')
    print(f' Shape of new u : {np.shape(u_new)}')
    print(f' Shape of new v : {np.shape(v_new)}')

    eta_x, eta_y = eta_lon[1:-2, 2:-1], eta_lat[1:-2, 2:-1]
    u_x, u_y = u_lon[1:-2, 1:-1], u_lat[1:-2, 1:-1]
    v_x, v_y = v_lon[1:-1, 2:-1], v_lat[1:-1, 2:-1]

    contour(eta_x, eta_y, eta_new, f'no_bc', 'Elevation')
    contour(u_x, u_y, u_new, f'no_bc', 'Zonal Velocity')
    contour(v_x, v_y, v_new, f'no_bc', 'Meridional Velocity')

    # find the sf and vp for full field
    if conv is None:
        sf_new, vp_new = tik_reg(alpha, u_new, v_new, dy, dx, ny, nx)
    elif conv == 'convergence':
        sf_new, vp_new, cf_list, grad_list = tik_reg(alpha, u_new, v_new, dy, dx, ny, nx, conv)
        # plot the convergences
        plot_one_convergence(cf_list, 'Cost Function')
        plot_one_convergence(grad_list, 'Gradient Norm')

    # Reconstruct the velocity fields using u_transform
    u_re, v_re = vel_from_helm(sf_new, vp_new, dx, dy)

    # differences between reconstructed field and originals
    du = u_new - u_re
    dv = v_new - v_re

    # plot sf and vp
    sf_new, vp_new = sf_new[1:-1, 1:-1], vp_new[1:-1, 1:-1]
    contour(eta_x, eta_y, sf_new, f'at time {time[time_index + 1]}', 'SF')
    contour(eta_x, eta_y, vp_new, f'at time {time[time_index + 1]}', 'VP')

    # plot reconstructed and differences
    contour(u_x, u_y, u_re, f'Reconstructed', 'Zonal Velocity')
    contour(v_x, v_y, v_re, f'Reconstructed', 'Meridional Velocity')

    contour(u_x, u_y, du, f'Reconstruction difference', 'Zonal Velocity')
    contour(v_x, v_y, dv, f'Reconstruction difference', 'Meridional Velocity')

    # calculate the relative error of the fields and plot
    # add a constant to the denominator to ensure we are not dividing by zero
    u_denom = u_new + 1e-6
    v_denom = v_new + 1e-6

    rel_err_u = abs((u_new - u_re) / u_denom) * 100
    rel_err_v = abs((v_new - v_re) / v_denom) * 100

    contour_err(u_x, u_y, rel_err_u, f'Relative Error', 'Zonal Velocity')
    contour_err(v_x, v_y, rel_err_v, f'Relative Error', 'Meridional Velocity')

    ######################################################################

    ##### FIND SF AND VP FOR WITHIN THE DOMAIN, NO BOUNDARY for increments
    deta_new = eta_diff[1:-2, 2:-1]
    du_new = u_diff[1:-2, 1:-1]
    dv_new = v_diff[1:-1, 2:-1]
    ny, nx = np.shape(eta_new)
    print(f'Shape of new elevation : {np.shape(deta_new)}')
    print(f' Shape of new u : {np.shape(du_new)}')
    print(f' Shape of new v : {np.shape(dv_new)}')

    eta_x, eta_y = eta_lon[1:-2, 2:-1], eta_lat[1:-2, 2:-1]
    u_x, u_y = u_lon[1:-2, 1:-1], u_lat[1:-2, 1:-1]
    v_x, v_y = v_lon[1:-1, 2:-1], v_lat[1:-1, 2:-1]

    contour(eta_x, eta_y, deta_new, f'no_bc increment', 'Elevation')
    contour(u_x, u_y, du_new, f'no_bc increment', 'Zonal Velocity')
    contour(v_x, v_y, dv_new, f'no_bc increment', 'Meridional Velocity')

    # find the sf and vp for increment
    if conv is None:
        sf_new, vp_new = tik_reg(alpha, du_new, dv_new, dy, dx, ny, nx)
    elif conv == 'convergence':
        sf_new, vp_new, cf_list, grad_list = tik_reg(alpha, du_new, dv_new, dy, dx, ny, nx, conv)
        # plot the convergences
        plot_one_convergence(cf_list, 'Cost Function for increments')
        plot_one_convergence(grad_list, 'Gradient Norm increments')

    dsf_new, dvp_new = sf_new[1:-1, 1:-1], vp_new[1:-1, 1:-1]

    contour(eta_x, eta_y, dsf_new, f'increment', 'SF')
    contour(eta_x, eta_y, dvp_new, f'increment', 'VP')

def balance_plots(time_index):
    """
    Plots from the gyre configuration for the full model fields, daily increments, and the balanced/unbalanced
    components of the velocities.
    Inputs: time_index, time of plots
    """
    dy, dx = param['dy'], param['dx']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

    # read eta, u and v from files at two times (1 day apart)
    # calculate the increment
    eta = read_file(eta_input_file, "sossheig", time_index=time_index)

    u = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]

    v = read_file(v_input_file, "vomecrty", time_index=time_index)[0]

    # lon and lat for each grid
    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    u_x, u_y = u_lon[1:-2, 1:-1], u_lat[1:-2, 1:-1]
    v_x, v_y = v_lon[1:-1, 2:-1], v_lat[1:-1, 2:-1]

    # balanced components
    # Use geostrophic balance to find the balanced velocities
    u_b = geostrophic_balance_D(eta[1:-2, 2:-1], 'u', u_y, dy)
    v_b = geostrophic_balance_D(eta[1:-2, 2:-1], 'v', v_y, dx)

    # Find the unbalanced components of the velocities
    u_u = u[1:-2, 1:-1] - u_b
    v_u = v[1:-1, 2:-1] - v_b
    # plot full fields, and balanced and unbalanced components
    contour(u_x, u_y, u[1:-2, 1:-1], f'at time {time[time_index]}', 'Full Zonal V')
    contour(v_x, v_y, v[1:-1, 2:-1], f'at time {time[time_index]}', 'Full Meridional V')

    contour(u_x, u_y, u_b, f'at time {time[time_index]}', 'Balanced Zonal V')
    contour(v_x, v_y, v_b, f'at time {time[time_index]}', 'Balanced Meridional V')

    contour(u_x, u_y, u_u, f'at time {time[time_index]}', 'Unbalanced Zonal V')
    contour(v_x, v_y, v_u, f'at time {time[time_index]}', 'Unbalanced Meridional V')

    ##################################################################################
    # increments
    # netcdf file locations
    eta_1 = read_file(eta_input_file, "sossheig", time_index=time_index + 1)
    eta_diff = eta_1 - eta

    u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)[0]
    u_diff = u_1 - u

    v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
    v_diff = v_1 - v

    # balanced components
    # Use geostrophic balance to find the balanced velocities
    du_b = geostrophic_balance_D(eta_diff[1:-2, 2:-1], 'u', u_y, dy)
    dv_b = geostrophic_balance_D(eta_diff[1:-2, 2:-1], 'v', v_y, dx)

    # Find the unbalanced components of the velocities
    du_u = u_diff[1:-2, 1:-1] - du_b
    dv_u = v_diff[1:-1, 2:-1] - dv_b
    # plot full fields, and balanced and unbalanced components
    contour(u_x, u_y, u_diff[1:-2, 1:-1], f'at time {time[time_index]}', 'Zonal V Increment')
    contour(v_x, v_y, v_diff[1:-1, 2:-1], f'at time {time[time_index]}', 'Meridional V Increment')

    contour(u_x, u_y, du_b, f'at time {time[time_index]}', 'Balanced Zonal V Increment')
    contour(v_x, v_y, dv_b, f'at time {time[time_index]}', 'Balanced Meridional V Increment ')

    contour(u_x, u_y, du_u, f'at time {time[time_index]}', 'Unbalanced Zonal V Increment')
    contour(v_x, v_y, dv_u, f'at time {time[time_index]}', 'Unbalanced Meridional V Increment')


def control_plots(alpha, time_index):
    """
    Plots from the gyre configuration for the control variables - elevation increment, unbalanced SF and VP.
    Inputs: alpha, Tikhonov's regularisation parameter
            time_index, time of plots
    """
    dy, dx = param['dy'], param['dx']

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

    # lon and lat for each grid
    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    # size of eta
    ny, nx = np.shape(eta_lat)[0] - 1, np.shape(eta_lon)[1] - 1

    ##### FIND SF AND VP FOR WITHIN THE DOMAIN, NO BOUNDARY for increments
    deta_new = eta_diff[1:-2, 2:-1]
    du_new = u_diff[1:-2, 1:-1]
    dv_new = v_diff[1:-1, 2:-1]
    ny, nx = np.shape(deta_new)
    print(f'Shape of new elevation : {np.shape(deta_new)}')
    print(f' Shape of new u : {np.shape(du_new)}')
    print(f' Shape of new v : {np.shape(dv_new)}')

    eta_x, eta_y = eta_lon[1:-2, 2:-1], eta_lat[1:-2, 2:-1]
    u_x, u_y = u_lon[1:-2, 1:-1], u_lat[1:-2, 1:-1]
    v_x, v_y = v_lon[1:-1, 2:-1], v_lat[1:-1, 2:-1]

    # find the control variables from the model increments
    d_eta, sf_u, vp_u, du_mean, dv_mean = T_transform(deta_new, du_new, dv_new, dx, dy, u_y, v_y, alpha)

    dsf_new, dvp_new = sf_u[1:-1, 1:-1], vp_u[1:-1, 1:-1]

    contour(eta_x, eta_y, dsf_new, f'Unbalanced increment', 'SF')
    contour(eta_x, eta_y, dvp_new, f'Unbalanced increment', 'VP')

if __name__ == '__main__':
    # Tikhonov's regularisation parameter
    alpha = 0
    # time to plot
    time_index = 700
    conv = 'convergence'
    #transform_plots(alpha, time_index, conv)
    #control_plots(alpha, time_index)
    balance_plots(time_index)

