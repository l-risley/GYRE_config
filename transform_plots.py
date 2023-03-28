"""
Plot the model variables, control variables, full sf and vp fields as well.
@author: Laura Risley 2023
"""
import numpy as np

from transforms.U_transform import *
from transforms.T_transform import *
from read_nemo_fields import *
from plot_gyre_fields import *
from general_functions import *

def transform_plots(alpha, time_index):
    """
    """
    dy, dx = param['dy'], param['dx']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

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
    num_times = np.size(time)

    print(f'Length of elevation lattitude : {np.shape(eta_lat)} and longitude : {np.shape(eta_lon)}.')
    print(f'Length of u lattitude : {np.shape(u_lat)} and longitude : {np.shape(u_lon)}.')
    print(f'Length of v lattitude : {np.shape(v_lat)} and longitude : {np.shape(v_lon)}.')

    ## 2d arrays for correlation analysis
    # size of eta
    ny, nx = np.shape(eta_lat)[0]-1, np.shape(eta_lon)[1]-1
    print(f'ny = {ny}, nx = {nx}')

    # plot fields at both times
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

    # set mask fill value

    # The fields are not the correct shape
    eta_0, eta_diff = eta_0[:-1, 1:], eta_diff[:-1, 1:]
    u_0, u_diff = u_0[:-1, :], u_diff[:-1, :]
    v_0, v_diff = v_0[:, 1:], v_diff[:, 1:]

    print(f'Shape of elevation : {np.shape(eta_0)}')
    print(f' Shape of u : {np.shape(u_0)}')
    print(f' Shape of v : {np.shape(v_0)}')

    # find the sf and vp for full field
    #print(u_0)
    #print(v_0)
    sf_0, vp_0 = tik_reg(alpha, u_0, v_0, dy, dx, ny, nx)

    # reconstruct
    u_re, v_re = vel_from_helm(sf_0, vp_0, dx, dy)

    # differences
    du = u_0 - u_re
    dv = v_0 - v_re
    sf_0, vp_0 = sf_0[1:-1, 1:-1], vp_0[1:-1, 1:-1]
    contour(eta_lon, eta_lat, sf_0, f'at time {time[time_index + 1]}', 'Streamfunction')
    contour(eta_lon, eta_lat, vp_0, f'at time {time[time_index + 1]}', 'Velocity Potential')

    contour(u_lon, u_lat, u_re, f'recon', 'Zonal Velocity')
    contour(v_lon, v_lat, v_re, f'recon', 'Meridional Velocity')

    contour(u_lon, u_lat, du, f'recon diff', 'Zonal Velocity')
    contour(v_lon, v_lat, dv, f'recon-diff', 'Meridional Velocity')

    """
    # find sf and vp for increment
    sf_diff, vp_diff = tik_reg(alpha, u_diff, v_diff, dy, dx, ny, nx)
    sf_diff, vp_diff = sf_diff[1:-1, 1:-1], vp_diff[1:-1, 1:-1]
    
    # plot both
    
    contour(eta_lon, eta_lat, sf_diff, f'increment', 'Streamfunction')
    contour(eta_lon, eta_lat, vp_diff, f'increment', 'Velocity Potential')
    
    # find control variables
    eta_diff, sf_u, vp_u, du_mean, dv_mean = T_transform(eta_diff, u_diff, v_diff, dx, dy, u_lat, v_lat, alpha)

    # plot control variables
    contour(eta_lon, eta_lat, sf_u[1:-1, 1:-1], f'increment', 'Unbalanced Streamfunction')
    contour(eta_lon, eta_lat, vp_u[1:-1, 1:-1], f'increment', 'Velocity Potential')
    
    # recalculate the model variables
    eta_diff, u_re, v_re = U_transform(eta_diff, sf_u, vp_u, du_mean, dv_mean, dx, dy, u_lat, v_lat)

    # plot reconstructed velocities

    # calculate the relative error of the fields and plot
# add a constant to the denominator to ensure we are not dividing by zero
    u_denom = u_diff + 1e-6
    v_denom = v_diff + 1e-6

    rel_err_u = abs((u_diff - u_re) / u_denom) * 100
    rel_err_v = abs((v_diff - v_re) / v_denom) * 100
    """
if __name__ == '__main__':
    # Tikhonov's regularisation parameter
    alpha = 9e-8
    # time to plot
    time_index = 700
    transform_plots(alpha, time_index)

