"""
Correlation analysis for the gyre configuration. Finding the correlations at each grid point for the model
and control variables.
@author: Laura Risley 2023
"""

import numpy as np
import matplotlib.pyplot as plt

from read_nemo_fields import *
from gyre_setup import *
from transforms.T_transform import *

# location of correlation plots
lx, ux, ly, uy = 150, -2, 10, 102 #2, 52, 2, 52
eta_loc_y, eta_loc_x = [ly, uy], [lx, ux]
u_loc_y, u_loc_x = [ly, uy], [lx, ux+1]
v_loc_y, v_loc_x = [ly, uy+1], [lx, ux]

def find_corr_matrix(x, ny, nx):
    """
    Find the correlation matrix of two variables at each grid point, take from the correlation matrix
    of all grid points.
    Input: x, full correlation matrix
    Output: x_diag, correlation matrix at grid points
    """
    # take the diagonal of the botton left of matrix
    m, n = np.shape(x)
    m, n = np.int(m * 0.5), np.int(n * 0.5)
    x_new = x[m:, :n]
    # take the diagonal
    x_diag = np.diag(x_new)
    # reshape into an array
    return np.reshape(x_diag, (ny, nx))

def plot_corr(corr_matrix, variable_1:str, variable_2: str, lon, lat):
    """
    Plot the correlation matrix or variable 1 and 2.
    Inputs: corr_matrix, correlation matrix
            variable_1, first variable
            variable_2, second variable
            lon, longitude coordinates
            lat, lattitude coordinates
    """
    x, y = lon, lat
    plt.pcolormesh(x, y, corr_matrix, cmap='viridis', shading='auto')
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.title(f'Corr of {variable_1} and {variable_2}')
    plt.colorbar(label=f'Correlation coefficient')
    plt.savefig(f'corr{variable_1}{variable_2}.png')
    plt.show()

def corr_mv(n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """
    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_u, sum_sq_v = np.zeros_like(eta_lat), np.zeros_like(u_lat), np.zeros_like(v_lat)

    sum_eta, sum_u, sum_v = np.zeros_like(eta_lat), np.zeros_like(u_lat), np.zeros_like(v_lat)

    sum_etau, sum_etav, sum_uv = np.zeros_like(eta_lat), np.zeros_like(eta_lat), np.zeros_like(eta_lat)

    for time_index in range(n): #range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta = read_file(eta_input_file, "sossheig", time_index=time_index)
        u = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
        v = read_file(v_input_file, "vomecrty", time_index=time_index)[0]

        sum_sq_eta += np.square(eta)
        sum_sq_u += np.square(u)
        sum_sq_v += np.square(v)

        sum_eta += eta
        sum_u += u
        sum_v += v

        sum_etau += eta*u
        sum_etav += eta*v
        sum_uv += u*v

    ## calculate values of correlations
    # mean values
    mn_eta, mn_u, mn_v = sum_eta/n, sum_u/n, sum_v/n
    # variances
    var_eta = (sum_sq_eta/n) - (np.square(mn_eta))
    var_u = (sum_sq_u/n) - (np.square(mn_u))
    var_v = (sum_sq_v/n) - (np.square(mn_v))
    # covariances
    cov_eta_u = (sum_etau/n) - (mn_eta*mn_u)
    cov_eta_v = (sum_etav / n) - (mn_eta * mn_v)
    cov_u_v = (sum_uv / n ) - (mn_u * mn_v)

    # correlations
    corr_eta_u = cov_eta_u / np.sqrt(var_eta * var_u)
    corr_eta_v = cov_eta_v / np.sqrt(var_eta * var_v)
    corr_u_v = cov_u_v / np.sqrt(var_u * var_v)

    ## plot correlation matrix
    # full model fields
    plot_corr(corr_eta_u, 'Elevation', 'Zonal Velocity', eta_lon, eta_lat)
    plot_corr(corr_eta_v, 'Elevation', 'Meridional Velocity', eta_lon, eta_lat)
    plot_corr(corr_u_v, 'Zonal Velocity', 'Meridional Velocity', eta_lon, eta_lat)


def corr_mv_increments(n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """
    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_u, sum_sq_v = np.zeros_like(eta_lat), np.zeros_like(u_lat), np.zeros_like(v_lat)

    sum_eta, sum_u, sum_v = np.zeros_like(eta_lat), np.zeros_like(u_lat), np.zeros_like(v_lat)

    sum_etau, sum_etav, sum_uv = np.zeros_like(eta_lat), np.zeros_like(eta_lat), np.zeros_like(eta_lat)

    for time_index in range(n):  # range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)
        eta_1 = read_file(eta_input_file, "sossheig", time_index=time_index + 1)
        eta_diff = eta_1 - eta_0

        u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
        u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)[0]
        u_diff = u_1 - u_0

        v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
        v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
        v_diff = v_1 - v_0

    sum_sq_eta += np.square(eta)
        sum_sq_u += np.square(u)
        sum_sq_v += np.square(v)

        sum_eta += eta
        sum_u += u
        sum_v += v

        sum_etau += eta * u
        sum_etav += eta * v
        sum_uv += u * v

    ## calculate values of correlations
    # mean values
    mn_eta, mn_u, mn_v = sum_eta / n, sum_u / n, sum_v / n
    # variances
    var_eta = (sum_sq_eta / n) - (np.square(mn_eta))
    var_u = (sum_sq_u / n) - (np.square(mn_u))
    var_v = (sum_sq_v / n) - (np.square(mn_v))
    # covariances
    cov_eta_u = (sum_etau / n) - (mn_eta * mn_u)
    cov_eta_v = (sum_etav / n) - (mn_eta * mn_v)
    cov_u_v = (sum_uv / n) - (mn_u * mn_v)

    # correlations
    corr_eta_u = cov_eta_u / np.sqrt(var_eta * var_u)
    corr_eta_v = cov_eta_v / np.sqrt(var_eta * var_v)
    corr_u_v = cov_u_v / np.sqrt(var_u * var_v)

    ## plot correlation matrix
    # full model fields
    plot_corr(corr_eta_u, 'Elevation', 'Zonal Velocity', eta_lon, eta_lat)
    plot_corr(corr_eta_v, 'Elevation', 'Meridional Velocity', eta_lon, eta_lat)
    plot_corr(corr_u_v, 'Zonal Velocity', 'Meridional Velocity', eta_lon, eta_lat)

def corr_mv_at_grid_points(n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    lon, lat, time = read_file_info(eta_input_file)

    ## 2d arrays for correlation analysis
    # size of eta
    ny, nx = np.shape(lat[ly:uy, lx:ux])[0], np.shape(lon[ly:uy, lx:ux])[1]
    eta_u = np.empty((n, 2*ny*nx))
    eta_v = np.empty((n, 2*ny*nx))
    u_v = np.empty((n, 2*ny*nx))
    print(np.shape(eta_u))

    for time_index in range(n): #range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)
        u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
        v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]

        ## flatten model arrays
        eta_0 = eta_0[ly:uy, lx:ux].flatten()
        #print(np.shape(eta_diff))
        u_0 = u_0[ly:uy, lx:ux].flatten()
        #print(np.shape(u_diff))
        v_0 = v_0[ly:uy, lx:ux].flatten()
        # interp velocities to eta grid
        #u_diff, v_diff = interp_zonal(u_diff).flatten(), interp_merid(v_diff).flatten()

        # add to matrix to find correlations
        #print(np.shape(np.concatenate((eta_diff, u_diff))))
        eta_u[time_index, :] = np.concatenate((eta_0, u_0))
        eta_v[time_index, :] = np.concatenate((eta_0, v_0))
        u_v[time_index, :] = np.concatenate((u_0, v_0))

    #print('Finding the correlations.')
    # find model increment correlations
    corr_deta_du_full = np.corrcoef(eta_u, rowvar=False)
    corr_deta_dv_full = np.corrcoef(eta_v, rowvar=False)
    corr_du_dv_full = np.corrcoef(u_v, rowvar=False)

    ## extract the correlations we need
    # find model increment correlations
    corr_deta_du = find_corr_matrix(corr_deta_du_full, ny, nx)
    corr_deta_dv = find_corr_matrix(corr_deta_dv_full, ny, nx)
    corr_du_dv = find_corr_matrix(corr_du_dv_full, ny, nx)

    ## plot correlation matrix
    # model increments
    eta_lon, eta_lat = lon[ly:uy, lx:ux], lat[ly:uy, lx:ux]
    plot_corr(corr_deta_du, 'Elevation', 'Zonal Velocity', eta_lon, eta_lat)
    plot_corr(corr_deta_dv, 'Elevation', 'Meridional Velocity', eta_lon, eta_lat)
    plot_corr(corr_du_dv, 'Zonal Velocity', 'Meridional Velocity', eta_lon, eta_lat)

def corr_inc_mv_at_grid_points(n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    lon, lat, time = read_file_info(eta_input_file)

    ## 2d arrays for correlation analysis
    # size of eta
    ny, nx = np.shape(lat[ly:uy, lx:ux])[0], np.shape(lon[ly:uy, lx:ux])[1]
    eta_u = np.empty((n, 2*ny*nx))
    eta_v = np.empty((n, 2*ny*nx))
    u_v = np.empty((n, 2*ny*nx))
    print(np.shape(eta_u))

    for time_index in range(n): #range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)
        eta_1 = read_file(eta_input_file, "sossheig", time_index=time_index + 1)
        eta_diff = eta_1 - eta_0

        u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
        u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)[0]
        u_diff = u_1 - u_0

        v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
        v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
        v_diff = v_1 - v_0

        ## flatten model arrays
        eta_diff = eta_diff[ly:uy, lx:ux].flatten()
        #print(np.shape(eta_diff))
        u_diff = u_diff[ly:uy, lx:ux].flatten()
        #print(np.shape(u_diff))
        v_diff = v_diff[ly:uy, lx:ux].flatten()
        # interp velocities to eta grid
        #u_diff, v_diff = interp_zonal(u_diff).flatten(), interp_merid(v_diff).flatten()

        # add to matrix to find correlations
        #print(np.shape(np.concatenate((eta_diff, u_diff))))
        eta_u[time_index, :] = np.concatenate((eta_diff, u_diff))
        eta_v[time_index, :] = np.concatenate((eta_diff, v_diff))
        u_v[time_index, :] = np.concatenate((u_diff, v_diff))

    #print('Finding the correlations.')
    # find model increment correlations
    corr_deta_du_full = np.corrcoef(eta_u, rowvar=False)
    corr_deta_dv_full = np.corrcoef(eta_v, rowvar=False)
    corr_du_dv_full = np.corrcoef(u_v, rowvar=False)

    ## extract the correlations we need
    # find model increment correlations
    corr_deta_du = find_corr_matrix(corr_deta_du_full, ny, nx)
    corr_deta_dv = find_corr_matrix(corr_deta_dv_full, ny, nx)
    corr_du_dv = find_corr_matrix(corr_du_dv_full, ny, nx)

    ## plot correlation matrix
    # model increments
    eta_lon, eta_lat = lon[ly:uy, lx:ux], lat[ly:uy, lx:ux]
    plot_corr(corr_deta_du, 'Elevation', 'Zonal Velocity increments', eta_lon, eta_lat)
    plot_corr(corr_deta_dv, 'Elevation', 'Meridional Velocity increments', eta_lon, eta_lat)
    plot_corr(corr_du_dv, 'Zonal Velocity', 'Meridional Velocity increments', eta_lon, eta_lat)

def corr_mv_ub_at_grid_points(n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """
    dy, dx = param['dy'], param['dx']
    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    lon, lat, time = read_file_info(eta_input_file)

    ## 2d arrays for correlation analysis
    # size of eta
    ny, nx = np.shape(lat[2:52, 2:52])[0], np.shape(lon[2:52, 2:52])[1]
    eta_u = np.empty((n, 2*ny*nx))
    eta_v = np.empty((n, 2*ny*nx))
    u_v = np.empty((n, 2*ny*nx))
    print(np.shape(eta_u))

    for time_index in range(n): #range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)
        eta_1 = read_file(eta_input_file, "sossheig", time_index=time_index + 1)
        eta_diff = eta_1 - eta_0

        u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
        u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)[0]
        u_diff = u_1 - u_0

        v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
        v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
        v_diff = v_1 - v_0

        # find the balanced velocities
        eta_diff = eta_diff[ly:uy, lx:ux]
        u_lat = lat[ly:uy, lx:ux+1]
        v_lat = lat[ly:uy+1, lx:ux]
        u_diff_b = geostrophic_balance_D(eta_diff, 'u', u_lat, dy)
        v_diff_b = geostrophic_balance_D(eta_diff, 'v', v_lat, dx)
        u_diff_u = u_diff[ly:uy, lx:ux+1] - u_diff_b
        v_diff_u = v_diff[ly:uy+1, lx:ux] - v_diff_b

        ## flatten model arrays
        eta_diff = eta_diff.flatten()

        # interp velocities to eta grid
        u_diff, v_diff = interp_zonal(u_diff_u).flatten(), interp_merid(v_diff_u).flatten()

        # add to matrix to find correlations
        #print(np.shape(np.concatenate((eta_diff, u_diff))))
        eta_u[time_index, :] = np.concatenate((eta_diff, u_diff))
        eta_v[time_index, :] = np.concatenate((eta_diff, v_diff))
        u_v[time_index, :] = np.concatenate((u_diff, v_diff))

    #print('Finding the correlations.')
    # find model increment correlations
    corr_deta_du_full = np.corrcoef(eta_u, rowvar=False)
    corr_deta_dv_full = np.corrcoef(eta_v, rowvar=False)
    corr_du_dv_full = np.corrcoef(u_v, rowvar=False)

    ## extract the correlations we need
    # find model increment correlations
    corr_deta_du = find_corr_matrix(corr_deta_du_full, ny, nx)
    corr_deta_dv = find_corr_matrix(corr_deta_dv_full, ny, nx)
    corr_du_dv = find_corr_matrix(corr_du_dv_full, ny, nx)

    ## plot correlation matrix
    # model unbalanced velocity increments
    eta_lon, eta_lat = lon[ly:uy, lx:ux], lat[ly:uy, lx:ux]
    plot_corr(corr_deta_du, 'Elevation', 'Unbal Zonal Velocity increments', eta_lon, eta_lat)
    plot_corr(corr_deta_dv, 'Elevation', 'Unbal Meridional Velocity increments', eta_lon, eta_lat)
    plot_corr(corr_du_dv, 'Unbal Zonal Velocity', 'Unbal Meridional Velocity increments', eta_lon, eta_lat)

def corr_cv_at_grid_points(alpha, n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """
    dy, dx = param['dy'], param['dx']
    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    lon, lat, time = read_file_info(eta_input_file)

    ## 2d arrays for correlation analysis
    # size of eta
    ny, nx = np.shape(lat[ly:uy, lx:ux])[0], np.shape(lon[ly:uy, lx:ux])[1]
    eta_sf = np.empty((n, 2*ny*nx))
    eta_vp = np.empty((n, 2*ny*nx))
    sf_vp = np.empty((n, 2*ny*nx))
    print(np.shape(eta_sf))

    for time_index in range(n): #range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)
        eta_1 = read_file(eta_input_file, "sossheig", time_index=time_index + 1)
        eta_diff = eta_1 - eta_0

        u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
        u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)[0]
        u_diff = u_1 - u_0

        v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
        v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
        v_diff = v_1 - v_0

        # find the balanced velocities
        eta_diff = eta_diff[ly:uy, lx:ux]
        u_diff, u_lat = u_diff[ly:uy, lx:ux+1], lat[ly:uy, lx:ux+1]
        v_diff, v_lat = v_diff[ly:uy+1, lx:ux], lat[ly:uy+1, lx:ux]

        # find the control variables
        print(f'Finding control variables at {time_index} days.')
        eta_diff, sf_u, vp_u, du_mean, dv_mean = T_transform(eta_diff, u_diff, v_diff, dx, dy, u_lat, v_lat, alpha)

        ## flatten model arrays
        eta_diff = eta_diff.flatten()

        # flatten control arrays
        d_sf_u = sf_u[1:-1, 1:-1].flatten()
        d_vp_u = vp_u[1:-1, 1:-1].flatten()

        # add to matrix to find control correlations
        eta_sf[time_index, :] = np.concatenate((eta_diff, d_sf_u))
        eta_vp[time_index, :] = np.concatenate((eta_diff, d_vp_u))
        sf_vp[time_index, :] = np.concatenate((d_sf_u, d_vp_u))

        # find control increment correlations
    corr_deta_dsf_full = np.corrcoef(eta_sf, rowvar=False)
    corr_deta_dvp_full = np.corrcoef(eta_vp, rowvar=False)
    corr_dsf_dvp_full = np.corrcoef(sf_vp, rowvar=False)

    ## extract the correlations we need
    # find control increment correlations
    corr_deta_dsf = find_corr_matrix(corr_deta_dsf_full, ny, nx)
    corr_deta_dvp = find_corr_matrix(corr_deta_dvp_full, ny, nx)
    corr_dsf_dvp = find_corr_matrix(corr_dsf_dvp_full, ny, nx)

    ## plot correlation matrix
    # control increments
    eta_lon, eta_lat = lon[ly:uy, lx:ux], lat[ly:uy, lx:ux]
    plot_corr(corr_deta_dsf, 'Elevation', 'Unbal SF increments', eta_lon, eta_lat)
    plot_corr(corr_deta_dvp, 'Elevation', 'Unbal VP increments', eta_lon, eta_lat)
    plot_corr(corr_dsf_dvp, 'Unbal SF', 'Unbal VP increments', eta_lon, eta_lat)

def corr_inc_sfvp_at_grid_points(alpha, n):
    """
    Find the correlations of the control variable increments at each grid point. Produce correlation matrix plots.
    Inputs: alpha, Tikhonov's regularisation parameter
            n, number of time intervals to store perturbations
    """
    # dimensions
    dx, dy = param['dx'], param['dy']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(eta_input_file)

    ## 2d arrays for correlation analysis
    # size of eta
    ny, nx = np.shape(lat[ly:uy, lx:ux])[0], np.shape(lon[ly:uy, lx:ux])[1]
    eta_sf = np.empty((n, 2 * ny * nx))
    eta_vp = np.empty((n, 2 * ny * nx))
    sf_vp = np.empty((n, 2 * ny * nx))

    for time_index in range(n):  # range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)
        eta_1 = read_file(eta_input_file, "sossheig", time_index=time_index + 1)
        eta_diff = eta_1 - eta_0

        u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
        u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)[0]
        u_diff = u_1 - u_0

        v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
        v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
        v_diff = v_1 - v_0

        # find the balanced velocities
        eta_diff = eta_diff[ly:uy, lx:ux]
        u_diff, u_lat = u_diff[ly:uy, lx:ux+1], lat[ly:uy, lx:ux+1]
        v_diff, v_lat = v_diff[ly:uy+1, lx:ux], lat[ly:uy+1, lx:ux]

        ny, nx = np.shape(eta_diff)
        sf, vp = tik_reg(alpha, u_diff, v_diff, dy, dx, ny, nx)

        ## flatten model arrays
        eta_diff = eta_diff.flatten()

        # flatten control arrays
        sf = sf[1:-1, 1:-1].flatten()
        vp = vp[1:-1, 1:-1].flatten()

        # add to matrix to find control correlations
        eta_sf[time_index, :] = np.concatenate((eta_diff, sf))
        eta_vp[time_index, :] = np.concatenate((eta_diff, vp))
        sf_vp[time_index, :] = np.concatenate((sf, vp))

    # find control increment correlations
    corr_deta_dsf_full = np.corrcoef(eta_sf, rowvar=False)
    corr_deta_dvp_full = np.corrcoef(eta_vp, rowvar=False)
    corr_dsf_dvp_full = np.corrcoef(sf_vp, rowvar=False)

    ## extract the correlations we need
    # find control increment correlations
    corr_deta_dsf = find_corr_matrix(corr_deta_dsf_full, ny, nx)
    corr_deta_dvp = find_corr_matrix(corr_deta_dvp_full, ny, nx)
    corr_dsf_dvp = find_corr_matrix(corr_dsf_dvp_full, ny, nx)

    ## plot correlation matrix
    # control increments
    eta_lon, eta_lat = lon[ly:uy, lx:ux], lat[ly:uy, lx:ux]
    plot_corr(corr_deta_dsf, 'Elevation', 'SF increments', eta_lon, eta_lat)
    plot_corr(corr_deta_dvp, 'Elevation', 'VP increments', eta_lon, eta_lat)
    plot_corr(corr_dsf_dvp, 'SF', 'VP increments', eta_lon, eta_lat)


def corr_full_sfvp_at_grid_points(alpha, n):
    """
    Find the correlations of the control variable increments at each grid point. Produce correlation matrix plots.
    Inputs: alpha, Tikhonov's regularisation parameter
            n, number of time intervals to store perturbations
    """
    # dimensions
    dx, dy = param['dx'], param['dy']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(eta_input_file)

    ## 2d arrays for correlation analysis
    # size of eta
    ny, nx = np.shape(lat[ly:uy, lx:ux])[0], np.shape(lon[ly:uy, lx:ux])[1]
    eta_sf = np.empty((n, 2 * ny * nx))
    eta_vp = np.empty((n, 2 * ny * nx))
    sf_vp = np.empty((n, 2 * ny * nx))

    for time_index in range(n):  # range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)

        u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
        print(np.shape(u_0))
        v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]

        # find the control variables
        print(f'Finding control variables at {time_index} days.')
        eta_new = eta_0[ly:uy, lx:ux]
        u_new = u_0[ly:uy, lx:ux+1]
        v_new = v_0[ly:uy+1, lx:ux]

        ny, nx = np.shape(eta_new)
        sf, vp = tik_reg(alpha, u_new, v_new, dy, dx, ny, nx)

        ## flatten model arrays
        eta_new = eta_new.flatten()

        # flatten control arrays
        sf = sf[1:-1, 1:-1].flatten()
        vp = vp[1:-1, 1:-1].flatten()

        # add to matrix to find control correlations
        eta_sf[time_index, :] = np.concatenate((eta_new, sf))
        eta_vp[time_index, :] = np.concatenate((eta_new, vp))
        sf_vp[time_index, :] = np.concatenate((sf, vp))

    # find control increment correlations
    corr_deta_dsf_full = np.corrcoef(eta_sf, rowvar=False)
    corr_deta_dvp_full = np.corrcoef(eta_vp, rowvar=False)
    corr_dsf_dvp_full = np.corrcoef(sf_vp, rowvar=False)

    ## extract the correlations we need
    # find control increment correlations
    corr_deta_dsf = find_corr_matrix(corr_deta_dsf_full, ny, nx)
    corr_deta_dvp = find_corr_matrix(corr_deta_dvp_full, ny, nx)
    corr_dsf_dvp = find_corr_matrix(corr_dsf_dvp_full, ny, nx)

    ## plot correlation matrix
    # control increments
    eta_lon, eta_lat = lon[ly:uy, lx:ux], lat[ly:uy, lx:ux]
    plot_corr(corr_deta_dsf, 'Elevation', 'Streamfunction', eta_lon, eta_lat)
    plot_corr(corr_deta_dvp, 'Elevation', 'Velocity Potential', eta_lon, eta_lat)
    plot_corr(corr_dsf_dvp, 'Streamfunction', 'Velocity Potential', eta_lon, eta_lat)

if __name__ == '__main__':
        # Tikhonov's regularisation parameter
        alpha = 0
        # number of samples
        n = 150
        #corr_mv_at_grid_points(n)
        #corr_inc_mv_at_grid_points(n)
        #corr_mv_ub_at_grid_points(n)
        #corr_cv_at_grid_points(alpha, n)
        #corr_full_sfvp_at_grid_points(alpha, n)
        #corr_inc_sfvp_at_grid_points(alpha, n)
        corr_mv(n)
