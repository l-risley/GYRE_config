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

        sum_sq_eta += np.square(eta_diff)
        sum_sq_u += np.square(u_diff)
        sum_sq_v += np.square(v_diff)

        sum_eta += eta_diff
        sum_u += u_diff
        sum_v += v_diff

        sum_etau += eta_diff * u_diff
        sum_etav += eta_diff * v_diff
        sum_uv += u_diff * v_diff

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
    plot_corr(corr_eta_u, 'Elevation', 'Zonal Velocity increments', eta_lon, eta_lat)
    plot_corr(corr_eta_v, 'Elevation', 'Meridional Velocity increments', eta_lon, eta_lat)
    plot_corr(corr_u_v, 'Zonal Velocity', 'Meridional Velocity increments', eta_lon, eta_lat)

def corr_mv_ub_increments(n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """

    dy, dx = param['dy'], param['dx']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    eta_x, eta_y = eta_lon[1:-2, 2:-1], eta_lat[1:-2, 2:-1]
    u_x, u_y = u_lon[1:-2, 1:-1], u_lat[1:-2, 1:-1]
    v_x, v_y = v_lon[1:-1, 2:-1], v_lat[1:-1, 2:-1]

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_u, sum_sq_v = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

    sum_eta, sum_u, sum_v = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

    sum_etau, sum_etav, sum_uv = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

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
        eta_diff = eta_diff[1:-2, 2:-1]
        u_diff_b = geostrophic_balance_D(eta_diff, 'u', u_y, dy)
        v_diff_b = geostrophic_balance_D(eta_diff, 'v', v_y, dx)
        u_diff_u = u_diff[1:-2, 1:-1] - u_diff_b
        v_diff_u = v_diff[1:-1, 2:-1] - v_diff_b

        # interp velocities to eta grid
        u_diff, v_diff = interp_zonal(u_diff_u), interp_merid(v_diff_u)

        sum_sq_eta += np.square(eta_diff)
        sum_sq_u += np.square(u_diff)
        sum_sq_v += np.square(v_diff)

        sum_eta += eta_diff
        sum_u += u_diff
        sum_v += v_diff

        sum_etau += eta_diff * u_diff
        sum_etav += eta_diff * v_diff
        sum_uv += u_diff * v_diff

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
    plot_corr(corr_eta_u, 'Elevation', 'Unbal Zonal Velocity increments', eta_x, eta_y)
    plot_corr(corr_eta_v, 'Elevation', 'Unbal Meridional Velocity increments', eta_x, eta_y)
    plot_corr(corr_u_v, 'Unbal Zonal Velocity', 'Unbal Meridional Velocity increments', eta_x, eta_y)

def corr_sf_vp(alpha, n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """

    dy, dx = param['dy'], param['dx']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    eta_x, eta_y = eta_lon[1:-2, 2:-1], eta_lat[1:-2, 2:-1]

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_sf, sum_sq_vp = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

    sum_eta, sum_sf, sum_vp = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

    sum_etasf, sum_etavp, sum_sfvp = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

    for time_index in range(n):  # range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta = read_file(eta_input_file, "sossheig", time_index=time_index)

        u = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]

        v = read_file(v_input_file, "vomecrty", time_index=time_index)[0]

        # remove boundaries
        eta = eta[1:-2, 2:-1]
        u = u[1:-2, 1:-1]
        v = v[1:-1, 2:-1]

        ny, nx = np.shape(eta)
        sf, vp = tik_reg(alpha, u, v, dy, dx, ny, nx)

        # reduce the control arrays to be the same size as eta grid
        sf = sf[1:-1, 1:-1]
        vp = vp[1:-1, 1:-1]

        sum_sq_eta += np.square(eta)
        sum_sq_sf += np.square(sf)
        sum_sq_vp += np.square(vp)

        sum_eta += eta
        sum_sf += sf
        sum_vp += vp

        sum_etasf += eta * sf
        sum_etavp += eta * vp
        sum_sfvp += sf * vp

    ## calculate values of correlations
    # mean values
    mn_eta, mn_sf, mn_vp = sum_eta / n, sum_sf / n, sum_vp / n
    # variances
    var_eta = (sum_sq_eta / n) - (np.square(mn_eta))
    var_sf = (sum_sq_sf / n) - (np.square(mn_sf))
    var_vp = (sum_sq_vp / n) - (np.square(mn_vp))
    # covariances
    cov_eta_sf = (sum_etasf / n) - (mn_eta * mn_sf)
    cov_eta_vp = (sum_etavp / n) - (mn_eta * mn_vp)
    cov_sf_vp = (sum_sfvp / n) - (mn_sf * mn_vp)

    # correlations
    corr_eta_sf = cov_eta_sf / np.sqrt(var_eta * var_sf)
    corr_eta_vp = cov_eta_vp / np.sqrt(var_eta * var_vp)
    corr_sf_vp = cov_sf_vp / np.sqrt(var_sf * var_vp)

    ## plot correlation matrix
    # full model fields
    plot_corr(corr_eta_sf, 'Elevation', 'SF', eta_x, eta_y)
    plot_corr(corr_eta_vp, 'Elevation', 'VP', eta_x, eta_y)
    plot_corr(corr_sf_vp, 'SF', 'VP', eta_x, eta_y)

def corr_sf_vp_increments(alpha, n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """

    dy, dx = param['dy'], param['dx']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    eta_x, eta_y = eta_lon[1:-2, 2:-1], eta_lat[1:-2, 2:-1]
    u_x, u_y = u_lon[1:-2, 1:-1], u_lat[1:-2, 1:-1]
    v_x, v_y = v_lon[1:-1, 2:-1], v_lat[1:-1, 2:-1]

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_sf, sum_sq_vp = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

    sum_eta, sum_sf, sum_vp = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

    sum_etasf, sum_etavp, sum_sfvp = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

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

        # remove boundaries
        eta_diff = eta_diff[1:-2, 2:-1]
        u_diff = u_diff[1:-2, 1:-1]
        v_diff = v_diff[1:-1, 2:-1]

        ny, nx = np.shape(eta_diff)
        sf, vp = tik_reg(alpha, u_diff, v_diff, dy, dx, ny, nx)

        # reduce the control arrays to be the same size as eta grid
        sf = sf[1:-1, 1:-1]
        vp = vp[1:-1, 1:-1]

        sum_sq_eta += np.square(eta_diff)
        sum_sq_sf += np.square(sf)
        sum_sq_vp += np.square(vp)

        sum_eta += eta_diff
        sum_sf += sf
        sum_vp += vp

        sum_etasf += eta_diff * sf
        sum_etavp += eta_diff * vp
        sum_sfvp += sf * vp

    ## calculate values of correlations
    # mean values
    mn_eta, mn_sf, mn_vp = sum_eta / n, sum_sf / n, sum_vp / n
    # variances
    var_eta = (sum_sq_eta / n) - (np.square(mn_eta))
    var_sf = (sum_sq_sf / n) - (np.square(mn_sf))
    var_vp = (sum_sq_vp / n) - (np.square(mn_vp))
    # covariances
    cov_eta_sf = (sum_etasf / n) - (mn_eta * mn_sf)
    cov_eta_vp = (sum_etavp / n) - (mn_eta * mn_vp)
    cov_sf_vp = (sum_sfvp / n) - (mn_sf * mn_vp)

    # correlations
    corr_eta_sf = cov_eta_sf / np.sqrt(var_eta * var_sf)
    corr_eta_vp = cov_eta_vp / np.sqrt(var_eta * var_vp)
    corr_sf_vp = cov_sf_vp / np.sqrt(var_sf * var_vp)

    ## plot correlation matrix
    # full model fields
    plot_corr(corr_eta_sf, 'Elevation', 'SF increments', eta_x, eta_y)
    plot_corr(corr_eta_vp, 'Elevation', 'VP increments', eta_x, eta_y)
    plot_corr(corr_sf_vp, 'SF', 'VP increments', eta_x, eta_y)

def corr_cv(alpha, n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """

    dy, dx = param['dy'], param['dx']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant_surface.grid_V.nc"

    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    eta_x, eta_y = eta_lon[1:-2, 2:-1], eta_lat[1:-2, 2:-1]
    u_x, u_y = u_lon[1:-2, 1:-1], u_lat[1:-2, 1:-1]
    v_x, v_y = v_lon[1:-1, 2:-1], v_lat[1:-1, 2:-1]

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_sf, sum_sq_vp = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

    sum_eta, sum_sf, sum_vp = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

    sum_etasf, sum_etavp, sum_sfvp = np.zeros_like(eta_y), np.zeros_like(eta_y), np.zeros_like(eta_y)

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

        # remove boundaries
        eta_diff = eta_diff[1:-2, 2:-1]
        u_diff = u_diff[1:-2, 1:-1]
        v_diff = v_diff[1:-1, 2:-1]

        # find the control variables
        print(f'Finding control variables at {time_index} days.')
        eta_diff, sf_u, vp_u, du_mean, dv_mean = T_transform(eta_diff, u_diff, v_diff, dx, dy, u_lat, v_lat, alpha)

        # reduce the control arrays to be the same size as eta grid
        sf = sf_u[1:-1, 1:-1]
        vp = vp_u[1:-1, 1:-1]

        sum_sq_eta += np.square(eta_diff)
        sum_sq_sf += np.square(sf)
        sum_sq_vp += np.square(vp)

        sum_eta += eta_diff
        sum_sf += sf
        sum_vp += vp

        sum_etasf += eta_diff * sf
        sum_etavp += eta_diff * vp
        sum_sfvp += sf * vp

    ## calculate values of correlations
    # mean values
    mn_eta, mn_sf, mn_vp = sum_eta / n, sum_sf / n, sum_vp / n
    # variances
    var_eta = (sum_sq_eta / n) - (np.square(mn_eta))
    var_sf = (sum_sq_sf / n) - (np.square(mn_sf))
    var_vp = (sum_sq_vp / n) - (np.square(mn_vp))
    # covariances
    cov_eta_sf = (sum_etasf / n) - (mn_eta * mn_sf)
    cov_eta_vp = (sum_etavp / n) - (mn_eta * mn_vp)
    cov_sf_vp = (sum_sfvp / n) - (mn_sf * mn_vp)

    # correlations
    corr_eta_sf = cov_eta_sf / np.sqrt(var_eta * var_sf)
    corr_eta_vp = cov_eta_vp / np.sqrt(var_eta * var_vp)
    corr_sf_vp = cov_sf_vp / np.sqrt(var_sf * var_vp)

    ## plot correlation matrix
    # full model fields
    plot_corr(corr_eta_sf, 'Elevation', 'Unbal SF increments', eta_x, eta_y)
    plot_corr(corr_eta_vp, 'Elevation', 'Unbal VP increments', eta_x, eta_y)
    plot_corr(corr_sf_vp, 'UnbalancedSF', 'VP increments', eta_x, eta_y)

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

def corr_comparison(n):
    """
    Find the correlations of the model variable increments at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """

    # location of correlation plots
    lx, ux, ly, uy = 2, 50, 2, 50

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

    ny, nx = np.shape(eta_lat[ly:uy, lx:ux])[0], np.shape(eta_lon[ly:uy, lx:ux])[1]
    eta_u = np.empty((n, 2 * ny * nx))
    eta_v = np.empty((n, 2 * ny * nx))
    u_v = np.empty((n, 2 * ny * nx))

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

        ## flatten model arrays
        eta = eta[ly:uy, lx:ux].flatten()
        # print(np.shape(eta_diff))
        u= u[ly:uy, lx:ux].flatten()
        # print(np.shape(u_diff))
        v= v[ly:uy, lx:ux].flatten()
        # interp velocities to eta grid
        # u_diff, v_diff = interp_zonal(u_diff).flatten(), interp_merid(v_diff).flatten()

        # add to matrix to find correlations
        # print(np.shape(np.concatenate((eta_diff, u_diff))))
        eta_u[time_index, :] = np.concatenate((eta, u))
        eta_v[time_index, :] = np.concatenate((eta, v))
        u_v[time_index, :] = np.concatenate((u, v))

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
    plot_corr(corr_deta_du, 'MAtRIX Elevation', 'Zonal Velocity', eta_lon, eta_lat)
    plot_corr(corr_deta_dv, 'Elevation', 'Meridional Velocity', eta_lon, eta_lat)
    plot_corr(corr_du_dv, 'Zonal Velocity', 'Meridional Velocity', eta_lon, eta_lat)

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
    corr_eta_u1 = cov_eta_u / np.sqrt(var_eta * var_u)
    corr_eta_v1 = cov_eta_v / np.sqrt(var_eta * var_v)
    corr_u_v1 = cov_u_v / np.sqrt(var_u * var_v)

    ## plot correlation matrix
    # full model fields
    plot_corr(corr_eta_u1[ly:uy, lx:ux], 'Maths method Elevation', 'U', eta_lon, eta_lat)
    plot_corr(corr_eta_v1[ly:uy, lx:ux], 'Elevation', 'V', eta_lon, eta_lat)
    plot_corr(corr_u_v1[ly:uy, lx:ux], 'U', 'V', eta_lon, eta_lat)

    # plot method differences
    plot_corr(corr_eta_u1[ly:uy, lx:ux] - corr_deta_du, 'E', 'U', eta_lon, eta_lat)
    plot_corr(corr_eta_v1[ly:uy, lx:ux] - corr_deta_dv, 'E', 'V', eta_lon, eta_lat)
    plot_corr(corr_u_v1[ly:uy, lx:ux] - corr_du_dv, 'U', 'V', eta_lon, eta_lat)

if __name__ == '__main__':
        # Tikhonov's regularisation parameter
        alpha = 0
        # number of samples
        n = 150
        #corr_mv(n)
        #corr_mv_increments(n)
        #corr_mv_ub_increments(n)
        #corr_sf_vp(alpha, n)
        #corr_sf_vp_increments(alpha, n)
        #corr_cv(alpha, n)
        corr_comparison(n)