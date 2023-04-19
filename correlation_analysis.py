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
    plt.savefig(f'plots/corr{variable_1}{variable_2}.png')
    plt.show()

def corr_mv(n):
    """
    Find the correlations of the full model variables at each grid point. Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations.
    """
    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config//instant.grid_V_depth0.nc"

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
    Inputs: n, number of time intervals to store perturbations.
    """
    # netcdf file locations
    input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/output_file_combined.nc"

    lon, lat, time = read_file_info(input_file)

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_u, sum_sq_v = np.zeros_like(lat), np.zeros_like(lat), np.zeros_like(lat)
    sum_eta, sum_u, sum_v = np.zeros_like(lat), np.zeros_like(lat), np.zeros_like(lat)
    sum_etau, sum_etav, sum_uv = np.zeros_like(lat), np.zeros_like(lat), np.zeros_like(lat)

    for time_index in range(n):  # range(num_times - 1):
        print(f'Increment at time {time_index}.')
        # find increments
        eta_diff = read_file(input_file, "sossheig", time_index=time_index)
        u_diff = read_file(input_file, "vozocrtx", time_index=time_index)
        v_diff = read_file(input_file, "vomecrty", time_index=time_index)

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
    plot_corr(corr_eta_u, 'Elevation', 'Zonal Velocity increments', lon, lat)
    plot_corr(corr_eta_v, 'Elevation', 'Meridional Velocity increments', lon, lat)
    plot_corr(corr_u_v, 'Zonal', 'Meridional Velocity increments', lon, lat)

def corr_mv_b_increments(n):
    """
    Find the correlations of the model variable increments (for unbalanced velocities) at each grid point.
    Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """
    dy, dx = param['dy'], param['dx']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

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

        # interp velocities to eta grid
        u_diff, v_diff = interp_zonal(u_diff_b), interp_merid(v_diff_b)

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
    plot_corr(corr_eta_u[1:-1, 1:-1], 'Elevation', 'Balanced U increments', eta_x[1:-1, 1:-1], eta_y[1:-1, 1:-1])
    plot_corr(corr_eta_v[1:-1, 1:-1], 'Elevation', 'Balanced V increments', eta_x[1:-1, 1:-1], eta_y[1:-1, 1:-1])
    plot_corr(corr_u_v[1:-1, 1:-1], 'Balanced', 'velocity increments', eta_x[1:-1, 1:-1], eta_y[1:-1, 1:-1])

def corr_mv_ub_increments(n):
    """
    Find the correlations of the model variable increments (for unbalanced velocities) at each grid point.
    Produce correlation matrix plots.
    Inputs: n, number of time intervals to store perturbations
    """
    dy, dx = param['dy'], param['dx']

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

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
    plot_corr(corr_eta_u[1:-1, 1:-1], 'Elevation', 'Unbalanced U increments', eta_x[1:-1, 1:-1], eta_y[1:-1, 1:-1])
    plot_corr(corr_eta_v[1:-1, 1:-1], 'Elevation', 'Unbalanced V increments', eta_x[1:-1, 1:-1], eta_y[1:-1, 1:-1])
    plot_corr(corr_u_v[1:-1, 1:-1], 'Unbalanced', 'velocity increments', eta_x[1:-1, 1:-1], eta_y[1:-1, 1:-1])

def corr_sf_vp(n):
    """
    Find the correlations of the control variable increments at each grid point. Produce correlation matrix plots.
    control variables have been run on monsoon and outputs stored.
    Inputs: n, number of time intervals to store perturbations
    """

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/output_sf_vp_combined.nc"

    eta_lon, eta_lat, time = read_file_info(input_file)

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_sf, sum_sq_vp = np.zeros_like(eta_lon), np.zeros_like(eta_lon), np.zeros_like(eta_lon)
    sum_eta, sum_sf, sum_vp = np.zeros_like(eta_lon), np.zeros_like(eta_lon), np.zeros_like(eta_lon)
    sum_etasf, sum_etavp, sum_sfvp = np.zeros_like(eta_lon), np.zeros_like(eta_lon), np.zeros_like(eta_lon)

    for time_index in range(n):
        print(f'Time {time_index}.')

        eta = read_file(eta_input_file, "sossheig", time_index=time_index)[1:-2, 2:-1]
        sf = read_file(input_file, "sf_arr", time_index=time_index)
        vp = read_file(input_file, "vp_arr", time_index=time_index)

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
    plot_corr(corr_eta_sf[1:-1, 1:-1], 'Elevation', 'SF', eta_lon[1:-1, 1:-1], eta_lat[1:-1, 1:-1])
    plot_corr(corr_eta_vp[1:-1, 1:-1], 'Elevation', 'VP', eta_lon[1:-1, 1:-1], eta_lat[1:-1, 1:-1])
    plot_corr(corr_sf_vp[1:-1, 1:-1], 'SF', 'VP', eta_lon[1:-1, 1:-1], eta_lat[1:-1, 1:-1])

def corr_sf_vp_increments(n):
    """
    Find the correlations of the control variable increments at each grid point. Produce correlation matrix plots.
    control variables have been run on monsoon and outputs stored.
    Inputs: n, number of time intervals to store perturbations
    """

    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/output_file_combined.nc"
    input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/output_diff_sf_vp_combined.nc"

    eta_lon, eta_lat, time = read_file_info(input_file)

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_sf, sum_sq_vp = np.zeros_like(eta_lon), np.zeros_like(eta_lon), np.zeros_like(eta_lon)
    sum_eta, sum_sf, sum_vp = np.zeros_like(eta_lon), np.zeros_like(eta_lon), np.zeros_like(eta_lon)
    sum_etasf, sum_etavp, sum_sfvp = np.zeros_like(eta_lon), np.zeros_like(eta_lon), np.zeros_like(eta_lon)

    for time_index in range(n): #range(num_times - 1):
        print(f'Time {time_index}.')

        eta_diff = read_file(eta_input_file, "sossheig", time_index=time_index)
        sf = read_file(input_file, "sfd_arr", time_index=time_index)
        vp = read_file(input_file, "vpd_arr", time_index=time_index)

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
    plot_corr(corr_eta_sf[1:-1, 1:-1], 'Elevation', 'SF increments', eta_lon[1:-1, 1:-1], eta_lat[1:-1, 1:-1])
    plot_corr(corr_eta_vp[1:-1, 1:-1], 'Elevation', 'VP increments', eta_lon[1:-1, 1:-1], eta_lat[1:-1, 1:-1])
    plot_corr(corr_sf_vp[1:-1, 1:-1], 'SF', 'VP increments', eta_lon[1:-1, 1:-1], eta_lat[1:-1, 1:-1])

def corr_cv(n):
    """
    Find the correlations of the control variable increments at each grid point. Produce correlation matrix plots.
    control variables have been run on monsoon and outputs stored.
    Inputs: n, number of time intervals to store perturbations
    """

    # netcdf file locations
    input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/output_file_combined.nc"

    eta_lon, eta_lat, time = read_file_info(input_file)

    # initialise arrays for correlations calculations
    sum_sq_eta, sum_sq_sf, sum_sq_vp = np.zeros_like(eta_lon), np.zeros_like(eta_lon), np.zeros_like(eta_lon)
    sum_eta, sum_sf, sum_vp = np.zeros_like(eta_lon), np.zeros_like(eta_lon), np.zeros_like(eta_lon)
    sum_etasf, sum_etavp, sum_sfvp = np.zeros_like(eta_lon), np.zeros_like(eta_lon), np.zeros_like(eta_lon)

    for time_index in range(n):  # range(num_times - 1):
        print(f'Time {time_index}.')

        eta_diff = read_file(input_file, "sossheig", time_index=time_index)
        sf = read_file(input_file, "streamfunction", time_index=time_index)
        vp = read_file(input_file, "velocity_potential", time_index=time_index)

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
    plot_corr(corr_eta_sf[1:-1, 1:-1], 'Elevation', 'Unbal SF increments', eta_lon[1:-1, 1:-1], eta_lat[1:-1, 1:-1])
    plot_corr(corr_eta_vp[1:-1, 1:-1], 'Elevation', 'Unbal VP increments', eta_lon[1:-1, 1:-1], eta_lat[1:-1, 1:-1])
    plot_corr(corr_sf_vp[1:-1, 1:-1], 'Unbalanced SF', 'VP increments', eta_lon[1:-1, 1:-1], eta_lat[1:-1, 1:-1])

if __name__ == '__main__':
        # Tikhonov's regularisation parameter, alpha = 0
        # number of samples
        n = 729
        corr_mv(n)
        corr_mv_increments(n)
        corr_mv_b_increments(n)
        corr_mv_ub_increments(n)
        corr_sf_vp(n)
        corr_sf_vp_increments(n)
        corr_cv(n)