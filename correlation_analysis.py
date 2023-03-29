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
    ny, nx = np.shape(lat)[0], np.shape(lon)[1]
    eta_u = np.empty((n, 2*ny*nx))
    eta_v = np.empty((n, 2*ny*nx))
    u_v = np.empty((n, 2*ny*nx))

    for time_index in range(n): #range(num_times - 1):
        # find increments
        eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)
        eta_1 = read_file(eta_input_file, "sossheig", time_index=time_index + 1)
        eta_diff = eta_1 - eta_0

        u_0 = read_file(u_input_file, "vozocrtx", time_index=time_index)
        u_1 = read_file(u_input_file, "vozocrtx", time_index=time_index + 1)
        u_diff = u_1 - u_0

        v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)
        v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)
        v_diff = v_1 - v_0

        ## flatten model arrays
        eta_diff = eta_diff.flatten()
        u_diff = u_diff.flatten()
        v_diff = v_diff.flatten()
        # interp velocities to eta grid
        #u_diff, v_diff = interp_zonal(u_diff).flatten(), interp_merid(v_diff).flatten()

        # add to matrix to find correlations
        eta_u[time_index, :] = np.concatenate((eta_diff, u_diff))
        eta_v[time_index, :] = np.concatenate((eta_diff, v_diff))
        u_v[time_index, :] = np.concatenate((u_diff, v_diff))

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
    plot_corr(corr_deta_du, 'Elevation', 'Zonal Velocity')
    plot_corr(corr_deta_dv, 'Elevation', 'Meridional Velocity')
    plot_corr(corr_du_dv, 'Zonal Velocity', 'Meridional Velocity')

def corr_cv_at_grid_points(alpha, n):
    """
    Find the correlations of the control variable increments at each grid point. Produce correlation matrix plots.
    Inputs: alpha, Tikhonov's regularisation parameter
            n, number of time intervals to store perturbations
    """
    # dimensions
    dx, dy = param['dx'], ['dy']

    eta_input_file = "/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T.nc"
    u_input_file = "/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"
    v_input_file = "/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant_surface.grid_U.nc"

    lon, lat, time = read_file_info(input_file)
    num_times = np.size(time)

    ## 2d arrays for correlation analysis
    # size of eta
    ny, nx = len(lat), len(lon)
    eta_sf = np.empty((n, 2 * ny * nx))
    eta_vp = np.empty((n, 2 * ny * nx))
    sf_vp = np.empty((n, 2 * ny * nx))

    for time_index in range(n):  # range(num_times - 1):
        # find increments
        eta_0 = read_file(eta_input_file, "sossheig", time_index=time_index)
        eta_1 = read_file(eta_input_file, "sossheig", time_index=time_index + 1)
        eta_diff = eta_1 - eta_0

        u_0 = read_file(u_input_file, "", time_index=time_index)
        u_1 = read_file(u_input_file, "", time_index=time_index + 1)
        u_diff = u_1 - u_0

        v_0 = read_file(v_input_file, "", time_index=time_index)
        v_1 = read_file(v_input_file, "", time_index=time_index + 1)
        v_diff = v_1 - v_0

        # find the control variables
        print(f'Finding control variables at {time_index} days.')
        eta_diff, sf_u, vp_u, du_mean, dv_mean = T_transform(eta_diff, u_diff, v_diff, dx, dy, lat, alpha)

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
    corr_deta_dsf = find_corr_matrix(corr_deta_dsf_full)
    corr_deta_dvp = find_corr_matrix(corr_deta_dvp_full)
    corr_dsf_dvp = find_corr_matrix(corr_dsf_dvp_full)

    ## plot correlation matrix
    # control increments
    plot_corr(corr_deta_dsf, 'Elevation', 'Streamfunction', lon, lat)
    plot_corr(corr_deta_dvp, 'Elevation', 'Velocity Potential', lon, lat)
    plot_corr(corr_dsf_dvp, 'Streamfunction', 'Velocity Potential', lon, lat)

if __name__ == '__main__':
        # Tikhonov's regularisation parameter
        alpha = 0
        # number of samples
        n = 199
        corr_mv_at_grid_points(n)
        #corr_cv_at_grid_points(alpha, n)
