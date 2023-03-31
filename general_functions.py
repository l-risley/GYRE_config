"""
General functions needed for the transforms and correlation analysis.

"""

import matplotlib.pyplot as plt
import numpy as np
import os

def dzdx(z, dx: int):
    # Take the difference between u at index j+1 and j
    # size will be (ny, nx) on the eta_grid if dudx
    # size will be (ny+1, nx-1) not on any grid if dvdx
    # eta is interp to the u_grid
    dz = np.diff(z, axis=1)
    return dz / dx


def dzdy(z, dy: int):
    # Take the difference between z at index j+1 and j
    # size will be (ny, nx) on the eta_grid id dvdy
    # size will be (ny-1, nx+1) not on any grid if dudy
    # eta is interp to the v_grid
    dz = np.diff(z, axis=0)
    return dz / dy

def divergence(u, v, dx: int, dy: int):
    # Take the difference between u or v at index j+1 and j
    # size will be (ny, nx) -  both are on the eta_gird
    dudx = dzdx(u, dx)
    dvdy = dzdy(v, dy)
    return dudx + dvdy


def interp_zonal(z):
    # interpolate in the zonal direction
    # interpolate u to eta or eta to u
    return 0.5 * (z[:, :-1] + z[:, 1:])


def interp_merid(z):
    # interpolate in the meridional direction
    # interpolate v to eta or eta to v
    return 0.5 * (z[:-1, :] + z[1:, :])

def outside_boundary(z: np.ndarray, location: str):
    # TODO: be able to make this more efficient (use hstack etc)
    """
    Add zeros outside the boundary of z, as eta, u and v are all zero on land.
    Inputs: z, either eta, u or v
            location, either ew (east, west), ns (north, south) or nesw (north, east, south, west)
    Output: z_new, either one column of zeros is added either side of the matrix, or one row either end or both.
    """
    if location == 'ew':
        return np.c_[np.zeros(np.shape(z)[0]), z, np.zeros(np.shape(z)[0])]
    elif location == 'ns':
        return np.r_[[np.zeros(np.shape(z)[1])], z, [np.zeros(np.shape(z)[1])]]
    elif location == 'nesw':
        z_new = np.c_[np.zeros(np.shape(z)[0]), z, np.zeros(np.shape(z)[0])]
        return np.r_[[np.zeros(np.shape(z_new)[1])], z_new, [np.zeros(np.shape(z_new)[1])]]


def contour(x, y, z, plot_of: str, variable_name: str):
    # 2D contour plot of one variable
    # switch coords from m to km
    plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')
    # ax = sns.heatmap(z, cmap = 'ocean')
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.title(f'{variable_name} - {plot_of}')
    if variable_name == 'Elevation':
        units = '$m$'
    elif variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
    else:
        units = '$ms^{-1}$'
    plt.colorbar(label=f'{variable_name} ({units})')
    plt.savefig(f'plots/{plot_of}{variable_name}.png')
    plt.show()

def plot_one_convergence(x_it, plot_of):
    """
    Plot convergence of the cost function/gradient norm at a certain cycle during the assimilation routine.
    Inputs: - x_it, values of the cost function at each iteration
            - plot_of, cost function or gradient
    """
    number_it = len(x_it)
    iterations = np.arange(0, number_it)
    plt.plot(iterations, x_it)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel(f'{plot_of}')
    plt.title(f'Convergence of {plot_of}')
    plt.savefig(f'sweConvergence{plot_of}.png')
    plt.show()

def contour_err(x, y, z, plot_of: str, variable_name: str):
    # 2D contour plot of one variable
    # switch coords from m to km
    x, y = x / 1000, y / 1000
    plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')
    # ax = sns.heatmap(z, cmap = 'ocean')
    plt.xlabel('Longitude (km)')
    plt.ylabel('Lattitude (km)')
    plt.title(f'{plot_of} - {variable_name}')
    if variable_name == 'Elevation':
        units = '$m$'
    elif variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
    else:
        units = '$ms^{-1}$'

    plt.colorbar(extend='max', label=f'{variable_name} ({units})')
    plt.clim(0, 20)
    plt.savefig(f'SI_error_{variable_name}.png')
    plt.show()