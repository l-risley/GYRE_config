""""""
# from transforms.balance import *
import numpy as np
import numpy.ma as ma
from general_functions import *
from gyre_setup import *
from read_nemo_fields import *


def contour_balance(x, y, z, variable_name: str):
    # 2D contour plot of one variable
    # switch coords from m to km
    plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')  # , vmin=-2.5, vmax=1.5)
    # ax = sns.heatmap(z, cmap = 'ocean')
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.title(f'{variable_name}')
    if variable_name == 'Elevation':
        units = '$m$'
    elif variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
    else:
        units = '$ms^{-1}$'
    plt.colorbar(label=f'{variable_name}')
    plt.savefig(f'plots/{variable_name}.png')
    plt.show()


def geostrophic_balance_test(eta: np.ndarray, vel: str, lat, lon, vel_lat, vel_lon, dz, mask):
    """
    Use geostrophic balance to find the horizontal velocity vectors.
    Inputs: eta, elevation
            vel, either 'u' or 'v'
            lat, lattitude coordinates
            dz, spatial grid length (either dy or dx)
    Output: u/v, the horizontal velocity vector
    """
    print('Calculating geostrophic balance.')
    contour_balance(lon, lat, eta, 'Elevation')
    f_0, g, beta = f0_calc(), param['g'], beta_calc()
    print(f'The coriolis parameters:')
    print(f'The value of f_0 = {f_0} and the value of beta = {beta}.')
    print(f'Gravitation acceleration = {g}.')

    if vel == 'u':
        dy = dz
        print(f'dy = {dy}.')
        # find dndy on the v grid
        dndy = dzdy(eta, dy)
        contour_balance(vel_lon[1:, :], vel_lat[1:, :], dndy, 'd_eta dy')
        ## interp to the u grid using four point averaging
        # u will have the same shape as eta, with zeros on the boundary
        dndy_ugrid = np.zeros_like(eta)
        dndy_ugrid[1:-1, :-1] = interp_merid(interp_zonal(dndy))
        contour_balance(vel_lon, vel_lat, dndy_ugrid, 'd_eta dy on u grid')
        ## calculate the coriolis term on the u grid
        # f = f0 + by where y is the distance between the most southern point (where f) is calculated).
        cor = f_0 + beta * ((vel_lat - param['phi0']) * param['rad'])
        print(f'Coriolis  = {cor}.')
        contour_balance(vel_lon, vel_lat, cor, 'coriolis on v grid')
        u_bal = - g * np.divide(dndy_ugrid, cor)
        return ma.array(u_bal, mask=mask)

    elif vel == 'v':
        dx = dz
        print(f'dx = {dx}.')
        # find dndx on the u grid
        dndx = dzdx(eta, dx)
        contour_balance(vel_lon[:, 1:], vel_lat[:, 1:], dndx, 'd_eta dx')
        ## interp to the v grid using four point averaging
        # v will have the same shape as eta, with zeros on the boundary
        dndx_vgrid = np.zeros_like(eta)
        dndx_vgrid[:-1, 1:-1] = interp_zonal(interp_merid(dndx))
        contour_balance(vel_lon, vel_lat, dndx_vgrid, 'd_eta dx on v grid')
        ## calculate the coriolis term on the vgrid
        # f = f0 + by where y is the distance between the most southern point (where f) is calculated).
        cor = f_0 + beta * ((vel_lat - param['phi0']) * param['rad'])
        print(f'Coriolis  = {cor}.')
        contour_balance(vel_lon, vel_lat, cor, 'coriolis on u grid')
        v_bal = g * np.divide(dndx_vgrid, cor)
        return ma.array(v_bal, mask=mask)


if __name__ == '__main__':
    time_index = 100
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

    eta_mask, u_mask, v_mask = eta_0.mask, u_0.mask, v_0.mask
    print(f'eta mask: {eta_mask}')
    print(f'u mask: {u_mask}')
    print(f'v mask: {v_mask}')

    # lon and lat for each grid
    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)
    geostrophic_balance_test(eta_0, 'u', eta_lat, eta_lon, u_lat, u_lon, dy, u_mask)
    geostrophic_balance_test(eta_0, 'v', eta_lat, eta_lon, v_lat, v_lon, dx, v_mask)
