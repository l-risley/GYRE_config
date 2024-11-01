import numpy as np
from general_functions import *
from read_nemo_fields import *

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

def divergence(u, v, dy: int, dx: int):
    # Take the divergence of u or v at index j+1 and j
    # size will be (ny, nx) -  both are on the eta_grid
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

def vorticity(u, v, dy: int, dx: int, ny: int, nx: int):
    # Take the vorticity of u or v at index j+1 and j
    # size will be (ny, nx) -  both are on the eta_grid
    dvdx = np.zeros((ny + 1, nx + 1))
    dudy = np.zeros((ny+1, nx + 1))
    dvdx[:, 1:-1] = dzdx(v, dx)
    dudy[1:-1, :] = dzdy(u, dy)
    dvdx = interp_zonal(interp_merid(dvdx))
    dudy = interp_zonal(interp_merid(dudy))
    return dvdx - dudy

def main(time_index):
    """
    Plot the divergence and vorticity of the gyre configuration output.
    """

    dy, dx = 106000/12, 106000/12

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

    # need u to be size (ny,nx+1)
    u_diff_new = u_diff[1:-2, 1:-1]

    v_0 = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
    v_1 = read_file(v_input_file, "vomecrty", time_index=time_index + 1)[0]
    v_diff = v_1 - v_0

    # need v to be size (ny+1, nx)
    v_diff_new = v_diff[1:-1, 1:-2]

    # lon and lat for each grid
    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)
    div_lon, div_lat = eta_lon[2:-1, 2:-1], eta_lat[2:-1, 2:-1]

    ny, nx = np.shape(div_lat)

    # find divergence and vorticity
    div = divergence(u_diff_new, v_diff_new, dy, dx)
    vor = vorticity(u_diff_new, v_diff_new, dy, dx, ny, nx)

    # plot everything
    contour(eta_lon, eta_lat, eta_0, f'Gyre12', 'Elevation')
    contour(u_lon, u_lat, u_diff, f'Gyre12', 'U')
    contour(v_lon, v_lat, v_diff, f'Gyre12', 'V')
    contour(div_lon, div_lat, div, f'Gyre12', 'Divergence')
    contour(div_lon, div_lat, vor, f'Gyre12', 'Vorticity')

def sf_x_derivative(sf, dx, ny, nx):
    # x-derivative of streamfunction
    v_sf = np.zeros((ny + 1, nx))
    v_sf[1:-1, 0] = 0.5 * 1 / dx * (-sf[:-1, 0] - sf[1:, 0] + sf[:-1, 1] + sf[1:, 1])
    v_sf[1:-1, -1] = 0.5 * 1 / dx * (-sf[:-1, -2] - sf[1:, -2] + sf[:-1, -1] + sf[1:, -1])
    v_sf[1:-1, 1:-1] = 0.25 * 1 / dx * (-sf[:-1, :-2] - sf[1:, :-2] + sf[:-1, 2:] + sf[1:, 2:])
    v_sf[:, [0, -1]] = 0
    return v_sf

def sf_y_derivative(sf, dy, ny, nx):
    # y-derivative of streamfunction
    u_sf = np.zeros((ny, nx + 1))
    u_sf[0, 1:-1] = 0.5 * 1 / dy * (-sf[0, :-1] - sf[0, 1:] + sf[1, :-1] + sf[1, 1:])
    u_sf[-1, 1:-1] = 0.5 * 1 / dy * (-sf[-2, :-1] - sf[-2, 1:] + sf[-1, :-1] + sf[-1, 1:])
    u_sf[1:-1, 1:-1] = 0.25 * 1 / dy * (-sf[:-2, :-1] - sf[:-2, 1:] + sf[2:, :-1] + sf[2:, 1:])
    u_sf[[0, -1], :] = 0
    return -u_sf

def v_psi_div():
    """
    Plot the divergence and vorticity of the gyre configuration output.
    """

    dy, dx = 106000/12, 106000/12

    # netcdf file locations
    gyre12_input_file_U = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/vel_errors/vel_error_20090201T0000Z.grid_U.nc"
    gyre12_input_file_V = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/vel_errors/vel_error_20090201T0000Z.grid_V.nc"
    gyre12_input_file_psi = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp2.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(gyre12_input_file_U)

    # gyre12 outputs
    u = read_file(gyre12_input_file_U, "vozocrtx")[0]
    v = read_file(gyre12_input_file_V, "vomecrty")[0]
    psi = read_file(gyre12_input_file_psi, "psi")[0]

    ny_u, nx_u = np.shape(u)
    ny, nx = ny_u, nx_u

    sf_dx = sf_x_derivative(psi, dx, ny, nx)
    sf_dy = sf_y_derivative(psi, dy, ny, nx)


    # lon and lat for each grid
    #div_lon, div_lat = lon[2:-1, 2:-1], lat[2:-1, 2:-1]

    # find divergence of the veloicty generated by streamfunction
    div = divergence(sf_dy, sf_dx, dy, dx)

    # plot everything
    contour(lon[:100, :100], lat[:100, :100], div[:100, :100], f'Gyre12', 'Divergence')

if __name__ == '__main__':
    # time to plot
    time_index = 700
    #main(time_index)
    v_psi_div()