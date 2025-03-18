"""
PLot outputs from the gyre12 config.
"""
import numpy as np
import numpy.ma as ma
import netCDF4
from scipy.interpolate import griddata
from read_nemo_fields import *
from general_functions import *
import matplotlib.colors as mcolors

def contour_gyre_x_z(x, y, z, plot_of: str, variable_name: str):
    # 2D contour_gyre plot of one variable
    # switch coords from m to km
    plt.title(f'{plot_of}') #(- Experiment {exp}')
    if variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')#, vmin=-15000, vmax=15000)
    elif variable_name == 'u_err' or variable_name == 'v_err':
        units = '$ms^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-1, vmax=1)
    else:
        units = '$ms^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-0.06, vmax=0.06)
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Depth (m)')
    plt.gca().invert_yaxis()
    plt.colorbar(label=f'{plot_of} ({units})')
    #plt.savefig(f'f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/{variable_name}_{exp}.png')
    plt.show()


def plot_gyre_inverse_tests(exp):
    """
    Plot the output of the gyre inverse tests.
    Input:
    exp, which experiment number.
    Ouput:
    plots of the velocities and errors.
    """
    # netcdf file locations
    exp_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp}.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(exp_input_file)
    ny, nx = np.shape(lon)

    line = np.int(np.floor(8 * ny / 9))
    # gyre12 outputs
    u = read_file(exp_input_file, "u")[:, :,  line]
    v = read_file(exp_input_file, "v")[:, :,  line]
    psi = read_file(exp_input_file, "psi")[:, :,  line]
    chi = read_file(exp_input_file, "chi")[:, :,  line]
    u_recon = read_file(exp_input_file, "u_inv")[:, :,  line]
    v_recon = read_file(exp_input_file, "v_inv")[:, :,  line]
    u_rel_err = read_file(exp_input_file, "u_rel_err")[:, :,  line]
    v_rel_err = read_file(exp_input_file, "v_rel_err")[:, :,  line]

    print(np.shape(psi))
    # calculate the raw error
    u_err = u - u_recon
    v_err = v - v_recon

    depths = np.loadtxt(f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/gyre_depths.txt',delimiter=",")

    #contour_gyre_x_z(lat[:, line], depths, psi, 'Streamfunction', 'SF')
    #contour_gyre_x_z(lat[:, line], depths, u_err, 'Zonal velocity reconstruction error', 'u_err')
    #contour_gyre_x_z(lat[:, line], depths, v_err, 'Meridional velocity reconstruction error', 'v_err')

    contour_gyre_x_z(lat[:, line], depths, u_rel_err, 'Zonal velocity relative error', 'u_err')
    contour_gyre_x_z(lat[:, line], depths, v_rel_err, 'Meridional velocity relative error', 'v_err')


if __name__ == '__main__':
    plot_gyre_inverse_tests('39')
