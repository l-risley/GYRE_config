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

def contour_gyre_x_z(x, y, z, plot_of: str, variable_name: str, exp:str):
    # 2D contour_gyre plot of one variable
    # switch coords from m to km
    plt.title(f'{plot_of}') #(- Experiment {exp}')
    if variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-15000, vmax=15000)
    elif variable_name == 'u_err' or variable_name == 'v_err':
        units = None
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-1, vmax=1)
    else:
        units = '$ms^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-0.06, vmax=0.06)
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Depth ($^\circ$)')
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
    print(np.shape(lon))

    # gyre12 outputs
    gyre12_u = read_file(exp_input_file, "u")
    gyre12_v = read_file(exp_input_file, "v")[0]
    gyre12_psi = read_file(exp_input_file, "psi")[0]
    gyre12_chi = read_file(exp_input_file, "chi")[0]
    gyre12_u_inv = read_file(exp_input_file, "u_inv")[0]
    gyre12_v_inv = read_file(exp_input_file, "v_inv")[0]
    gyre12_u_err = read_file(exp_input_file, "u_rel_err")[0]
    gyre12_v_err = read_file(exp_input_file, "v_rel_err")[0]

    # calculate the raw error
    u_err = gyre12_u - gyre12_u_inv
    v_err = gyre12_v - gyre12_v_inv

    print(np.shape(gyre12_u))
    print(gyre12_u[:, 1, 1])

    v_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/stats_u-di338_vomecrty.nc"
    fileid = netCDF4.Dataset(v_input_file, mode='r')
    depth = fileid.variables['depthv'][:]
    fileid.close()
    print(depth)
    np.savetxt('/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/gyre_depths.txt', depth, delimiter=',')
if __name__ == '__main__':
    plot_gyre_inverse_tests('39')
