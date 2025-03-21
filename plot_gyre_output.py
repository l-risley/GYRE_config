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

def calculate_rms_2d(field):
    # Convert the field to a numpy array for easier numerical operations
    field_array = np.array(field, dtype=np.float64)
    # Calculate the square of each element
    squared_field = np.square(field_array)
    # Calculate the mean of the squared values
    mean_squared = np.mean(squared_field)
    # Take the square root to get the RMS value
    rms_value = np.sqrt(mean_squared)
    return rms_value

def contour_gyre(x, y, z, variable_name: str):
    # 2D contour_gyre plot of one variable
    # switch coords from m to km
    plt.title(f'{variable_name}')# - {plot_of}')
    if variable_name == 'Elevation':
        units = '$m$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-0.5, vmax=1)
    elif variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')
    elif variable_name == 'Temperature':
        units = '$degrees$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')
    else:
        units = '$ms^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-0.6, vmax=0.6)
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.colorbar(label=f'{variable_name} ({units})')
    #plt.savefig(f'plots/{plot_of}{variable_name}.png')
    plt.show()

def plot_gyre12(date):
    """
    Plot a gyre12 output for a particular date.
    """

    # netcdf file locations
    gyre12_input_file = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/gyre12_{date}_restart.nc"

    # lon and lat for each grid
    gyre12_lon, gyre12_lat, time = read_file_info(gyre12_input_file)

    # gyre12 outputs
    gyre12_eta = read_file(gyre12_input_file, "sshn")
    gyre12_u = read_file(gyre12_input_file, "un")[0]
    gyre12_v = read_file(gyre12_input_file, "vn")[0]
    gyre12_t = read_file(gyre12_input_file, "tn")[0]

    print(gyre12_t[1, :100])
    plt.plot(np.arange(0,100),gyre12_t[1, :100])
    plt.show()
    # extract the year only
    year = int(str(date)[:4])
    actual_date = int(year) - 2000

    # plot gyre12
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_eta, f'Gyre12 {actual_date} yrs', 'Elevation')
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_u, f'Gyre12 {actual_date} yrs', 'Zonal Velocity')
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_v, f'Gyre12 {actual_date} yrs', 'Meridional Velocity')
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_t, f'Gyre12 {actual_date} yrs', 'Temperature')

def plot_gyre12_diff(date1, date2):
    """
    Plot the difference between two dates.
    """

    # netcdf file locations
    gyre12_input_file_1 = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/gyre12_{date1}_restart.nc"
    gyre12_input_file_2 = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/gyre12_{date2}_restart.nc"

    # lon and lat for each grid
    gyre12_lon, gyre12_lat, time = read_file_info(gyre12_input_file_1)

    # gyre12 outputs
    gyre12_1_eta = read_file(gyre12_input_file_1, "sshn")
    gyre12_1_u = read_file(gyre12_input_file_1, "un")[0]
    gyre12_1_v = read_file(gyre12_input_file_1, "vn")[0]

    # gyre12 outputs
    gyre12_2_eta = read_file(gyre12_input_file_2, "sshn")
    gyre12_2_u = read_file(gyre12_input_file_2, "un")[0]
    gyre12_2_v = read_file(gyre12_input_file_2, "vn")[0]

    # differences
    diff_eta = gyre12_1_eta - gyre12_2_eta
    diff_u = gyre12_1_u - gyre12_2_u
    diff_v = gyre12_1_v - gyre12_2_v

    # extract the year only
    year1 = int(str(date1)[:4])
    actual_date1 = int(year1) - 2000
    year2 = int(str(date2)[:4])
    actual_date2 = int(year2) - 2000

    # plot gyre12
    contour_gyre(gyre12_lon, gyre12_lat, diff_eta, f'Gyre12 diff from {actual_date2} to {actual_date1}', 'Elevation')
    contour_gyre(gyre12_lon, gyre12_lat, diff_u, f'Gyre12 diff from {actual_date2} to {actual_date1}', 'Zonal Velocity')
    contour_gyre(gyre12_lon, gyre12_lat, diff_v, f'Gyre12 diff from {actual_date2} to {actual_date1}', 'Meridional Velocity')

def plot_gyre36():
    """
    Plot a gyre36 output.
    """
    date = 30100201
    # netcdf file locations
    gyre36_input_file = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/gyre36_30100201_restart.nc"

    # lon and lat for each grid
    gyre36_lon, gyre36_lat, time = read_file_info(gyre36_input_file)

    # gyre36 outputs
    gyre36_eta = read_file(gyre36_input_file, "sshn")
    gyre36_u = read_file(gyre36_input_file, "un")[0]
    gyre36_v = read_file(gyre36_input_file, "vn")[0]
    gyre36_t = read_file(gyre36_input_file, "tn")[0]

    print(gyre36_t[1, :100])
    plt.plot(np.arange(0,100),gyre36_t[1, :100])
    plt.show()
    # extract the year only
    year = int(str(date)[:4])
    actual_date = int(year) - 2000

    # plot gyre36
    contour_gyre(gyre36_lon, gyre36_lat, gyre36_eta, f'Gyre36 {actual_date} yrs', 'Elevation')
    contour_gyre(gyre36_lon, gyre36_lat, gyre36_u, f'Gyre36 {actual_date} yrs', 'Zonal Velocity')
    contour_gyre(gyre36_lon, gyre36_lat, gyre36_v, f'Gyre36 {actual_date} yrs', 'Meridional Velocity')
    contour_gyre(gyre36_lon, gyre36_lat, gyre36_t, f'Gyre36 {actual_date} yrs', 'Temperature')

def contour_gyre_inv(x, y, z, plot_of: str, variable_name: str, exp:str):
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
    plt.ylabel('Lattitude ($^\circ$)')
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

    # gyre12 outputs
    gyre12_u = read_file(exp_input_file, "u")[0]
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

    # RMS of raw-error
    u_rms = calculate_rms_2d(u_err)
    v_rms = calculate_rms_2d(v_err)

    print(f'RMS of the zonal error = {u_rms}')
    print(f'RMS of the meridional error = {v_rms}')

    # plot gyre12
    #contour_gyre_inv(lon, lat, gyre12_u, 'Zonal Velocity', 'u', f'{exp}')
    #contour_gyre_inv(lon, lat, gyre12_v, 'Meridional Velocity', 'v', f'{exp}')
    contour_gyre_inv(lon, lat, gyre12_psi, 'Streamfunction', 'SF', f'{exp}')
    #contour_gyre_inv(lon, lat, gyre12_chi, 'Velocity Potential', 'VP', f'{exp}')
    contour_gyre_inv(lon, lat, gyre12_u_err, 'Zonal Velocity Relative Error', 'u_err', f'{exp}')
    contour_gyre_inv(lon, lat, gyre12_v_err, 'Meridional Velocity Relative Error', 'v_err', f'{exp}')

    contour_gyre_inv(lon, lat, u_err, 'Zonal Velocity Error', 'u', f'{exp}')
    contour_gyre_inv(lon, lat, v_err, 'Meridional Velocity Error', 'v', f'{exp}')

    #print(f'Min velocity increment, u = {np.min(gyre12_u)}, v = {np.min(gyre12_v)}')
    #print(f'Min abs velocity increment, u = {np.min(abs(gyre12_u))}, v = {np.min(abs(gyre12_v))}')
    #print(f'Max velocity increment, u = {np.max(gyre12_u)}, v = {np.max(gyre12_v)}')
    #print(f'Max abs velocity increment, u = {np.max(abs(gyre12_u))}, v = {np.max(abs(gyre12_v))}')
    #print(f'Min velocity reconstructed increment, u = {np.min(gyre12_u_inv)}, v = {np.min(gyre12_v_inv)}')
    #print(f'Min abs velocity reconstructed increment, u = {np.min(abs(gyre12_u_inv))}, v = {np.min(abs(gyre12_v_inv))}')
    #print(f'Max velocity reconstructed increment, u = {np.max(gyre12_u_inv)}, v = {np.max(gyre12_v_inv)}')
    #print(f'Max abs velocity reconstructed increment, u = {np.max(abs(gyre12_u_inv))}, v = {np.max(abs(gyre12_v_inv))}')

def plot_nature_run_gyre12():
    """
    Plot a gyre12 output for a particular date.
    """

    # netcdf file locations
    gyre12_input_file_T = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/lunchtime_seminar_24/20090801T0000Z_mersea.grid_T_gyre12.nc"
    gyre12_input_file_U = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/lunchtime_seminar_24/20090801T0000Z_mersea.grid_U_gyre12.nc"
    gyre12_input_file_V = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/lunchtime_seminar_24/20090801T0000Z_mersea.grid_V_gyre12.nc"

    # lon and lat for each grid
    gyre12_lon, gyre12_lat, time = read_file_info(gyre12_input_file_T)

    # gyre12 outputs
    gyre12_eta = read_file(gyre12_input_file_T, "sossheig")
    gyre12_u = read_file(gyre12_input_file_U, "vozocrtx")[0]
    gyre12_v = read_file(gyre12_input_file_V, "vomecrty")[0]
    gyre12_t = read_file(gyre12_input_file_T, "votemper")[0]
    gyre12_s = read_file(gyre12_input_file_T, "vosaline")[0]

    print(gyre12_t[1, :100])
    plt.plot(np.arange(0,100),gyre12_t[1, :100])
    plt.show()
    # extract the year only
    year = int(str(20090801)[:4])
    actual_date = int(year) - 2000

    # plot gyre12
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_eta, 'Elevation') #f'Gyre12 {actual_date} yrs', 'Elevation')
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_u, 'Zonal Velocity') #f'Gyre12 {actual_date} yrs', 'Zonal Velocity')
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_v, 'Meridional Velocity') #f'Gyre12 {actual_date} yrs', 'Meridional Velocity')
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_t, 'Temperature') # f'Gyre12 {actual_date} yrs', 'Temperature')
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_s, 'Salinity')  # f'Gyre12 {actual_date} yrs', 'Salinity')

def plot_vel_errors():
    """
    Plot velocity error files for inverse transformation.
    """
    # netcdf file locations
    u_input_file = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/vel_errors/vel_error_20090201T0000Z.grid_U.nc"
    v_input_file = f"/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/vel_errors/vel_error_20090201T0000Z.grid_V.nc"

    # lon and lat for each grid
    gyre12_lon, gyre12_lat, time = read_file_info(u_input_file)

    # gyre12 outputs
    gyre12_u = read_file(u_input_file, "vozocrtx")[0]
    gyre12_v = read_file(v_input_file, "vomecrty")[0]

    # plot gyre12
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_u, 'Zonal velocity increment')  # f'Gyre12 {actual_date} yrs', 'Zonal Velocity')
    contour_gyre(gyre12_lon, gyre12_lat, gyre12_v, 'Meridional velocity increment')  # f'Gyre12 {actual_date} yrs', 'Meridional Velocity')

def plot_gyre_div_vor_tests(exp):
    """
    Plot the output of the gyre inverse tests.
    Input:
    exp, which experiment number.
    Ouput:
    plots of the velocities and errors.
    """
    # netcdf file locations
    chi_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vor_div_chi_vel_to_psichi_to_vel_exp{exp}.nc"
    psi_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vor_div_psi_vel_to_psichi_to_vel_exp{exp}.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(chi_input_file)

    # gyre12 outputs
    chi_vor = read_file(chi_input_file, "chi_vor")[0]
    psi_div = read_file(psi_input_file, "psi_div")[0]
    print(psi_div)

    # plot gyre12
    plt.title(f'Divergence of the velocity generated by streamfunction')
    units = '$s^{-1}$'
    cmap = plt.cm.viridis
    newcolors = cmap(np.linspace(0, 1, cmap.N))
    # Set the middle color (which corresponds to 0) to white
    midpoint = int(cmap.N / 2)
    newcolors[midpoint] = np.array([1, 1, 1, 1])  # white
    # Create a new colormap
    new_cmap = mcolors.ListedColormap(newcolors)
    # Create a Normalize object
    #norm = mcolors.Normalize(vmin=np.min(psi_div), vmax=np.max(psi_div))
    # Set values close to zero (within the range -1e-11 to 1e-11) to zero
    threshold = 1e-19
    psi_div[np.abs(psi_div) < threshold] = 0  # Set small values to zero
    plt.pcolormesh(lon, lat, psi_div, cmap=new_cmap, shading='auto', vmin=-0.00006, vmax=0.00006)
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.colorbar(label=f'Divergence ({units})')
    plt.show()

    plt.title(f'Divergence of the velocity generated by streamfunction')
    units = '$s^{-1}$'
    cmap = plt.cm.viridis
    newcolors = cmap(np.linspace(0, 1, cmap.N))
    # Set the middle color (which corresponds to 0) to white
    midpoint = int(cmap.N / 2)
    newcolors[midpoint] = np.array([1, 1, 1, 1])  # white
    # Create a new colormap
    new_cmap = mcolors.ListedColormap(newcolors)
    # Create a Normalize object
    # norm = mcolors.Normalize(vmin=np.min(psi_div), vmax=np.max(psi_div))
    # Set values close to zero (within the range -1e-11 to 1e-11) to zero
    threshold = 1e-25
    psi_div[np.abs(psi_div) < threshold] = 0  # Set small values to zero
    plt.pcolormesh(lon[70:, 70:], lat[70:, 70:], psi_div[70:, 70:], cmap=new_cmap, shading='auto', vmin=-0.00006, vmax=0.00006)
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.colorbar(label=f'Divergence ({units})')
    plt.show()
    """
    plt.title(f'Vorticity of the velocity generated by velocity potential')
    units = '$m^{-1}$'
    plt.pcolormesh(lon, lat, chi_vor, cmap='viridis', shading='auto')  # , vmin=-1, vmax=1)
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.colorbar(label=f'Vorticity ({units})')
    plt.show()

    plt.title(f'Vorticity of the velocity generated by velocity potential')
    units = '$m^{-1}$'
    plt.pcolormesh(lon[:30, :30], lat[:30, :30], chi_vor[:30, :30], cmap='viridis',
                   shading='auto')  # , vmin=-1, vmax=1)
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.colorbar(label=f'Vorticity ({units})')
    plt.show()

    #print(f'Min velocity increment, u = {np.min(gyre12_u)}, v = {np.min(gyre12_v)}')
    """
if __name__ == '__main__':
    #plot_gyre12(30100101)
    #plot_gyre12_diff(30100101, 30020101)
    #plot_gyre36()
    plot_gyre_inverse_tests('35')
    #plot_nature_run_gyre12()
    #plot_vel_errors()
    #plot_gyre_div_vor_tests('7')
