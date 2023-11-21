"""
PLot outputs from the gyre12 config.
"""
import numpy as np
import numpy.ma as ma
import netCDF4
from scipy.interpolate import griddata
from read_nemo_fields import *
from general_functions import *

def contour_gyre(x, y, z, plot_of: str, variable_name: str):
    # 2D contour_gyre plot of one variable
    # switch coords from m to km
    plt.title(f'{variable_name} - {plot_of}')
    if variable_name == 'Elevation':
        units = '$m$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-1, vmax=1.5)
    elif variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')
    elif variable_name == 'Temperature':
        units = '$degrees$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')
    else:
        units = '$ms^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-2, vmax=2)
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.colorbar(label=f'{variable_name} ({units})')
    plt.savefig(f'plots/{plot_of}{variable_name}.png')
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

if __name__ == '__main__':
    plot_gyre12(30100101)
    #plot_gyre12_diff(30100101, 30020101)
