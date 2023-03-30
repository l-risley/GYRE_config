"""
Functions to just plot the full fields of elevation and the horizontal velocity vectors.
"""

from read_nemo_fields import *
from general_functions import *

def plots(time_index):
    """
    """
    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"
    #w_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_W_depth0.nc"

    # read eta, u, v and w from files
    eta = read_file(eta_input_file, "sossheig", time_index=time_index)
    u = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
    v = read_file(v_input_file, "vomecrty", time_index=time_index)[0]
    #w = read_file(w_input_file, "vomecrty", time_index=time_index)[0] #TODO: Check code in morning, WHERE IS THIS SAVED IN WORK DIRECTORY?

    # lon and lat for each grid
    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    print(f'Length of elevation lattitude : {np.shape(eta_lat)} and longitude : {np.shape(eta_lon)}.')
    print(f'Length of u lattitude : {np.shape(u_lat)} and longitude : {np.shape(u_lon)}.')
    print(f'Length of v lattitude : {np.shape(v_lat)} and longitude : {np.shape(v_lon)}.')

    contour(eta_lon, eta_lat, eta, f'at time {time[time_index]}', 'Elevation')
    contour(u_lon, u_lat, u, f'at time {time[time_index]}', 'Zonal Velocity')
    contour(v_lon, v_lat, v, f'at time {time[time_index]}', 'Meridional Velocity')

if __name__ == '__main__':
    # time to plot
    time_index = 0
    plots(time_index)

