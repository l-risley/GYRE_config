"""
Functions to just plot the full fields of elevation and the horizontal velocity vectors.
"""

from read_nemo_fields import *
from general_functions import *

def one_plot(time_index):
    """
    """
    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

    # read eta, u and v from files
    eta = read_file(eta_input_file, "sossheig", time_index=time_index)
    u = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
    v = read_file(v_input_file, "vomecrty", time_index=time_index)[0]

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

def plots_full():
    """
    """
    # netcdf file locations
    eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

    # lon and lat for each grid
    eta_lon, eta_lat, time = read_file_info(eta_input_file)
    u_lon, u_lat, time = read_file_info(u_input_file)
    v_lon, v_lat, time = read_file_info(v_input_file)

    print(f'Length of elevation lattitude : {np.shape(eta_lat)} and longitude : {np.shape(eta_lon)}.')
    print(f'Length of u lattitude : {np.shape(u_lat)} and longitude : {np.shape(u_lon)}.')
    print(f'Length of v lattitude : {np.shape(v_lat)} and longitude : {np.shape(v_lon)}.')

    num_times = np.size(time)
    print(num_times)

    for time_index in range(num_times - 1):
        print(f'Plots at {time_index}')
        #eta = read_file(eta_input_file, "sossheig", time_index=time_index)
        #u = read_file(u_input_file, "vozocrtx", time_index=time_index)[0]
        v = read_file(v_input_file, "vomecrty", time_index=time_index)[0]

        #contour(eta_lon, eta_lat, eta, f'time_{time[time_index]}', 'Elevation')
        #contour(u_lon, u_lat, u, f'time_{time[time_index]}', 'Zonal Velocity')
        contour(v_lon, v_lat, v, f'time_{time[time_index]}', 'Meridional Velocity')

def plots_1000yrs():
    """
    """
    # netcdf file locations
    input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/model_output_1000yrs.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(input_file)

    print(f'Lattitude : {np.shape(lat)} and longitude : {np.shape(lon)}.')

    num_times = np.size(time)
    print(num_times)
    """
    for time_index in range(num_times - 1):
        print(f'Plots at {time_index}')
        eta = read_file(input_file, "sossheig", time_index=time_index)
        u = read_file(input_file, "vozocrtx", time_index=time_index)[0]
        v = read_file(input_file, "vomecrty", time_index=time_index)[0]

        contour(lon, lat, eta, f'time_{time[time_index]}', 'Elevation')
        contour(lon, lat, u, f'time_{time[time_index]}', 'Zonal Velocity')
        contour(lon, lat, v, f'time_{time[time_index]}', 'Meridional Velocity')
        """
    time_index = 0
    eta = read_file(input_file, "sshn", time_index=time_index)
    #u = read_file(input_file, "vozocrtx", time_index=time_index)[0]
    #v = read_file(input_file, "vomecrty", time_index=time_index)[0]

    # time in years
    yr = time[time_index] #/ 3.154e7
    contour(lon, lat, eta, f'{yr}', 'Elevation')
    #contour(lon, lat, u, f'{yr}yrs', 'Zonal Velocity')
    #contour(lon, lat, v, f'{yr}yrs', 'Meridional Velocity')

if __name__ == '__main__':
    # time to plot
    time_index = 0
    #one_plot(time_index)
    #plots_full()
    plots_1000yrs()

