import netCDF4
import numpy as np

def read_file_info(input_file):
    '''
    Read information about the grid and time from a netCDF file
    '''

    fileid = netCDF4.Dataset(input_file, mode='r')
    lon = fileid.variables['nav_lon'][:]
    lat = fileid.variables['nav_lat'][:]
    time = fileid.variables['time_counter'][:]
    fileid.close()

    return lon, lat, time

def read_file(input_file, var_name, time_index=0):
    '''
    Read variable var_name from a netCDF file
    '''

    fileid = netCDF4.Dataset(input_file, mode='r')
    variable = fileid.variables[var_name][time_index,0,:, :]
    fileid.close()

    return variable

if __name__ == "__main__":

    input_file = "/projects/jodap/frwn/GYRE/results_gyre12/mersea.grid_T.nc"

    lon, lat, time = read_file_info(input_file)
    num_times = np.size(time)
    print(num_times)

    for time_index in range(num_times-1):
        ssh = read_file(input_file, "sossheig", time_index=time_index)
        sshp1 = read_file(input_file, "sossheig", time_index=time_index+1)
        diff = sshp1 - ssh
        print(diff)





