"""
Interpolate between different grid resolutions.
"""
import numpy as np
import numpy.ma as ma
import netCDF4
from scipy.interpolate import griddata
from read_nemo_fields import *
from general_functions import *

def points_array(lon, lat):
    """
    Create a 2d array of coordinates (x coordiates in first column, y coodrinates in 2nd column)
    from 2 arrays of x and y separately.
    """
    lon_flat, lat_flat = lon.flatten(), lat.flatten()
    points_arr = np.empty((len(lon_flat), 2))
    points_arr[:, 0] = lon_flat
    points_arr[:, -1] = lat_flat
    return points_arr

def gyre1_gyre12():
    """
    Interpolate from 1 degree gyre domain (coarse) to the 1/12th degree gyre domain (high resolution).
    """

    time_index = 0

    # netcdf file locations
    gyre1_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/model_output_1000yrs.nc"
    gyre12_eta_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_T_depth0.nc"
    gyre12_u_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_U_depth0.nc"
    gyre12_v_input_file = "/c/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/instant.grid_V_depth0.nc"

    # lon and lat for each grid
    gyre1_lon, gyre1_lat, time = read_file_info(gyre1_input_file)
    gyre12_eta_lon, gyre12_eta_lat, time = read_file_info(gyre12_eta_input_file)
    gyre12_u_lon, gyre12_u_lat, time = read_file_info(gyre12_u_input_file)
    gyre12_v_lon, gyre12_v_lat, time = read_file_info(gyre12_v_input_file)


    # gyre1 outputs
    gyre1_eta = read_file(gyre1_input_file, "sshn", time_index=time_index)
    gyre1_u = read_file(gyre1_input_file, "un", time_index=time_index)[0]
    gyre1_v = read_file(gyre1_input_file, "vn", time_index=time_index)[0]

    # plot gyre1
    contour(gyre1_lon, gyre1_lat, gyre1_eta, f'Gyre1 1000yrs', 'Elevation')
    contour(gyre1_lon, gyre1_lat, gyre1_u, f'Gyre1 1000yrs', 'Zonal Velocity')
    contour(gyre1_lon, gyre1_lat, gyre1_v, f'Gyre1 1000yrs', 'Meridional Velocity')

    gyre1_points = points_array(gyre1_lon, gyre1_lat)

    # interpolate to gyre12
    gyre12_eta_points = points_array(gyre12_eta_lon, gyre12_eta_lat)
    gyre12_eta = griddata(gyre1_points, gyre1_eta.flatten(), gyre12_eta_points, method='cubic')
    gyre12_eta = ma.reshape(gyre12_eta, ma.shape(gyre12_eta_lon))

    gyre12_u_points = points_array(gyre12_u_lon, gyre12_u_lat)
    gyre12_u = griddata(gyre1_points, gyre1_u.flatten(), gyre12_u_points, method='cubic')
    gyre12_u = ma.reshape(gyre12_u, ma.shape(gyre12_u_lon))

    gyre12_v_points = points_array(gyre12_v_lon, gyre12_v_lat)
    gyre12_v = griddata(gyre1_points, gyre1_v.flatten(), gyre12_v_points, method='cubic')
    gyre12_v = ma.reshape(gyre12_v, ma.shape(gyre12_v_lon))

    # plot gyre 12
    contour(gyre12_eta_lon, gyre12_eta_lat, gyre12_eta, f'Gyre12 1000yrs', 'Elevation')
    contour(gyre12_u_lon, gyre12_u_lat, gyre12_u, f'Gyre12 1000yrs', 'Zonal Velocity')
    contour(gyre12_v_lon, gyre12_v_lat, gyre12_v, f'Gyre12 1000yrs', 'Meridional Velocity')

    # convert to netcdf files
    fileid = netCDF4.Dataset('gyre12_1000yr_restart.nc', 'r+')
    fileid.variables['sshn'][:] = gyre12_eta
    #fileid.variables['un'][:] = gyre12_u
    #fileid.variables['vn'][:] = gyre12_v
    fileid.close()

if __name__ == '__main__':
   gyre1_gyre12()

