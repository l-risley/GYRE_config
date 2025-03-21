"""
PLot outputs from the gyre12 config.
"""
import matplotlib.pyplot as plt
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

def contour_gyre_inv(x, y, z, plot_of: str, variable_name: str, exp:str):
    # 2D contour_gyre plot of one variable
    # switch coords from m to km
    plt.title(f'{plot_of} - Experiment {exp}')
    if variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-15000, vmax=15000)
    elif variable_name == 'u_err' or variable_name == 'v_err':
        units = None
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-1, vmax=1)
    else:
        units = '$ms^{-1}$'
        plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=0, vmax=1)
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.colorbar(label=f'{plot_of} ({units})')
    #plt.savefig(f'f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/{variable_name}_{exp}.png')
    plt.show()

def contour_line(x, y, z, line):
    # 2D contour plot of one variable
    # switch coords from m to km
    plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')#, vmin=-15000, vmax=15000)
    # ax = sns.heatmap(z, cmap = 'ocean')
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    x_1 = x[line, :]
    y_1 = y[line, :]
    plt.plot(x_1, y_1, c='r')

    plt.show()

def comp_gyre_inverse_tests(exp1, exp2):
    """
    Plot the output of the gyre inverse tests.
    Input:
    exp, which experiment number.
    Ouput:
    plots of the velocities and errors.
    """
    # netcdf file locations
    exp1_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp1}.nc"
    exp2_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp2}.nc"
    # lon and lat for each grid
    lon, lat, time = read_file_info(exp1_input_file)


    # gyre12 outputs
    gyre12_u_1 = read_file(exp1_input_file, "u")[0]
    gyre12_v_1 = read_file(exp1_input_file, "v")[0]
    gyre12_psi_1 = read_file(exp1_input_file, "psi")[0]
    gyre12_chi_1 = read_file(exp1_input_file, "chi")[0]
    gyre12_u_inv_1 = read_file(exp1_input_file, "u_inv")[0]
    gyre12_v_inv_1 = read_file(exp1_input_file, "v_inv")[0]
    gyre12_u_err_1 = read_file(exp1_input_file, "u_rel_err")[0]
    gyre12_v_err_1 = read_file(exp1_input_file, "v_rel_err")[0]

    gyre12_u_2 = read_file(exp2_input_file, "u")[0]
    gyre12_v_2 = read_file(exp2_input_file, "v")[0]
    gyre12_psi_2 = read_file(exp2_input_file, "psi")[0]
    gyre12_chi_2 = read_file(exp2_input_file, "chi")[0]
    gyre12_u_inv_2 = read_file(exp2_input_file, "u_inv")[0]
    gyre12_v_inv_2 = read_file(exp2_input_file, "v_inv")[0]
    gyre12_u_err_2 = read_file(exp2_input_file, "u_rel_err")[0]
    gyre12_v_err_2 = read_file(exp2_input_file, "v_rel_err")[0]

    ny, nx = np.shape(gyre12_v_err_2)

    # calculate the raw error
    u_err_1 = gyre12_u_1 - gyre12_u_inv_1
    v_err_1 = gyre12_v_1 - gyre12_v_inv_1
    u_err_2 = gyre12_u_2 - gyre12_u_inv_2
    v_err_2 = gyre12_v_2 - gyre12_v_inv_2
    u_err_diff = u_err_2 - u_err_1
    u_err_diff_rel_1 = u_err_diff / u_err_1
    u_err_diff_rel_2 = u_err_diff / u_err_2

    # find location of maximum value
    max_index = np.unravel_index(np.argmax(u_err_diff, axis=None), u_err_diff.shape)
    print(max_index)
    print(u_err_diff[max_index])

    # plot gyre12
    contour_gyre_inv(lon, lat, u_err_1, 'Zonal error', 'v_raw_err', f'{exp1}')
    contour_gyre_inv(lon, lat, u_err_2, 'Zonal error', 'v_raw_err', f'{exp2}')
    contour_gyre_inv(lon, lat, u_err_diff, 'Difference in zonal error', 'diff_vel_err', f'{exp2} - {exp1}')
    contour_gyre_inv(lon, lat, abs(u_err_diff_rel_1), 'Relative difference in zonal error', 'dif_err', f'{exp1}')
    contour_gyre_inv(lon, lat, abs(u_err_diff_rel_2), 'Relative difference in zonal error', 'dif_err', f'{exp2}')

    # the lattitudes
    strip_lat = max_index[0]

    ### FFT FOR STREAMFUNCTION ###
    # take strips of x
    u_err_1_strip = u_err_1[strip_lat, :]
    u_err_2_strip = u_err_2[strip_lat, :]
    u_err_diff_strip = u_err_diff[strip_lat, :]
    u_err_diff_rel_1_strip = u_err_diff_rel_1[strip_lat, :]
    u_err_diff_rel_2_strip = u_err_diff_rel_2[strip_lat, :]

    x_1 = lon[strip_lat, :]
    y_1 = lat[strip_lat, :]
    plt.plot(x_1, u_err_1_strip, c='k', label=f'Experiment {exp1}')
    plt.plot(x_1, u_err_2_strip, c='b', label=f'Experiment {exp2}')
    plt.plot(x_1, u_err_diff_strip, c='r', label=f'Difference')
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('($ms^{-1}$)')
    plt.title(f'Zonal velocity error near centre')
    plt.legend()
    plt.show()

    plt.plot(x_1,  u_err_diff_rel_1_strip, c='k', label=f'Experiment {exp1}')
    plt.plot(x_1, u_err_diff_rel_2_strip, c='b', label=f'Experiment {exp2}')
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('%')
    plt.title(f'Zonal velocity error difference relative near north boundary')
    plt.legend()
    plt.show()

    contour_line(lon, lat, u_err_diff, strip_lat)

def save_rms_its():
    info = np.loadtxt(
        f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/exps_mu_cc.txt',
        delimiter=",")
    exps = info[:, 0]
    mu_arr = info[:, 1]
    cc_arr = info[:, 2]
    u_rms_arr = np.empty_like(exps)
    v_rms_arr = np.empty_like(exps)
    its_arr = np.empty_like(exps)
    # Ensure all arrays are of the same length
    assert len(mu_arr) == len(cc_arr) == len(u_rms_arr) == len(v_rms_arr), "Arrays must be the same length"

    idx = 0

    for i in exps:
        i = int(i)
        exp_nc = f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{i}.nc'
        conv = np.loadtxt(f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/convergence_files/exp{i}_level1.txt', delimiter=",")
        lon, lat, time = read_file_info(exp_nc)

        u = read_file(exp_nc, "u")[0]
        v = read_file(exp_nc, "v")[0]
        psi = read_file(exp_nc, "psi")[0]
        chi = read_file(exp_nc, "chi")[0]
        u_inv = read_file(exp_nc, "u_inv")[0]
        v_inv = read_file(exp_nc, "v_inv")[0]
        u_rel_err = read_file(exp_nc, "u_rel_err")[0]
        v_rel_err = read_file(exp_nc, "v_rel_err")[0]

        # calculate the raw error
        u_err = u - u_inv
        v_err = v - v_inv

        # RMS of raw-error
        u_rms = calculate_rms_2d(u_err)
        v_rms = calculate_rms_2d(v_err)

        u_rms_arr[idx] = u_rms
        v_rms_arr[idx] = v_rms
        its_arr[idx] = len(conv)

        idx += 1

    u_rms_col = u_rms_arr[:, np.newaxis]  # Convert to a column vector
    v_rms_col = v_rms_arr[:, np.newaxis]  # Convert to a column vector
    its_col = its_arr[:, np.newaxis]  # Convert to a column vector

    # Ensure the `rms` vector has the same number of rows as `data`
    if len(u_rms_col) != len(info):
        raise ValueError("Length of `rms` vector must match the number of rows in the input file.")
    if len(v_rms_col) != len(info):
        raise ValueError("Length of `rms` vector must match the number of rows in the input file.")

    # Concatenate the new column to the existing data
    updated_data = np.hstack((info, u_rms_col))
    updated_data_2 = np.hstack((updated_data, v_rms_col))
    final_data = np.hstack((updated_data_2, its_col))

    # Save the modified data to a new file
    np.savetxt("exps_mu_cc_rms.txt", final_data, delimiter=",", fmt="%0.7f")  # Adjust `fmt` as needed

    print("File saved as exps_mu_cc_rms.txt")

def save_rms_choose_depth(level):
    info = np.loadtxt(
        f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/exps_mu_cc.txt',
        delimiter=",")
    exps = info[:, 0]
    mu_arr = info[:, 1]
    cc_arr = info[:, 2]
    u_rms_arr = np.empty_like(exps)
    v_rms_arr = np.empty_like(exps)
    its_arr = np.empty_like(exps)
    # Ensure all arrays are of the same length
    assert len(mu_arr) == len(cc_arr) == len(u_rms_arr) == len(v_rms_arr), "Arrays must be the same length"
    idx = 0
    for i in exps:
        i = int(i)
        exp_nc = f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{i}.nc'
        lon, lat, time = read_file_info(exp_nc)

        u = read_file(exp_nc, "u")[level]
        v = read_file(exp_nc, "v")[level]
        psi = read_file(exp_nc, "psi")[level]
        chi = read_file(exp_nc, "chi")[level]
        u_inv = read_file(exp_nc, "u_inv")[level]
        v_inv = read_file(exp_nc, "v_inv")[level]
        u_rel_err = read_file(exp_nc, "u_rel_err")[level]
        v_rel_err = read_file(exp_nc, "v_rel_err")[level]

        # calculate the raw error
        u_err = u - u_inv
        v_err = v - v_inv

        # RMS of raw-error
        u_rms = calculate_rms_2d(u_err)
        v_rms = calculate_rms_2d(v_err)

        u_rms_arr[idx] = u_rms
        v_rms_arr[idx] = v_rms

        idx += 1

    u_rms_col = u_rms_arr[:, np.newaxis]  # Convert to a column vector
    v_rms_col = v_rms_arr[:, np.newaxis]  # Convert to a column vector
    its_col = its_arr[:, np.newaxis]  # Convert to a column vector

    # Ensure the `rms` vector has the same number of rows as `data`
    if len(u_rms_col) != len(info):
        raise ValueError("Length of `rms` vector must match the number of rows in the input file.")
    if len(v_rms_col) != len(info):
        raise ValueError("Length of `rms` vector must match the number of rows in the input file.")

    # Concatenate the new column to the existing data
    updated_data = np.hstack((info, u_rms_col))
    updated_data_2 = np.hstack((updated_data, v_rms_col))
    final_data = np.hstack((updated_data_2, its_col))

    # Save the modified data to a new file
    np.savetxt(f"exps_mu_cc_rms_level{level+1}.txt", final_data, delimiter=",", fmt="%0.7f")  # Adjust `fmt` as needed

    print(f"File saved as exps_mu_cc_rms_level{level+1}.txt")

def comp_gyre_inverse_rms(exp):
    """
    """
    info = np.loadtxt(f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/exps_mu_cc_rms_{exp}.txt',delimiter=",")
    exps = info[:, 0]
    mu_arr = info[:, 1]
    cc_arr = info[:, 2]
    u_rms_arr = info[:, 3]
    v_rms_arr = info[:, 4]
    iters_arr = info[:,5]

    ##### PLOT RMS against mu #####
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
    # Separate data by cc values
    unique_cc = np.unique(cc_arr)  # Find unique values of cc
    for value in unique_cc:
        indices = np.where(cc_arr == value)[0]  # Get indices where cc equals the current value
        ax1.plot(mu_arr[indices], u_rms_arr[indices], marker='o', linestyle='-',  label=rf"$10^{{{int(np.log10(value))}}}$")
        print(f'RMS for cc = {value} and mu={mu_arr[indices]} = {u_rms_arr[indices]}.')
        print(f'Iterations for cc = {value} and mu={mu_arr[indices]} = {iters_arr[indices]}.')
    ax1.set_xlabel('Regularisation parameter')
    ax1.set_ylabel(f'RMS')
    ax1.set_xscale('log')
    ax1.legend(title = "Accuracy tolerance")
    ax1.set_title(f'Zonal velocity ')

    for value in unique_cc:
        indices = np.where(cc_arr == value)[0]  # Get indices where cc equals the current value
        ax2.plot(mu_arr[indices], v_rms_arr[indices], marker='o', linestyle='-', label=rf"$10^{{{int(np.log10(value))}}}$")
        print(f'RMS for cc = {value} and mu={mu_arr[indices]} = {v_rms_arr[indices]}.')
    ax2.set_xlabel('Regularisation parameter')
    #ax2.set_ylabel(f'RMS')
    ax2.set_xscale('log')
    #ax2.yscale('log')
    ax2.legend(title = "Accuracy tolerance")
    ax2.set_title(f'Meridional velocity')

    fig.suptitle('RMS of the reconstructed velocity error')
    plt.show()

    #################### PLOT COLOURMESH #########################
    # Get unique values for grid axes
    cc_unique = np.unique(cc_arr)
    mu_unique = np.unique(mu_arr)

    # Create an empty 2D grid initialized with NaN
    rms_grid = np.full((len(cc_unique), len(mu_unique)), np.nan)

    # Map each (cc, mu) pair to indices in the 2D array
    for cc, mu, rms in zip(cc_arr, mu_arr, u_rms_arr):
        i = np.where(cc_unique == cc)[0][0]  # Row index
        j = np.where(mu_unique == mu)[0][0]  # Column index
        rms_grid[i, j] = rms  # Fill in the correct position

    # Plot the 2D colormap
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(mu_unique, cc_unique, rms_grid, shading='auto', cmap='viridis')
    plt.colorbar(label="RMS Value")
    plt.xlabel("mu values")
    plt.ylabel("cc values")
    plt.title("Colormap of RMS Values")
    #plt.show()

def save_rms_all_depths():
    info = np.loadtxt('/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/exps_mu_cc_notik3.txt', delimiter=",")
    depths = np.loadtxt('/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/gyre_depths.txt', delimiter=',')
    exps = info[:, 0]
    mu_arr = info[:, 1]
    cc_arr = info[:, 2]
    u_rms_arr = np.zeros_like(exps)
    v_rms_arr = np.zeros_like(exps)
    # Ensure all arrays are of the same length
    assert len(mu_arr) == len(cc_arr) == len(u_rms_arr) == len(v_rms_arr), "Arrays must be the same length"

    for d in range(len(depths)):
        idx = 0
        for i in exps:
            i = int(i)
            exp_nc = f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{i}.nc'

            u = read_file(exp_nc, "u")[d]
            v = read_file(exp_nc, "v")[d]
            u_inv = read_file(exp_nc, "u_inv")[d]
            v_inv = read_file(exp_nc, "v_inv")[d]

            # calculate the raw error
            u_err = u - u_inv
            v_err = v - v_inv

            # RMS of raw-error
            u_rms = calculate_rms_2d(u_err)
            v_rms = calculate_rms_2d(v_err)

            u_rms_arr[idx] = u_rms_arr[idx] + u_rms
            v_rms_arr[idx] = v_rms_arr[idx] + v_rms

            idx += 1
    u_rms_arr = u_rms_arr / len(depths)
    v_rms_arr = v_rms_arr / len(depths)
    u_rms_col = u_rms_arr[:, np.newaxis]  # Convert to a column vector
    v_rms_col = v_rms_arr[:, np.newaxis]  # Convert to a column vector

    # Ensure the `rms` vector has the same number of rows as `data`
    if len(u_rms_col) != len(info):
        raise ValueError("Length of `rms` vector must match the number of rows in the input file.")
    if len(v_rms_col) != len(info):
        raise ValueError("Length of `rms` vector must match the number of rows in the input file.")

    # Concatenate the new column to the existing data
    updated_data = np.hstack((info, u_rms_col))
    updated_data_2 = np.hstack((updated_data, v_rms_col))
    # Save the modified data to a new file
    np.savetxt("exps_mu_cc_rms_depth_average_notik3.txt", updated_data_2, delimiter=",", fmt="%0.7f")  # Adjust `fmt` as needed

    print("File saved as exps_mu_cc_rms_depth_average_notik3.txt")

def comp_gyre_inverse_rms_all_depths():
    """
    """
    info = np.loadtxt(f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/exps_mu_cc_rms_depth_average_notik3.txt',delimiter=",")
    exps = info[:, 0]
    mu_arr = info[:, 1]
    cc_arr = info[:, 2]
    u_rms_arr = info[:, 3]
    v_rms_arr = info[:, 4]

    ##### PLOT RMS against mu #####
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
    # Separate data by cc values
    unique_cc = np.unique(cc_arr)  # Find unique values of cc
    for value in unique_cc:
        indices = np.where(cc_arr == value)[0]  # Get indices where cc equals the current value
        ax1.plot(mu_arr[indices], u_rms_arr[indices], marker='o', linestyle='-',  label=rf"$10^{{{int(np.log10(value))}}}$")
        print(f'RMS for cc = {value} and mu={mu_arr[indices]} = {u_rms_arr[indices]}.')
    ax1.set_xlabel('Regularisation parameter')
    ax1.set_ylabel(f'RMS')
    ax1.set_xscale('log')
    ax1.legend(title = "Accuracy tolerance")
    ax1.set_title(f'Zonal velocity ')

    for value in unique_cc:
        indices = np.where(cc_arr == value)[0]  # Get indices where cc equals the current value
        ax2.plot(mu_arr[indices], v_rms_arr[indices], marker='o', linestyle='-', label=rf"$10^{{{int(np.log10(value))}}}$")
        print(f'RMS for cc = {value} and mu={mu_arr[indices]} = {v_rms_arr[indices]}.')
    ax2.set_xlabel('Regularisation parameter')
    #ax2.set_ylabel(f'RMS')
    ax2.set_xscale('log')
    #ax2.yscale('log')
    ax2.legend(title = "Accuracy tolerance")
    ax2.set_title(f'Meridional velocity')

    fig.suptitle('Averaged RMS of the reconstructed velocity error across all depths')
    plt.show()

def plot_rms_filter_depths():
    info = np.loadtxt(
        '/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/exps_mu_cc_with_filter.txt',delimiter=",")
    depths = np.loadtxt(
        '/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/gyre_depths.txt',
        delimiter=',')
    exps = info[:, 0]
    latex_exp = info[:, 3]
    u_rms_arr = np.zeros_like(depths)
    v_rms_arr = np.zeros_like(depths)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
    for i, exp in enumerate(exps):
        exp = int(exp)
        exp_nc = f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp}.nc'
        for d in range(len(depths)):
            u = read_file(exp_nc, "u")[d]
            v = read_file(exp_nc, "v")[d]
            u_inv = read_file(exp_nc, "u_inv")[d]
            v_inv = read_file(exp_nc, "v_inv")[d]

            # calculate the raw error
            u_err = u - u_inv
            v_err = v - v_inv

            # RMS of raw-error
            u_rms = calculate_rms_2d(u_err)
            v_rms = calculate_rms_2d(v_err)

            u_rms_arr[d] = u_rms
            v_rms_arr[d] = v_rms
        ax1.plot(u_rms_arr, depths, linestyle='-', label=f"{int(latex_exp[i])}")
        ax2.plot(v_rms_arr, depths, linestyle='-', label=f"{int(latex_exp[i])}")
    ax1.set_xlabel('RMS')
    ax1.set_ylabel(f'Depth (m)')

    ax1.legend(title="Experiment number")
    ax1.set_title(f'Zonal velocity ')
    ax1.invert_yaxis()
    ax2.set_xlabel('Regularisation parameter')
    ax2.legend(title="Experiment number")
    ax2.set_title(f'Meridional velocity')

    fig.suptitle('RMS of the reconstructed velocity for all depths')
    plt.show()

if __name__ == '__main__':
    #comp_gyre_inverse_tests('16', '20')
    #save_rms_its()
    #comp_gyre_inverse_rms()
    #save_rms_all_depths()
    #comp_gyre_inverse_rms('notik3')
    #comp_gyre_inverse_rms_all_depths()
    """
    exp_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp40.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(exp_input_file)

    # gyre12 outputs
    u = read_file(exp_input_file, "u")[0]
    u_recon = read_file(exp_input_file, "u_inv")[0]
    v = read_file(exp_input_file, "v")[0]
    v_recon = read_file(exp_input_file, "v_inv")[0]

    # calculate the raw error
    u_err = u - u_recon
    v_err = v - v_recon
    u_rms, v_rms = calculate_rms_2d(u_err), calculate_rms_2d(v_err)
    print(u_rms, v_rms)
    """
    plot_rms_filter_depths()
    #save_rms_choose_depth(29)
    #comp_gyre_inverse_rms(f'level30')
