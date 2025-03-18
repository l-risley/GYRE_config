import numpy as np
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from read_nemo_fields import *
from general_functions import *

def normalised_fft(signal, signal_name):
    """
    Calculate the Fourier transform of 'signal' and plot the normalised FFT spectrum with the frequency components.
    Plot the frequency axis.
    Plot the real-vlaue signals.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "monospace",
        "font.monospace": 'Computer Modern Typewriter',
    })
    fourier = fft(signal)
    # Calculate N/2 to normalize the FFT output
    N = len(signal)
    normalize = N / 2
    # Plot the normalized FFT (|Xk|)/(N/2)
    plt.plot(np.abs(fourier) / normalize)
    plt.ylabel('Amplitude')
    plt.xlabel('Samples')
    plt.title(f'Normalized FFT Spectrum for {signal_name}')
    plt.show()

def normalised_fft_freq(signal, signal_name):
    """
    Calculate the Fourier transform of 'signal' and plot the normalised FFT spectrum.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "monospace",
        "font.monospace": 'Computer Modern Typewriter',
    })
    # Calculate N/2 to normalize the FFT output
    N = len(signal)
    # Get the frequency components of the spectrum
    sampling_rate = 100.0  # It's used as a sample spacing

    # Plot the actual spectrum of the signal
    plt.plot(2 * np.abs(rfft(signal)) / N) #rfftfreq(N, d=1 / sampling_rate)
    #print(np.max(2 * np.abs(rfft(signal)) / N))
    plt.title(f'Spectrum for {signal_name}')
    #plt.ylim(0, 0.08)
    # plt.ylim(0, 2500)
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude')
    plt.show()

def contour_fft_lines(x, y, z, fft, plot_of: str, variable_name: str, line : str):
    # 2D contour plot of one variable
    # switch coords from m to km
    plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')#, vmin=-15000, vmax=15000)
    # ax = sns.heatmap(z, cmap = 'ocean')
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.title(f'{plot_of}')
    if variable_name == 'Elevation':
        units = '$m$'
    elif variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
        variable_name = 'Streamfunction'
    else:
        units = '$ms^{-1}$'
    plt.colorbar(label=f'{variable_name} ({units})')
    if line == 'horizontal':
        x_1, x_2, x_3, x_4 = x[fft[0], :], x[fft[1], :], x[fft[2], :], x[fft[3], :]
        y_1, y_2, y_3, y_4 = y[fft[0], :], y[fft[1], :], y[fft[2], :], y[fft[3], :]
        plt.plot(x_1, y_1, c='k', label='Line 1')
        plt.plot(x_2, y_2, c='y', label='Line 2')
        plt.plot(x_3, y_3, c='w', label='Line 3')
        plt.plot(x_4, y_4, c='r', label='Line 4')
    elif line == 'vertical':
        x_1, x_2, x_3 = x[:, fft[0]], x[:, fft[1]], x[:, fft[2]]
        y_1, y_2, y_3 = y[:, fft[0]], y[:, fft[1]], y[:, fft[2]]
        plt.plot(x_1, y_1, c='k', label='Line 1')
        plt.plot(x_2, y_2, c='y', label='Line 2')
        plt.plot(x_3, y_3, c='w', label='Line 3')
    #plt.savefig(f'plots/{plot_of}{variable_name}.png')
    plt.legend()
    plt.show()

def fft_checkerboard_horizontal(exp_no, sf_filter: str):
    # input psi field
    # netcdf file locations
    exp_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp_no}.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(exp_input_file)

    # gyre12 outputs
    psi = read_file(exp_input_file, "psi")[-3]
    u = read_file(exp_input_file, "u")[-3]
    u_recon = read_file(exp_input_file, "u_inv")[-3]
    v = read_file(exp_input_file, "v")[-3]
    v_recon = read_file(exp_input_file, "v_inv")[-3]

    # calculate the raw error
    u_err = u - u_recon
    v_err = v - v_recon

    # run ffft to identify the strength of the checkerboard effect
    ny, nx = np.shape(psi)

    # the lattitudes
    x_1_lat = np.int(np.floor(2 * ny / 9))
    x_2_lat = np.int(np.floor(4 * ny / 9))
    x_3_lat = np.int(np.floor(6 * ny / 9))
    x_4_lat = np.int(np.floor(8 * ny / 9))
    x_lat = [x_1_lat, x_2_lat, x_3_lat, x_4_lat]

    ### FFT FOR STREAMFUNCTION ###
    # take strips of x
    x_1 = psi[x_1_lat, :]
    x_2 = psi[x_2_lat, :]
    x_3 = psi[x_3_lat, :]
    x_4 = psi[x_4_lat, :]

    # plot the field with the lines on top
    contour_fft_lines(lon, lat, psi, x_lat, f'Streamfunction - {sf_filter}', 'SF', 'horizontal')

    # plot the FFT
    normalised_fft_freq(x_1, f'$\psi$ for line 1.')
    normalised_fft_freq(x_2, f'$\psi$ for line 2.')
    normalised_fft_freq(x_3, f'$\psi$ at for line 3.')
    normalised_fft_freq(x_4, f'$\psi$ at for line 4.')

    """
    ### FFT FOR THE VELOCITIES ###
    u_1 = u[x_1_lat, :]
    u_2 = u[x_2_lat, :]
    u_3 = u[x_3_lat, :]

    v_1 = v[x_1_lat, :]
    v_2 = v[x_2_lat, :]
    v_3 = v[x_3_lat, :]

    # plot the field with the lines on top
    contour_fft_lines(lon, lat, u, x_lat, 'Zonal velocity increment', 'u')
    contour_fft_lines(lon, lat, v, x_lat, 'Meridional velocity increment', 'v')

    # plot the FFT
    normalised_fft_freq(u_1, f'$u$ for line 1.')
    normalised_fft_freq(u_2, f'$u$ for line 2.')
    normalised_fft_freq(u_3, f'$u$ at for line 3.')

    # plot the FFT
    normalised_fft_freq(v_1, f'$v$ for line 1.')
    normalised_fft_freq(v_2, f'$v$ for line 2.')
    normalised_fft_freq(v_3, f'$v$ at for line 3.')
    
    ### FFT FOR THE RECONSTRUCTED VELOCITIES ###
    u_r_1 = u_recon[x_1_lat, :]
    u_r_2 = u_recon[x_2_lat, :]
    u_r_3 = u_recon[x_3_lat, :]

    v_r_1 = v_recon[x_1_lat, :]
    v_r_2 = v_recon[x_2_lat, :]
    v_r_3 = v_recon[x_3_lat, :]

    # plot the field with the lines on top
    contour_fft_lines(lon, lat, u_recon, x_lat, 'Reconstructed zonal velocity increment', 'u', 'horizontal')
    contour_fft_lines(lon, lat, v_recon, x_lat, 'Reconstructed meridional velocity increment', 'v', 'horizontal')

    # plot the FFT
    normalised_fft_freq(u_r_1, '$u_{re}$ for line 1.')
    normalised_fft_freq(u_r_2, '$u_{re}$ for line 2.')
    normalised_fft_freq(u_r_3, '$u_{re}$ at for line 3.')

    # plot the FFT
    normalised_fft_freq(v_r_1, '$v_{re}$ for line 1.')
    normalised_fft_freq(v_r_2, '$v_{re}$ for line 2.')
    normalised_fft_freq(v_r_3, '$v_{re}$ at for line 3.')
    """
    ### FFT FOR THE RECONSTRUCTED VELOCITIES ERROR###
    u_err_1 = u_err[x_1_lat, :]
    u_err_2 = u_err[x_2_lat, :]
    u_err_3 = u_err[x_3_lat, :]
    u_err_4 = u_err[x_4_lat, :]

    v_err_1 = v_err[x_1_lat, :]
    v_err_2 = v_err[x_2_lat, :]
    v_err_3 = v_err[x_3_lat, :]
    v_err_4 = v_err[x_4_lat, :]

    # plot the field with the lines on top
    #contour_fft_lines(lon, lat, u_err, x_lat, 'Zonal velocity increment error', 'u', 'horizontal')
    #contour_fft_lines(lon, lat, v_err, x_lat, 'Meridional velocity increment error', 'v', 'horizontal')

    # plot the FFT
    #normalised_fft_freq(u_err_1, '$u_{err}$ for line 1.')
    #normalised_fft_freq(u_err_2, '$u_{err}$ for line 2.')
    #normalised_fft_freq(u_err_3, '$u_{err}$ at for line 3.')
    #normalised_fft_freq(u_err_4, '$u_{err}$ at for line 4.')

    # plot the FFT
    normalised_fft_freq(v_err_1, '$v_{err}$ for line 1.')
    normalised_fft_freq(v_err_2, '$v_{err}$ for line 2.')
    normalised_fft_freq(v_err_3, '$v_{err}$ at for line 3.')
    normalised_fft_freq(v_err_4, '$v_{err}$ at for line 4.')

def fft_checkerboard_vertical(exp_no, sf_filter: str):
    # input psi field
    # netcdf file locations
    exp_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp_no}.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(exp_input_file)

    # gyre12 outputs
    psi = read_file(exp_input_file, "psi")[0]
    u = read_file(exp_input_file, "u")[0]
    u_recon = read_file(exp_input_file, "u_inv")[0]
    v = read_file(exp_input_file, "v")[0]
    v_recon = read_file(exp_input_file, "v_inv")[0]

    # calculate the raw error
    u_err = u - u_recon
    v_err = v - v_recon

    # run ffft to identify the strength of the checkerboard effect
    ny, nx = np.shape(psi)

    # the longitudes
    x_1_lon = np.int(np.floor(nx / 7))
    x_2_lon = np.int(np.floor(nx / 2))
    x_3_lon = np.int(np.floor(3 * nx / 4))
    x_lon = [x_1_lon, x_2_lon, x_3_lon]

    ### FFT FOR STREAMFUNCTION ###
    # take strips of x
    x_1 = psi[:, x_1_lon]
    x_2 = psi[:, x_2_lon]
    x_3 = psi[:, x_3_lon]

    # plot the field with the lines on top
    contour_fft_lines(lon, lat, psi, x_lon, f'Streamfunction - {sf_filter}', 'SF', 'vertical')

    # plot the FFT
    normalised_fft_freq(x_1, f'$\psi$ for line 1.')
    normalised_fft_freq(x_2, f'$\psi$ for line 2.')
    normalised_fft_freq(x_3, f'$\psi$ for line 3.')

    """
    ### FFT FOR THE VELOCITIES ###
    u_1 = u[:, x_1_lon]
    u_2 = u[:, x_2_lon]
    u_3 = u[:, x_3_lon]

    v_1 = v[:, x_1_lon]
    v_2 = v[:, x_2_lon]
    v_3 = v[:, x_3_lon]

    # plot the FFT
    normalised_fft_freq(u_1, f'$u$ for line 1.')
    normalised_fft_freq(u_2, f'$u$ for line 2.')
    normalised_fft_freq(u_3, f'$u$ at for line 3.')

    # plot the FFT
    normalised_fft_freq(v_1, f'$v$ for line 1.')
    normalised_fft_freq(v_2, f'$v$ for line 2.')
    normalised_fft_freq(v_3, f'$v$ at for line 3.')
    
    ### FFT FOR THE RECONSTRUCTED VELOCITIES ###
    u_r_1 = u_recon[:, x_1_lon]
    u_r_2 = u_recon[:, x_2_lon]
    u_r_3 = u_recon[:, x_3_lon]

    v_r_1 = v_recon[:, x_1_lon]
    v_r_2 = v_recon[:, x_2_lon]
    v_r_3 = v_recon[:, x_3_lon]

    # plot the FFT
    normalised_fft_freq(u_r_1, '$u_{re}$ for line 1.')
    normalised_fft_freq(u_r_2, '$u_{re}$ for line 2.')
    normalised_fft_freq(u_r_3, '$u_{re}$ at for line 3.')

    # plot the FFT
    normalised_fft_freq(v_r_1, '$v_{re}$ for line 1.')
    normalised_fft_freq(v_r_2, '$v_{re}$ for line 2.')
    normalised_fft_freq(v_r_3, '$v_{re}$ at for line 3.')
    """
    ### FFT FOR THE RECONSTRUCTED VELOCITIES ERROR###
    u_err_1 = u_err[:, x_1_lon]
    u_err_2 = u_err[:, x_2_lon]
    u_err_3 = u_err[:, x_3_lon]

    v_err_1 = v_err[:, x_1_lon]
    v_err_2 = v_err[:, x_2_lon]
    v_err_3 = v_err[:, x_3_lon]

    # plot the FFT
    normalised_fft_freq(u_err_1, '$u_{err}$ for line 1.')
    normalised_fft_freq(u_err_2, '$u_{err}$ for line 2.')
    normalised_fft_freq(u_err_3, '$u_{err}$ at for line 3.')

    # plot the FFT
    normalised_fft_freq(v_err_1, '$v_{err}$ for line 1.')
    normalised_fft_freq(v_err_2, '$v_{err}$ for line 2.')
    normalised_fft_freq(v_err_3, '$v_{err}$ at for line 3.')

def rel_diff_fft(exp1, exp2):
    """
    Plot the relative difference between the FFT of two exps.
    """
    exp1_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp1}.nc"
    exp2_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp2}.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(exp1_input_file)

    # gyre12 outputs
    psi_1 = read_file(exp1_input_file, "psi")[0]
    psi_2 = read_file(exp2_input_file, "psi")[0]
    v_1 = read_file(exp1_input_file, "v")[0]
    v_recon_1 = read_file(exp1_input_file, "v_inv")[0]
    v_2 = read_file(exp2_input_file, "v")[0]
    v_recon_2 = read_file(exp2_input_file, "v_inv")[0]

    # calculate the raw error
    v_err_1 = v_1 - v_recon_1
    v_err_2 = v_2 - v_recon_2

    # run ffft to identify the strength of the checkerboard effect
    ny, nx = np.shape(psi_1)

    # the lattitudes
    x_1_lat = np.int(np.floor(2 * ny / 9))
    x_2_lat = np.int(np.floor(4 * ny / 9))
    x_3_lat = np.int(np.floor(6 * ny / 9))
    x_4_lat = np.int(np.floor(8 * ny / 9))
    x_lat = [x_1_lat, x_2_lat, x_3_lat, x_4_lat]


    for i in range(4):
        psi_line_1 = psi_1[x_lat[i], :]
        psi_line_2 = psi_2[x_lat[i], :]
        v_err_line_1 = v_err_1[x_lat[i], :]
        v_err_line_2 = v_err_2[x_lat[i], :]
        N_psi = len(psi_line_1)
        fft_psi_1 = 2 * np.abs(rfft(psi_line_1)) / N_psi
        fft_psi_2 = 2 * np.abs(rfft(psi_line_2)) / N_psi

        N_v = len(v_err_line_1)
        fft_v_1 = 2 * np.abs(rfft(v_err_line_1)) / N_v
        fft_v_2 = 2 * np.abs(rfft(v_err_line_2)) / N_v

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "monospace",
            "font.monospace": 'Computer Modern Typewriter',
        })

        # Plot the actual spectrum of the signal
        plt.plot(np.abs((fft_psi_1 - fft_psi_2)/fft_psi_1))
        plt.title(f'Spectrum difference for $\psi$ for line {i+1}.')
        #plt.ylim(0, 2500)
        plt.xlabel('Frequency[Hz]')
        plt.ylabel('Amplitude')
        plt.show()

        # Plot the actual spectrum of the signal
        plt.plot(np.abs((fft_v_1 - fft_v_2))/fft_v_1)
        plt.title(f'Spectrum difference for $v$ for line {i+1}.')
        #plt.ylim(0, 2500)
        plt.xlabel('Frequency[Hz]')
        plt.ylabel('Amplitude')
        plt.show()

def normalised_fft_freq_depths(signals, depths, signal_name):
    """
    Calculate the Fourier transform of 'signal' and plot the normalised FFT spectrum.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "monospace",
        "font.monospace": 'Computer Modern Typewriter',
    })
    # Initialize an empty list to store the FFT magnitudes
    fft_magnitudes = []

    # Loop over all depths and calculate the FFT magnitude
    for i in range(31):
        # Get the signal for depth i
        signal = signals[f'depth_{i}']
        N = len(signal)  # Length of the signal
        # Perform the rfft and compute the magnitude
        fft = rfft(signal)
        magnitude = 2 * np.abs(fft) / N  # Normalize the FFT magnitude
        fft_magnitudes.append(magnitude)

    # Convert the list of FFT magnitudes into a numpy array (31 x N)
    fft_magnitudes = np.array(fft_magnitudes)

    # Compute the average FFT magnitude across depths
    average_fft = np.mean(fft_magnitudes, axis=0)

    # Compute the min and max FFT magnitudes for the shaded region
    min_fft = np.min(fft_magnitudes, axis=0)
    max_fft = np.max(fft_magnitudes, axis=0)

    # Plot the average FFT magnitude
    plt.plot(np.arange(len(average_fft)), average_fft, label='Average FFT', color='b', linewidth=2)
    plt.plot(np.arange(len(average_fft)), fft_magnitudes[0], label='Surface FFT', color='r', linewidth=1)

    # Plot the shaded region representing the range of FFT magnitudes
    plt.fill_between(np.arange(len(average_fft)), min_fft, max_fft, color='gray', alpha=0.3,  label='Range of FFT')
    plt.title(f'Spectrum for {signal_name}')
    #plt.ylim(0, 2500)
    plt.ylim(0, 0.03)
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def fft_checkerboard_multiple_depths(exp_no, sf_filter: str):
    # input psi field
    # netcdf file locations
    exp_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp_no}.nc"
    depths = np.loadtxt('/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/gyre_depths.txt', delimiter=',')
    #idx = np.round(np.linspace(0, len(depths) - 1, 4)).astype(int)
    #depths = depths[idx]
    # lon and lat for each grid
    lon, lat, time = read_file_info(exp_input_file)
    ny, nx = np.shape(lon)

    # the lattitudes
    x_1_lat = np.int(np.floor(2 * ny / 9))
    x_2_lat = np.int(np.floor(4 * ny / 9))
    x_3_lat = np.int(np.floor(6 * ny / 9))
    x_4_lat = np.int(np.floor(8 * ny / 9))
    x_lat = [x_1_lat, x_2_lat, x_3_lat, x_4_lat]

    psi_1, psi_2, psi_3, psi_4 = {}, {}, {}, {}
    v_err_1, v_err_2, v_err_3, v_err_4 = {}, {}, {}, {}

    for d in range(len(depths)):
        print
        # gyre12 outputs
        psi = read_file(exp_input_file, "psi")[d]
        v = read_file(exp_input_file, "v")[d]
        v_recon = read_file(exp_input_file, "v_inv")[d]

        # calculate the raw error
        v_err = v - v_recon

        # take strips of x
        psi_1[f"depth_{d}"] = psi[x_1_lat, :]
        psi_2[f"depth_{d}"] = psi[x_2_lat, :]
        psi_3[f"depth_{d}"] = psi[x_3_lat, :]
        psi_4[f"depth_{d}"] = psi[x_4_lat, :]

        v_err_1[f"depth_{d}"] = v_err[x_1_lat, :]
        v_err_2[f"depth_{d}"] = v_err[x_2_lat, :]
        v_err_3[f"depth_{d}"] = v_err[x_3_lat, :]
        v_err_4[f"depth_{d}"] = v_err[x_4_lat, :]

    # plot the field with the lines on top
    #contour_fft_lines(lon, lat, psi, x_lat, f'Streamfunction - {sf_filter}', 'SF', 'horizontal')

    # plot the FFT
    normalised_fft_freq_depths(psi_1, depths, f'$\psi$ for line 1.')
    normalised_fft_freq_depths(psi_2, depths, f'$\psi$ for line 2.')
    normalised_fft_freq_depths(psi_3, depths, f'$\psi$ at for line 3.')
    normalised_fft_freq_depths(psi_4, depths, f'$\psi$ at for line 4.')

    normalised_fft_freq_depths(v_err_1, depths, '$v_{err}$ for line 1.')
    normalised_fft_freq_depths(v_err_2, depths, '$v_{err}$ for line 2.')
    normalised_fft_freq_depths(v_err_3, depths, '$v_{err}$ at for line 3.')
    normalised_fft_freq_depths(v_err_4, depths, '$v_{err}$ at for line 4.')

def fft_checkerboard_assim():
    # input psi field
    # netcdf file locations
    exp_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/assim_exps/unbal_increments_exp19.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(exp_input_file)

    # gyre12 outputs
    v = read_file(exp_input_file, "bckinpsiunbal")[0]

    # run ffft to identify the strength of the checkerboard effect
    ny, nx = np.shape(v)

    # the lattitudes
    x_1_lat = np.int(np.floor(2 * ny / 9))
    x_2_lat = np.int(np.floor(4 * ny / 9))
    x_3_lat = np.int(np.floor(6 * ny / 9))
    x_4_lat = np.int(np.floor(8 * ny / 9))
    x_lat = [x_1_lat, x_2_lat, x_3_lat, x_4_lat]

    ### FFT FOR STREAMFUNCTION ###
    # take strips of x
    x_1 = v[x_1_lat, :]
    x_2 = v[x_2_lat, :]
    x_3 = v[x_3_lat, :]
    x_4 = v[x_4_lat, :]

    # plot the field with the lines on top
    contour_fft_lines(lon, lat, v, x_lat, f'Meridional velocity', 'u', 'horizontal')

    # plot the FFT
    normalised_fft_freq(x_1, f'v for line 1.')
    x_1_fft = 2 * np.abs(rfft(x_1)) / len(x_1)
    normalised_fft_freq(x_2, f'v for line 2.')
    x_2_fft = 2 * np.abs(rfft(x_2)) / len(x_2)
    normalised_fft_freq(x_3, f'v at for line 3.')
    x_3_fft = 2 * np.abs(rfft(x_3)) / len(x_3)
    normalised_fft_freq(x_4, f'v at for line 4.')
    x_4_fft = 2 * np.abs(rfft(x_4)) / len(x_4)
    print(x_1 - x_2)
    print(x_3 - x_4)

if __name__ == '__main__':
    fft_checkerboard_horizontal(31, 'No filter')#'5 iterations of filter')
    #fft_checkerboard_vertical(7, 'No filter')
    #rel_diff_fft(39, 40)
    #fft_checkerboard_multiple_depths(40, 'No filter')  # '5 iterations of filter')
    #fft_checkerboard_assim()