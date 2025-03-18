import numpy as np
import netCDF4
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

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
    variable = fileid.variables[var_name][time_index,:, :]
    fileid.close()

    return variable

def normalised_fft_freq(signal, signal_name, line_no):
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
    plt.plot(2 * np.abs(rfft(signal)) / N)
    if signal_name == 'v_err':
        name = '$v_{err}$'
        plt.title(f'Spectrum for {name} for line {line_no}.')
        plt.ylim(0, 0.0006)
    elif signal_name == 'sf':
        name = '$\psi$'
        plt.title(f'Spectrum for {name} for line {line_no}.')
        plt.ylim(0, 2500)
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude')
    plt.savefig(f'projects/jodap/lrisley/GYRE/experiments/u-di463-invbal/fft_{name}_{line_no}.png')

def fft_checkerboard(field, variable):

    # run ffft to identify the strength of the checkerboard effect
    ny, nx = np.shape(field)

    # the lattitudes
    x_1_lat = np.int(np.floor(2 * ny / 9))
    x_2_lat = np.int(np.floor(4 * ny / 9))
    x_3_lat = np.int(np.floor(6 * ny / 9))
    x_4_lat = np.int(np.floor(8 * ny / 9))
    x_lat = [x_1_lat, x_2_lat, x_3_lat, x_4_lat]

    ### FFT FOR STREAMFUNCTION ###
    # take strips of x
    x_1 = field[x_1_lat, :]
    x_2 = field[x_2_lat, :]
    x_3 = field[x_3_lat, :]
    x_4 = field[x_4_lat, :]

    # plot the FFT
    normalised_fft_freq(x_1, 'variable', '1')
    normalised_fft_freq(x_2, 'variable', '2')
    normalised_fft_freq(x_3, 'variable', '3')
    normalised_fft_freq(x_4, 'variable', '4')

def fft_high_freq(field, new_file):
    ny, nx = np.shape(field)

    for i in range(ny):
        x = field[i, :]
        print(np.shape(x))
        N = len(x)
        line_no = 1 + i
        signal_fft = rfft(x)
        signal_fft[150:] = 0
        plt.plot(2 * np.abs(signal_fft) / N)
        name = '$\psi$'
        plt.title(f'Spectrum for {name} for line {line_no}.')
        plt.ylim(0, 2500)
        reconstructed_signal = np.fft.irfft(signal_fft, n=N)
        plt.show()
        field[i, :] - reconstructed_signal

    fileid = netCDF4.Dataset(new_file, 'r+')
    fileid.variables['psi'][0] = field
    fileid.close()


if __name__ == '__main__':
    input_file = f"/projects/jodap/lrisley/GYRE/experiments/u-di463-invbal/balance_vel_to_psichi_to_vel_exp31.nc"
    lon, lat, time = read_file_info(input_file)
    psi = read_file(input_file, "psi")[0]
    #u = read_file(input_file, "u")[0]
    #u_recon = read_file(input_file, "u_inv")[0]
    #v = read_file(input_file, "v")[-3]
    #v_recon = read_file(input_file, "v_inv")[0]

    # calculate the raw error
    #u_err = u - u_recon
    #v_err = v - v_recon

    #fft_checkerboard(psi, 'sf')

    new_file = "/projects/jodap/lrisley/GYRE/experiments/u-di463-invbal/fft/high_freq_exp31.nc"
    fft_high_freq(psi, new_file)