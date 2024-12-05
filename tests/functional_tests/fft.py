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
    fourier = fft(signal)
    # Calculate N/2 to normalize the FFT output
    N = len(signal)
    normalize = N / 2
    # Get the frequency components of the spectrum
    sampling_rate = 100.0  # It's used as a sample spacing
    frequency_axis = fftfreq(N, d=1.0 / sampling_rate)
    norm_amplitude = np.abs(fourier) / normalize
    # Plot the results
    #plt.plot(frequency_axis, norm_amplitude)
    #plt.xlabel('Frequency[Hz]')
    #plt.ylabel('Amplitude')
    #plt.title(f'Spectrum for {signal_name}')
    #plt.show()
    #plt.plot(frequency_axis)
    #plt.ylabel('Frequency[Hz]')
    #plt.title(f'Frequency Axis for {signal_name}')
    #plt.show()
    # Plot the actual spectrum of the signal
    plt.plot(rfftfreq(N, d=1 / sampling_rate), 2 * np.abs(rfft(signal)) / N)
    plt.title(f'Spectrum for {signal_name}')
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude')
    plt.show()

def contour_fft_lines(x, y, z, fft, plot_of: str, variable_name: str):
    # 2D contour plot of one variable
    # switch coords from m to km
    plt.pcolormesh(x, y, z, cmap='viridis', shading='auto', vmin=-15000, vmax=15000)
    # ax = sns.heatmap(z, cmap = 'ocean')
    plt.xlabel('Longitude ($^\circ$)')
    plt.ylabel('Lattitude ($^\circ$)')
    plt.title(f'{plot_of}')
    if variable_name == 'Elevation':
        units = '$m$'
    elif variable_name == 'SF' or variable_name == 'VP':
        units = '$m^2 s^{-1}$'
    else:
        units = '$ms^{-1}$'
    plt.colorbar(label=f'{variable_name} ({units})')
    # line plot of where the fft is applied
    #y_1, y_2, y_3 = [y[fft[0]]]*len(x), [y[fft[1]]]*len(x), [y[fft[2]]]*len(x)
    x_1, x_2, x_3 = x[fft[0], :], x[fft[1], :], x[fft[2], :]
    y_1, y_2, y_3 = y[fft[0], :], y[fft[1], :], y[fft[2], :]
    plt.plot(x_1, y_1, c='k', label='Line 1')
    plt.plot(x_2, y_2, c='y', label='Line 2')
    plt.plot(x_3, y_3, c='w', label='Line 3')
    #plt.savefig(f'plots/{plot_of}{variable_name}.png')
    plt.legend()
    plt.show()

def fft_checkerboard(exp_no):
    # input psi field
    # netcdf file locations
    exp_input_file = f"/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{exp_no}.nc"

    # lon and lat for each grid
    lon, lat, time = read_file_info(exp_input_file)

    # gyre12 outputs
    psi = read_file(exp_input_file, "psi")[0]

    # run ffft to identify the strength of the checkerboard effect
    ny, nx = np.shape(psi)

    # the lattitudes
    x_1_lat = np.int(np.floor(ny / 3))
    x_2_lat = np.int(np.floor(ny / 2))
    x_3_lat = np.int(np.floor(2 * ny / 3))
    x_lat = [x_1_lat, x_2_lat, x_3_lat]

    # take strips of x
    x_1 = psi[x_1_lat, :]
    x_2 = psi[x_2_lat, :]
    x_3 = psi[x_3_lat, :]

    # plot the field with the lines on top
    contour_fft_lines(lon, lat, psi, x_lat,'Streamfunction - no filter', 'SF')

    # plot the FFT
    # normalised_fft(x_1, f'{input} at {x_1_lat * 10} km lattitude.')
    normalised_fft_freq(x_1, f'$\psi$ for line 1.')
    # normalised_fft(x_2, f'{input} at {x_2_lat * 10} km lattitude.')
    normalised_fft_freq(x_2, f'$\psi$ for line 2.')
    # normalised_fft(x_3, f'{input} at {x_3_lat * 10} km lattitude.')
    normalised_fft_freq(x_3, f'$\psi$ at for line 3.')

if __name__ == '__main__':
    fft_checkerboard(7)