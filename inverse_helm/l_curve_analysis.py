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
from numpy.linalg import norm

def l_curve():
    exps = [27, 21, 31, 35, 37]
    label = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    log_reg = []
    log_residual = []
    #grays = np.linspace(0.3, 0.7, len(exps))  # Creates an array of grays from 0 (black) to 1 (white)
    # Create a dictionary to hold the arrays
    arrays = {}
    j = 0
    for i in exps:
        resid = np.loadtxt(f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/convergence_files/exp{i}_level1.txt',
            delimiter=",")
        exp_nc = f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/balance_vel_to_psichi_to_vel_exp{i}.nc'

        psi = read_file(exp_nc, "psi")[0]
        chi = read_file(exp_nc, "chi")[0]

        reg_result = np.append(psi, chi)
        log_reg.append(np.log(norm(reg_result)))
        log_residual.append(np.log(norm(resid)))

    plt.plot(log_residual, log_reg, marker='o')
    for i in range(len(label)):
        plt.text(log_residual[i], log_reg[i], rf"$10^{{{int(np.log10(label[i]))}}}$", bbox=dict(facecolor='yellow', alpha=0.5))
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.gca().invert_xaxis()
    plt.xlabel('Logarithm of the regularised solution')
    plt.ylabel(f'Logarithm of the residula norm')
    plt.title(f'L-curve analysis for accuracy tolerance = $10^{{{int(np.log10(1e-5))}}}$')
    plt.show()
if __name__ == '__main__':
    l_curve()