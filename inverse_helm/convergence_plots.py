import matplotlib.pyplot as plt
import numpy as np

def conv_plot():
    exps = [31, 35, 37]
    label = [1e-5, 1e-6, 1e-7]
    grays = np.linspace(0.3, 0.7, len(exps))  # Creates an array of grays from 0 (black) to 1 (white)
    # Create a dictionary to hold the arrays
    arrays = {}
    j=0
    for i in exps:
        arrays[f"exp_{i}"] = np.loadtxt(f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/convergence_files/exp{i}_level1.txt',
        delimiter=",")
        iterations = np.arange(0, np.shape(arrays[f"exp_{i}"])[0])
        print(f'Number of iterations for exp {i} = {iterations[-1]}')
        #plt.plot(iterations, arrays[f"exp_{i}"], label=f'Experiment {i}')
        plt.plot(iterations, arrays[f"exp_{i}"], color=str(grays[j]), label=rf"$10^{{{int(np.log10(label[j]))}}}$")
        j+=1
    plt.xlabel('Iterations')
    plt.ylabel(f'Norm of the residual')
    plt.yscale('log')
    plt.legend(title = 'Regularisation parameter')
    plt.title(f'Convergence with accuracy tolerance = $10^{{{int(np.log10(1e-5))}}}$')
    plt.show()



if __name__ == '__main__':
    conv_plot()