import matplotlib.pyplot as plt
import numpy as np

def conv_plot():
    exps = [14, 16, 18, 17]
    # Create a dictionary to hold the arrays
    arrays = {}
    for i in exps:
        arrays[f"exp_{i}"] = np.loadtxt(f'/Users/tk815965/OneDrive - University of Reading/Data_Assimilation/GYRE_config/inverse_helm/convergence_files/exp{i}_level1.txt',
        delimiter=",")
        iterations = np.arange(0, np.shape(arrays[f"exp_{i}"])[0])
        print(f'Number of iterations for exp {i} = {iterations[-1]}')
        plt.plot(iterations, arrays[f"exp_{i}"], label=f'Experiment {i}')
    plt.xlabel('Iterations')
    plt.ylabel(f'Convergence')
    plt.yscale('log')
    plt.legend()
    plt.title(f'Convergence at level 1 (tol=1e-5)')
    plt.show()



if __name__ == '__main__':
    conv_plot()