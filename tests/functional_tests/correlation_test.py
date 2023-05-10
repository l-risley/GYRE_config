""""
Test the correlation code created for two fields
"""
import numpy as np


def corr_test(n):
    """
    Find the correlations
    """
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = np.array([[2, 3, 4, 5], [6, 7, 8, 9]])

    # initialise arrays for correlations calculations
    sum_sq_x, sum_sq_y = np.zeros_like(x), np.zeros_like(y)
    sum_x, sum_y, = np.zeros_like(x), np.zeros_like(y)
    sum_xy = np.zeros_like(x)

    for time_index in range(n):  # range(num_times - 1):
        x += 1
        print(f'x = {x}')
        y += 1
        print(f'y = {y}')

        sum_sq_x += np.square(x)
        sum_sq_y += np.square(y)

        sum_x += x
        sum_y += y

        sum_xy += x * y

    ## calculate values of correlations
    # mean values
    mn_x, mn_y = sum_x/ n, sum_y / n
    # variances
    var_x = (sum_sq_x / n) - (np.square(mn_x))
    var_y = (sum_sq_y / n) - (np.square(mn_y))

    # covariances
    cov_x_y = (sum_xy / n) - (mn_x * mn_y)

    # correlations
    corr_x_y = cov_x_y / np.sqrt(var_x * var_y)

    ## plot correlation matrix
    # full model fields
    print(f'Correlation matrix of x and y  = {corr_x_y}')

if __name__ == '__main__':
    corr_test(10)