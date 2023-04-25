"""

"""
import numpy as np

from transforms.U_transform import *
from transforms.T_transform import *
from transforms.Tik_regularisation import *


def test_vel_from_helm():
    sf = np.array([[0, 0, 0, 0], [0, 3, -1, 6], [0, 9, 6, 6], [0, 3, -2, 8]])
    vp = np.array([[2, 4, 6, 0], [4, 2, -3, 0], [7, -2, 9, 0], [0, 0, 0, 0]])
    dy, dx = 1, 1
    sf_y = np.array([[3, -1, 6], [6, 7, 0], [-6, -8, 2]])
    sf_x = np.array([[3, -4, 7], [9, -3, 0], [3, -5, 10]])
    vp_y = np.array([[2, -2, -9], [3, -4, 12], [-7, 2, -9]])
    vp_x = np.array([[2, 2, -6], [-2, -5, 3], [-9, 11, -9]])
    u_ans = -sf_y + vp_x
    v_ans = sf_x + vp_y
    u_res, v_res = vel_from_helm_gyre(sf, vp, dy, dx)
    assert (u_ans == u_res).all() == True
    assert (v_ans == v_res).all() == True