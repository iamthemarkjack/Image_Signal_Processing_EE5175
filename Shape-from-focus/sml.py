import numpy as np
import matplotlib.pyplot as plt

from convolve import convolve
from affine_wrap import affine_transfrom

def sml(f, q):
    """
    Sum-Modified Laplacian
    """
    kernel_xx = np.array([[0, 0, 0],
                          [1, -2, 1],
                          [0, 0, 0]])

    kernel_yy = np.array([[0, 1, 0],
                          [0, -2, 0],
                          [0, 1, 0]])
    fxx = convolve(f, kernel_xx) 
    fyy = convolve(f, kernel_yy)
    ml = np.abs(fxx) + np.abs(fyy) # contains the ml values

    sml = np.zeros_like(f)
    for i in range(-q, q+1): # shifts along x axis
        T = np.array([[1, 0, i],
                      [0, 1, 0],
                      [0, 0, 1]])
        sml += affine_transfrom(ml, T, ml.shape[0], ml.shape[1])

    for j in range(-q, q+1):
        T = np.array([[1, 0, 0], # shifts along y axis
                      [0, 1, j],
                      [0, 0, 1]])
        sml += affine_transfrom(ml, T, ml.shape[0], ml.shape[1])

    return sml + 1e-12*np.eye(sml.shape[0]) # to avoid invalid scalar error in computing log