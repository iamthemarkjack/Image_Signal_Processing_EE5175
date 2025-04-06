import numpy as np 

def dft_2d(f):
    """
    Compute 2D DFTs using 1D FFTs in Row-Column Fashion
    """
    F = np.apply_along_axis(np.fft.fft, axis=1, arr=f)
    F = np.apply_along_axis(np.fft.fft, axis=0, arr=F)
    return F

def idft_2d(f):
    """
    Compute 2D IDFTs using 1D IFFTs in Row-Column Fashion
    """
    F = np.apply_along_axis(np.fft.ifft, axis=1, arr=f)
    F = np.apply_along_axis(np.fft.ifft, axis=0, arr=F)
    return F