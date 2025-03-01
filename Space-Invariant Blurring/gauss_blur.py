import numpy as np
import matplotlib.pyplot as plt

gaussian = lambda i, j, sigma : np.exp(-(i**2 + j**2)/(2*sigma**2))

def blur_kernel(sigma):
    if sigma == 0.0:
        return np.array([1])
    n = np.ceil(6*sigma + 1).astype(int) # size of the kernel matrix (odd number)
    c = n//2 # center coordinate
    kernel = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            kernel[i,j] = gaussian(i-c, j-c, sigma)
    # normalizing
    kernel = kernel/np.sum(kernel)
    return kernel
