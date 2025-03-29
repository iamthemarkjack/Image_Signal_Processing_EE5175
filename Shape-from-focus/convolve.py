import numpy as np

def convolve(I, kernel):
    """
    Both I and kernel are numpy darrays
    """
    n, m = I.shape
    k = kernel.shape[0]
    padded_I = np.zeros((n+k, m+k))
    final_out = np.zeros((n, m))
    padded_I[k//2 + 1 : n+k//2 + 1, k//2 + 1 : m+k//2 + 1] = I
    for i in range(k//2 + 1, n+k//2 + 1):
        for j in range(k//2 + 1, m+k//2 + 1):
            patch = padded_I[i - k//2 : i + k//2 + 1, j - k//2 : j + k//2 + 1]
            val = np.dot(np.ravel(patch), np.ravel(kernel))
            final_out[i - k//2 - 1, j - k//2 - 1] = val
    return final_out