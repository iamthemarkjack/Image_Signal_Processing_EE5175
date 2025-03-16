import numpy as np
import matplotlib.pyplot as plt

from gauss_blur import blur_kernel

# function for spave variant blurring
def space_variant_blur(I, sigma_map):
    """
    I is a numpy array and sigma_map is a function of spatial coordinates
    """
    n, m = I.shape

    # finding the kernel shape at the corners(assuming the sigma_map to be symmetric from the center of the image)
    sigma_corner = sigma_map(0,0)
    k_corner = np.ceil(6*sigma_corner + 1).astype(int)
    if k_corner % 2 == 0:
        k_corner += 1

    # zero padding 
    padded_I = np.zeros((n + k_corner, m + k_corner))
    final_out = np.zeros((n, m))
    padded_I[k_corner//2 + 1 : n + k_corner//2 + 1, k_corner//2 + 1 : m + k_corner//2 + 1] = I

    for i in range(k_corner//2 + 1, n + k_corner//2 + 1):
        for j in range(k_corner//2 + 1, m + k_corner//2 + 1):
            # coordinate in the original image
            x_coord, y_coord = i - k_corner//2 - 1, j - k_corner//2 - 1

            # finding the kernel size at this coordinate  
            sigma_val = sigma_map(x_coord, y_coord)
            kernel_size = np.ceil(6*sigma_val + 1).astype(int)
            if kernel_size % 2 == 0:
                kernel_size += 1

            patch = padded_I[i - kernel_size//2 : i + kernel_size//2 + 1, j - kernel_size//2 : j + kernel_size//2 + 1]
            kernel = blur_kernel(sigma_val)

            val = np.dot(np.ravel(patch), np.ravel(kernel))
            final_out[x_coord, y_coord] = val
    return final_out

# function for space invariant blurring
def convole(I, kernel):
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

# ------ QUESTION 1 --------

# reading in the image
img1 = plt.imread('Globe.png')

# parameters of the sigma map
N, _ = img1.shape
A = 2.0
B = (N**2)/(2*np.log(100*A))

sigma_map = lambda x,y : A*np.exp(-((x-N/2)**2 + (y-N/2)**2)/B)

img1_blurred = space_variant_blur(img1, sigma_map)
plt.imsave('Globe_blurred.png', img1_blurred, cmap='gray')


# ------ QUESTION 2 --------

# reading in the image
img2 = plt.imread('Nautilus.png')

# USING SPACE VARIANT BLURRING

# defining the sigma map
sigma_map = lambda x,y : 1.0

img2_svb = space_variant_blur(img2, sigma_map)
plt.imsave('Nautilus_Space_variant.png', img2_svb, cmap='gray')

# USING SPACE INVARIANT BLURRING
kernel = blur_kernel(1.0)

img2_sivb = convole(img2, kernel)
plt.imsave('Nautilus_Space_invariant.png', img2_sivb, cmap='gray')

# plotting the difference
plt.imsave('Diff.png', np.abs(img2_svb-img2_sivb), cmap='gray')