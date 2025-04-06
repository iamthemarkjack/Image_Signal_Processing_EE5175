import numpy as np
import matplotlib.pyplot as plt

from dfts import dft_2d, idft_2d

# importing the images
I1 = plt.imread("fourier.png")
I2 = plt.imread("fourier_transform.png")

# compute the 2d DFTs
dft_1 = dft_2d(I1)
dft_2 = dft_2d(I2)

# get the magnitudes and phases
mag1, mag2 = np.abs(dft_1), np.abs(dft_2)
phase1, phase2 = np.angle(dft_1), np.angle(dft_2)

# dfts of new images
dft_3 = mag1 * np.exp(1j * phase2)
dft_4 = mag2 * np.exp(1j * phase1)

# reconstructing the images
I3 = idft_2d(dft_3)
I4 = idft_2d(dft_4)

# saving the images
plt.imsave('outs/I3.png', I3.real, cmap='gray')
plt.imsave('outs/I4.png', I4.real, cmap='gray')