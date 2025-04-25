import numpy as np
import matplotlib.pyplot as plt

from dfts import dft_2d, idft_2d

# load the image
I = plt.imread(r"fourier.png")

direct_rotated = np.rot90(I)

dft = np.fft.fft2(I)
rotated_dft = np.rot90(dft)
frequency_rotated = np.abs(np.fft.ifft2(rotated_dft))

plt.figure(figsize=(12, 4))
    
plt.subplot(131)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(direct_rotated, cmap='gray')
plt.title(f'Directly Rotated')
plt.axis('off')

plt.subplot(133)
plt.imshow(frequency_rotated, cmap='gray')
plt.title(f'DFT Rotation + IDFT')
plt.axis('off')

plt.tight_layout()
plt.show()