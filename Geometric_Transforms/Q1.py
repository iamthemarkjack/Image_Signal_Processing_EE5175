import numpy as np
import matplotlib.pyplot as plt

from affine_wrap import affine_transfrom

# reading the image as a numpy array
I = np.array(plt.imread('lena_translate.png'))
n, m = I.shape

# affine matrix : translation
tx, ty = 3.75, 4.3
T = np.array([[1, 0, tx],
              [0, 1, ty],
              [0, 0, 1]])

I_out = affine_transfrom(I, T, n, m)
plt.imsave('Q1.png',I_out,cmap='gray')
