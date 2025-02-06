import numpy as np
import matplotlib.pyplot as plt

from affine_wrap import affine_transfrom

# reading the image as a numpy array
I = np.array(plt.imread('pisa_rotate.png'))
n, m = I.shape

theta = 0.06928957 # 3.97 degrees according to Wikipedia

# for zero-padding
n_ = int(np.ceil(n * abs(np.cos(theta)) + m * abs(np.sin(theta))))
m_ = int(np.ceil(n * abs(np.sin(theta)) + m * abs(np.cos(theta))))
x_c, y_c = n_ // 2, m_ // 2
x_c_old, y_c_old = n // 2, m // 2
zero_padded_I = np.zeros((n_, m_))

x_start, x_end = x_c - x_c_old, x_c - x_c_old + n
y_start, y_end = y_c - y_c_old, y_c - y_c_old + m
zero_padded_I[x_start:x_end, y_start:y_end] = I

# affine matrix : rotation plus translation of the center
T = np.array([[np.cos(theta), -np.sin(theta), -(np.cos(theta)-1)*x_c + y_c*np.sin(theta)],
              [np.sin(theta), np.cos(theta), -x_c*np.sin(theta) - (np.cos(theta)-1)*y_c],
              [0, 0, 1]])

I_out = affine_transfrom(zero_padded_I, T, n_, m_)
plt.imsave('Q2.png',I_out,cmap='gray')