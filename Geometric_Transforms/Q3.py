import numpy as np
import matplotlib.pyplot as plt

from affine_wrap import affine_transfrom

# reading the image as a numpy array
I = np.array(plt.imread('cells_scale.png'))
n, m = I.shape

# affine matrix : scaling
x_scale, y_scale = 0.8 ,1.3
n_out, m_out = int(np.ceil(x_scale*n)), int(np.ceil(y_scale*m))
T = np.array([[x_scale, 0, 0],
              [0, y_scale, 0],
              [0, 0, 1]])
              
I_out = affine_transfrom(I, T, n_out, m_out)
plt.imsave('Q3.png', I_out, cmap='gray')