import numpy as np
import matplotlib.pyplot as plt

from affine_wrap import affine_transfrom

# computing the affine Matrix
A = np.array([[29, -124, 1, 0],
              [124, 29, 0, 1],
              [157, -372, 1, 0],
              [372, 157, 0, 1]])
b = np.array([93, 248, 328, 399])

a, b ,tx, ty = np.linalg.inv(A).dot(b)

# affine - Rotation plus translation matrix)
H = np.array([[a , -b, tx],
              [b, a, ty],
              [0, 0, 1]])

H_inv = np.linalg.inv(H)

# reading in both images
IMG1 = plt.imread("IMG1.png")
IMG2 = plt.imread("IMG2.png")

# applying inverse affine to IMG2
n, m = IMG1.shape
IMG2_rewarped = affine_transfrom(IMG2, H_inv, n , m)

# taking the difference between the two
diff = IMG2_rewarped - IMG1

# applying thresholding
threshold = 0.3
diff[diff < threshold] = 0.0

# saving the image
plt.imsave('output.png', diff, cmap='gray')