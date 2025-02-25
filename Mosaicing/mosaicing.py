import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np

from compute_homography import compute_homography
from perspective_wrap import perspective_transfrom

# importing the three images
img1 = cv.imread('img1.png',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('img2.png',cv.IMREAD_GRAYSCALE)
img3 = cv.imread('img3.png',cv.IMREAD_GRAYSCALE)

h1, w1 = img1.shape
h2, w2 = img2.shape
h3, w3 = img3.shape

# computing the homography matrices
eps = 1
H21 = compute_homography(img2, img1, eps)
H23 = compute_homography(img2, img3, eps)

# calculating the bounds of the canvas in img2 coordinate system
corners1 = np.array([[0, 0, 1],
                     [0, w1-1, 1],
                     [h1-1, 0, 1],
                     [h1-1, w1-1, 1]]).T

corners3 = np.array([[0, 0, 1],
                     [0, w3-1, 1],
                     [h3-1, 0, 1],
                     [h3-1, w3-1, 1]]).T

# transforming the corners to img2 coordinate system
corners1_in_2 = np.zeros_like(corners1)
for i in range(4):
    p = np.dot(np.linalg.inv(H21), corners1[:, i])
    corners1_in_2[:, i] = p / p[2]

corners3_in_2 = np.zeros_like(corners3)
for i in range(4):
    p = np.dot(np.linalg.inv(H23), corners3[:, i])
    corners3_in_2[:, i] = p / p[2]

# finding the bounds
all_corners_in_2 = np.hstack([corners1_in_2,
                         np.array([[0, 0, h2-1, h2-1], [0, w2-1, 0, w2-1], [1, 1, 1, 1]]),
                         corners3_in_2])

min_x = np.floor(np.min(all_corners_in_2[0] / all_corners_in_2[2])).astype(int)
max_x = np.ceil(np.max(all_corners_in_2[0] / all_corners_in_2[2])).astype(int)
min_y = np.floor(np.min(all_corners_in_2[1] / all_corners_in_2[2])).astype(int)
max_y = np.ceil(np.max(all_corners_in_2[1] / all_corners_in_2[2])).astype(int)

height = max_x - min_x + 1
width = max_y - min_y + 1

C = np.zeros((height, width))

weight1 = np.zeros((height, width), dtype=np.float32)
weight2 = np.zeros((height, width), dtype=np.float32)
weight3 = np.zeros((height, width), dtype=np.float32)

# filling the canvas with target to source mapping using bilinear interpolation
for i in range(height):
    for j in range(width):
        # point in img1
        x, y, z = H21.dot(np.array([i + min_x, j + min_y, 1]))
        x, y = x/z, y/z
        x_1, y_1 = int(x), int(y)
        x_2, y_2 = x_1 + 1, y_1 + 1
        if 0 <= x_1 < h1 and 0 <= x_2 < h1 and 0 <= y_1 < w1 and 0 <= y_2 < w1:
            f = np.array([img1[x_1,y_1], img1[x_1,y_2], img1[x_2,y_1], img1[x_2,y_2]])
            N = np.array([[1, x_1, y_1, x_1*y_1],
                        [1, x_1, y_2, x_1*y_2],
                        [1, x_2, y_1, x_2*y_1],
                        [1, x_2, y_2, x_2*y_2]])
            a = np.linalg.inv(N).dot(f)
            val = np.dot(a,np.array([1, x, y, x*y]))
            C[i,j] += val
            weight1[i,j] = 1

        # point in img2
        x, y, z = np.array([i + min_x, j + min_y, 1])
        x, y = x/z, y/z
        x_1, y_1 = int(x), int(y)
        x_2, y_2 = x_1 + 1, y_1 + 1
        if 0 <= x_1 < h2 and 0 <= x_2 < h2 and 0 <= y_1 < w2 and 0 <= y_2 < w2:
            f = np.array([img2[x_1,y_1], img2[x_1,y_2], img2[x_2,y_1], img2[x_2,y_2]])
            N = np.array([[1, x_1, y_1, x_1*y_1],
                        [1, x_1, y_2, x_1*y_2],
                        [1, x_2, y_1, x_2*y_1],
                        [1, x_2, y_2, x_2*y_2]])
            a = np.linalg.inv(N).dot(f)
            val = np.dot(a,np.array([1, x, y, x*y]))
            C[i,j] += val
            weight2[i,j] = 1

        # point in img3
        x, y, z = H23.dot(np.array([i + min_x, j + min_y, 1]))
        x, y = x/z, y/z
        x_1, y_1 = int(x), int(y)
        x_2, y_2 = x_1 + 1, y_1 + 1
        if 0 <= x_1 < h3 and 0 <= x_2 < h3 and 0 <= y_1 < w3 and 0 <= y_2 < w3:
            f = np.array([img3[x_1,y_1], img3[x_1,y_2], img3[x_2,y_1], img3[x_2,y_2]])
            N = np.array([[1, x_1, y_1, x_1*y_1],
                        [1, x_1, y_2, x_1*y_2],
                        [1, x_2, y_1, x_2*y_1],
                        [1, x_2, y_2, x_2*y_2]])
            a = np.linalg.inv(N).dot(f)
            val = np.dot(a,np.array([1, x, y, x*y]))
            C[i,j] += val
            weight3[i,j] = 1

tot_weight = weight1 + weight2 + weight3
valid_idx = tot_weight > 0
C[valid_idx] /= tot_weight[valid_idx]

plt.imsave('assign_out.png', C, cmap='gray')