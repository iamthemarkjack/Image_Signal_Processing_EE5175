import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('img1.png', cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()
keypoints = sift.detect(img, None)