import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

from sift import sift
from perspective_wrap import perspective_transfrom

def compute_homography(img1, img2, eps):
    """
    To compute homography H s.t. img2 = H*img1
    """
    corresp1, corresp2 = sift(img1, img2)
    n = len(corresp1)
    # running ransac
    while True:
        consensus_idx = set()
        chosen_idx = np.array(random.sample(range(n), 4))
        unchosen_idx = np.array(list(set(range(n)) - set(chosen_idx)))
        chosen_pts_1, chosen_pts_2 = corresp1[chosen_idx], corresp2[chosen_idx]
        unchosen_pts_1, unchosen_pts_2 = corresp1[unchosen_idx], corresp2[unchosen_idx]
        # computing the homography
        A = []
        for i in range(4):
            x, y = chosen_pts_1[i,0], chosen_pts_1[i,1]
            x_, y_ = chosen_pts_2[i,0], chosen_pts_2[i,1]
            upper_row = [x, y, 1, 0, 0, 0, -x_*x, -x_*y, -x_]
            lower_row = [0, 0, 0, x, y, 1, -y_*x, -y_*y, -y_]
            A.append(upper_row)
            A.append(lower_row)
        A = np.array(A)
        _, _, Vh = np.linalg.svd(A)
        h = Vh[-1]
        H = h.reshape(3,3)

        # computing the error
        for i in unchosen_idx:
            x, y = corresp1[i][0], corresp1[i][1]
            x_, y_, z_ = H.dot(np.array([x, y, 1]))
            x_, y_ = x_/z_, y_/z_
            err = np.sqrt((corresp2[i][0]-x_)**2 + (corresp2[i][1]-y_)**2)
            if err < eps:
                consensus_idx.add(i)
        
        # termination : if the consensus set is large enough i.e. |C| > 0.8*|M|, C-> Consensus set, M-> Correspondence set
        if len(consensus_idx) > 0.7*n:
            break
    # a change could be made here by including all the consensus of the best fit to find the homography matrix
    return H