import numpy as np

def affine_transfrom(I, T, n_out, m_out):
    """
    Function to perform affine transformation T on I
    n_out, m_out : Desired shape of the output image
    """
    n, m = I.shape
    inv_T = np.linalg.inv(T)
    I_out = np.zeros((n_out,m_out))
    for x_ in range(n_out):
        for y_ in range(m_out):
            x, y, w = inv_T.dot(np.array([x_,y_,1]))
            x, y = x/w, y/w
            # bilinear interpolation
            x_1, y_1 = int(x), int(y)
            x_2, y_2 = int(x) + 1, int(y) + 1
            if 0 <= x_1 < n and 0 <= x_2 < n and 0 <= y_1 < m and 0 <= y_2 < m: 
                a, b = x - x_1, y - y_1
                val = (a)*(b)*I[x_2, y_2] + (1-a)*(b)*I[x_1, y_2] + (a)*(1-b)*I[x_2, y_2] + (1-a)*(1-b)*I[x_1, y_1]
            else:
                val = 0.0
            I_out[x_,y_] = val
    return I_out