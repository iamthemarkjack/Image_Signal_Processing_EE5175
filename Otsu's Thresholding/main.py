# Performs Otsu's thresholding by minimizing within class variance using Bayesion Optimization

import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skopt import gp_minimize
from skopt.space import Real

img1 = img_as_ubyte(plt.imread(r'palmleaf1.png'))
img2 = img_as_ubyte(plt.imread(r'palmleaf2.png'))

hist1, _ = np.histogram(img1, bins=256, range=(0,256), density=False)
hist2, _ = np.histogram(img2, bins=256, range=(0,256), density=False)

def varw(t, hist):
    t = int(t[0])
    t = max(1, min(t, 254))

    N = sum(hist)
    N1 = sum([hist[i] for i in range(t)])
    N2 = N - N1

    if N1 <= 0 or N2 <= 0:
        return 1e10

    mu1 = (1 / N1) * sum([i*hist[i] for i in range(t)])
    mu2 = (1 / N2) * sum([i*hist[i] for i in range(t+1, 256)])

    var1 = (1 / N1) * sum([(i - mu1)**2 * hist[i] for i in range(t)])
    var2 = (1 / N2) * sum([(i - mu2)**2 * hist[i] for i in range(t+1,256)])

    varw = (var1*N1 + var2*N2) / N

    return varw

obj1 = lambda t : varw(t, hist1)
obj2 = lambda t : varw(t, hist2)

space = [Real(1, 254, name='t')]

t1 = int(gp_minimize(obj1, space, n_calls=20, random_state=100).x[0])
t2 = int(gp_minimize(obj2, space, n_calls=30, random_state=100).x[0])

binimg1 = (img1 > t1).astype(int)
binimg2 = (img2 > t2).astype(int)

plt.imsave("result1.png", binimg1, cmap="gray")
plt.imsave("result2.png", binimg2, cmap="gray")

def plot_histogram_and_wcv(hist, threshold, filename):
    ts = np.arange(1, 255)
    wcv_vals = [varw([t], hist) for t in ts]
    optimal_wcv = varw([threshold], hist)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(ts, [hist[t] for t in ts], width=1, color='gray', label='Histogram')
    ax1.set_xlabel("Threshold (t)")
    ax1.set_ylabel("Histogram Frequency", color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    ax2 = ax1.twinx()
    ax2.plot(ts, wcv_vals, color='blue', label='Within-Class Variance')
    ax2.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    ax2.scatter([threshold], [optimal_wcv], color='red')
    ax2.set_ylabel("Within-Class Variance", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim([0, 5000])

    fig.suptitle("Histogram and Within-Class Variance vs Threshold")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_histogram_and_wcv(hist1, t1, "hist_wcv1.png")
plot_histogram_and_wcv(hist2, t2, "hist_wcv2.png")