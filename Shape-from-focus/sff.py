import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import multiprocessing as mp
import torch

from sml import sml

# loading the data
mat_file = sp.io.loadmat(r'stack.mat')
stack_np = np.array([value for key, value in mat_file.items() if 'frame' in key][1:], dtype=np.float32)

# saving few images from the stack
idx = [0, 26, 88]
imgs = [stack_np[i] for i in idx]
for i, img in enumerate(imgs):
    plt.imsave(f'results/stack_samples/sample_{idx[i]}.png', img, cmap='gray')

del_d = 50.50

q_vals = [0, 1, 2]

for q in q_vals:

    def compute_sml(i):
        return sml(stack_np[i], q)

    # compute the smls using multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        smls = pool.map(compute_sml, range(len(stack_np)))

    # saving few images from the smls
    smls_to_save = [smls[i] for i in idx]
    for i, img in enumerate(smls_to_save):
        plt.imsave(f'results/sml_samples/q{q}/sample_{idx[i]}.png', img, cmap='gray')

    # pushing the tensors to cuda for parallel processing
    smls = [torch.tensor(sml).cuda() for sml in smls]
    stack = torch.tensor(stack_np).cuda()

    Fs_stack = torch.stack(smls, dim=0)

    # finding F_m (The maximum value)
    Fm, peak_idx = torch.max(Fs_stack[1:-1, :, :], dim=0)
    peak_idx = peak_idx + 1

    # findig F_{m-1} and F_{m+1}
    Fmm1, Fmp1 = torch.zeros_like(Fm).cuda(), torch.zeros_like(Fm).cuda()
    for i in range(Fm.shape[0]):
        for j in range(Fm.shape[0]):
            Fmm1 = Fs_stack[peak_idx[i, j] - 1, i, j]
            Fmp1 = Fs_stack[peak_idx[i, j] + 1, i, j]

    dm = peak_idx.float() * del_d

    dmm1 = dm - del_d
    dmp1 = dm + del_d

    # guassian interpolation
    denominator = 2 * del_d * (2 * torch.log(Fm) - torch.log(Fmp1) - torch.log(Fmm1))
    depth = ((torch.log(Fm) - torch.log(Fmp1)) * (dm**2 - dmm1**2) - 
                (torch.log(Fm) - torch.log(Fmm1)) * (dm**2 - dmp1**2)) / denominator

    # wherever we get a NaN is a background point
    stack_idx = torch.round(depth / del_d).long()
    nan_mask = torch.isnan(depth)
    stack_idx[nan_mask] = 0

    # getting the all focused image and saving it
    all_focused = stack[stack_idx, torch.arange(stack.shape[1]).repeat(stack.shape[2], 1).T, torch.arange(stack.shape[1]).repeat(stack.shape[2], 1)]
    plt.imsave(f"results/all_focused/q{q}/all_focused.png", all_focused.cpu().numpy(), cmap='gray')

    # plotting and saving the 3d plot of depth
    Z = depth.cpu().numpy()
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(elev=62, azim=-60, roll=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Depth Map for q = {q}')
    plt.savefig(f"results/depth_map/q{q}/depth_map.png")
    plt.show()

    torch.cuda.empty_cache()