import matplotlib
matplotlib.use("agg")
import numpy as np
import matplotlib.pyplot as plt


tid_start = 1
num_steps = 100
stride = 10

input_file_fmt = "../src/density_{:06d}.dat"
fig_name_fmt = "field_vis_{:06d}.png"

def main():
    tid_start_loop = 0
    for i in [0] + list(range(num_steps)):
        tid = i * stride + tid_start_loop
        tid_start_loop = tid_start
        u_patch, v_patch, T_patch, p_patch, x_patch, y_patch = loadData(input_file_fmt.format(tid))
        f = u_patch - 0.5
        vmax = np.max(np.abs(f))
        print(vmax)
        plotData(f, x_patch, y_patch, figname=fig_name_fmt.format(tid), cmap="seismic", vmax=vmax, vmin=-vmax)
        #entropy_const = p_patch / ((p_patch/T_patch) ** 1.4)
        #vmax, vmin = np.max(entropy_const), np.min(entropy_const)
        #print(vmax, vmin, vmax - vmin)
        #plotData(entropy_const, x_patch, y_patch, figname=fig_name_fmt.format(tid), vmax=vmax, vmin=vmin)
    


def plotData(f_patch, x_patch, y_patch, figname="field_vis.png", cmap='gnuplot2_r', vmin=None, vmax=None, rasterized=False, figsize=None):
    if vmin is None:
        vmin = np.min(f_patch)
    if vmax is None:
        vmax = np.max(f_patch)

    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    for pid in range(f_patch.shape[0]):
        plt.pcolormesh(x_patch[pid], y_patch[pid], f_patch[pid], cmap=cmap, vmin=vmin, vmax=vmax, rasterized=rasterized)
    plt.colorbar()
    plt.grid(True)
    plt.gca().set_aspect(1)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close()


def loadData(filename : str, patch_size : int = 16):
    color, p_coord_i, p_coord_j, loc_i, loc_j, x_raw, y_raw, u_raw, v_raw, T_raw, p_raw = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)
    assert (len(color) % (patch_size * patch_size)) == 0, f"Data size {len(color)} is incompatible with the specified patch size {patch_size}."
    num_patches = len(color) // (patch_size * patch_size)
    u_patch = np.empty((num_patches, patch_size, patch_size), dtype=np.dtype(np.float64, align=True))
    v_patch = np.empty((num_patches, patch_size, patch_size), dtype=np.dtype(np.float64, align=True))
    T_patch = np.empty((num_patches, patch_size, patch_size), dtype=np.dtype(np.float64, align=True))
    p_patch = np.empty((num_patches, patch_size, patch_size), dtype=np.dtype(np.float64, align=True))
    x_patch = np.empty((num_patches, patch_size, patch_size), dtype=np.dtype(np.float64, align=True))
    y_patch = np.empty((num_patches, patch_size, patch_size), dtype=np.dtype(np.float64, align=True))
    patch_coord_i = np.empty((num_patches, ), dtype=np.dtype(np.int32, align=True))
    patch_coord_j = np.empty((num_patches, ), dtype=np.dtype(np.int32, align=True))
    num_elem_per_patch = patch_size * patch_size
    for pid in range(num_patches):
        idx_start = pid * num_elem_per_patch
        u_patch[pid, :, :] = u_raw[idx_start:idx_start + num_elem_per_patch].reshape((patch_size, patch_size))
        v_patch[pid, :, :] = v_raw[idx_start:idx_start + num_elem_per_patch].reshape((patch_size, patch_size))
        T_patch[pid, :, :] = T_raw[idx_start:idx_start + num_elem_per_patch].reshape((patch_size, patch_size))
        p_patch[pid, :, :] = p_raw[idx_start:idx_start + num_elem_per_patch].reshape((patch_size, patch_size))
        x_patch[pid, :, :] = x_raw[idx_start:idx_start + num_elem_per_patch].reshape((patch_size, patch_size))
        y_patch[pid, :, :] = y_raw[idx_start:idx_start + num_elem_per_patch].reshape((patch_size, patch_size))
        patch_coord_i[pid] = p_coord_i[idx_start]
        patch_coord_j[pid] = p_coord_j[idx_start]

    return u_patch, v_patch, T_patch, p_patch, x_patch, y_patch


if __name__ == "__main__":
    main()