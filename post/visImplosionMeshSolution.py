#import matplotlib
#matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt


dat_filename_fmt = "../src/data/density_{:06}.dat"
msh_filename_fmt = "../src/data/mesh_{:06d}.dat"
fig_filename_fmt = "../figures2/fig_{:06d}.png"

domain_offset_x   = -0.3
domain_offset_y   = -0.3
num_patches       = 6
domain_bounds     = [(-0.3, 0.3), (-0.3, 0.3)]
base_patch_length = 12.0 / num_patches


def main():
    tid_list = np.arange(1, 3000, 8)
    # ./compile_pde_local_upsample.sh
    # mpirun build/test_pde_local_upsample 504 0.05 4 -ll:util 10 -ll:cpu 10
    for i in range(len(tid_list)):
        tid = tid_list[i]
        dat_filename = dat_filename_fmt.format(tid)
        msh_filename = msh_filename_fmt.format(tid)
        fig_filename = fig_filename_fmt.format(i)
        generateVis(msh_filename, dat_filename, fig_filename)


def generateVis(input_file_msh, input_file_dat, fig_file):
    plt.figure(figsize=(15, 12))
    generateVisData(input_file_dat, vmin=None, vmax=None)
    generateVisMesh(input_file_msh)
    plt.gca().set_aspect(1.0)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xlim(domain_bounds[0][0], domain_bounds[0][1])
    plt.ylim(domain_bounds[1][0], domain_bounds[1][1])
    plt.tight_layout()
    plt.savefig(fig_file, bbox_inches='tight')
    plt.close()
    print("Generated file \"{:s}\".".format(fig_file))

def generateVisMesh(input_file_msh):
    pid, level, i_coord, j_coord, i_prev, i_next, j_prev, j_next, parent, child0, child1, child2, child3, r_req, c_req \
    = np.loadtxt(input_file_msh, delimiter=',', skiprows=1, unpack=True, dtype=int)

    txt_color = ['b', 'r', 'g', 'y']
    for i in range(len(level)):
        drawPatch(pid[i], level[i], i_coord[i], j_coord[i], i_prev[i], i_next[i], j_prev[i], j_next[i], text_color=txt_color[level[i]])



def drawPatch(pid:int, level:int, i_coord:int, j_coord:int, i_prev:int, i_next:int, j_prev:int, j_next:int, linsty='c-', text_color='b'):
    delta = base_patch_length * (0.5 ** level)
    x     = i_coord * delta + domain_offset_x
    y     = j_coord * delta + domain_offset_y
    plt.plot([x      , x+delta], [y      , y      ], linsty)
    plt.plot([x      , x+delta], [y+delta, y+delta], linsty)
    plt.plot([x      , x      ], [y      , y+delta], linsty)
    plt.plot([x+delta, x+delta], [y      , y+delta], linsty)
    #plt.annotate("pid={:d}\ni_nbr=({:d}, {:d})\nj_nbr=({:d}, {:d})".format(pid, i_prev, i_next, j_prev, j_next), \
    #        (x+0.5*delta, y+0.5*delta), color=text_color, horizontalalignment='center', verticalalignment='center', fontsize=12 * delta)
    


def generateVisData(data_file, cmap='gnuplot2_r', vmin=None, vmax=None, rasterized=False):
    u_patch, v_patch, T_patch, p_patch, x_patch, y_patch, levels = loadData(data_file, patch_size = 16)
    f_patch = p_patch / T_patch
    if vmin is None:
        vmin = np.min(f_patch)
    if vmax is None:
        vmax = np.max(f_patch)

    print(f_patch.shape)
    print(vmin, vmax)

    level_max = int(np.max(levels) + 1.1)
    # Which levels to visualize
    level_list = np.arange(0, 1)
    for l in level_list:
        pid_level = np.where(np.int32(levels) == l)[0]
        for pid in pid_level:
            plt.pcolormesh(x_patch[pid], y_patch[pid], f_patch[pid], cmap=cmap, vmin=vmin, vmax=vmax, rasterized=rasterized)
    plt.colorbar()


def loadData(filename : str, patch_size : int = 16):
    color, level, p_coord_i, p_coord_j, loc_i, loc_j, x_raw, y_raw, u_raw, v_raw, T_raw, p_raw = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)
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

    return u_patch, v_patch, T_patch, p_patch, x_patch, y_patch, level[::256]



if __name__ == "__main__":
    main()

