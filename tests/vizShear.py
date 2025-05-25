import numpy as np
import h5py
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

dat_filename_fmt = "/scratch2/anjiang/AMR/tests/ShearData/dat_shear_{:06d}.h5"
output_prefix    = "./ShearFigs/fig_shear_"

tid_start = 500
num_tids  = 50

tid_list = tid_start + np.arange(num_tids)


def main():
    for tid in tid_list:
        filename = dat_filename_fmt.format(tid)
        with h5py.File(filename, 'r') as dat:
            rho = dat["rho"][:,:]
            T   = dat["T"  ][:,:]
        temperature (  T, tid)
        numSchlieren(rho, tid)


def temperature(T, tid:int):
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(T), origin='lower', cmap="jet", extent=(0, 1, 0, 1))
    plt.gca().set_axis_off()
    plt.tight_layout()
    fig_name = output_prefix + "temperature_{:06d}.pdf".format(tid)
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    print("Saved \"{:s}\"".format(fig_name))



def numSchlieren(rho, tid:int, vmax=0.9):
    cmap = "gist_stern"
    N  = rho.shape[0]
    i  = np.arange(N)
    ip = (i + 1    ) % N
    im = (i - 1 + N) % N
    drdx = rho[ip, :] - rho[im, :]
    drdy = rho[:, ip] - rho[:, im]
    dr   = np.sqrt(drdx*drdx + drdy*drdy)
    dr[:] /= np.max(dr)
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(dr), origin='lower', cmap=cmap, vmin=0, vmax=vmax, extent=(0, 1, 0, 1))
    plt.gca().set_axis_off()
    plt.tight_layout()
    fig_name = output_prefix + "Schlieren_{:06d}.pdf".format(tid)
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    print("Saved \"{:s}\"".format(fig_name))


if __name__ == "__main__":
    main()
