import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

input_file_fmt = "../src/output_meta_{:s}_stage_{:d}.dat"
fig_file_fmt   = "../src/output_meta_{:s}_stage_{:d}.pdf"

def main():
    generateVis(input_file_fmt.format("refine", 0), fig_file_fmt.format("refine", 0))
    generateVis(input_file_fmt.format("refine", 1), fig_file_fmt.format("refine", 1))
    generateVis(input_file_fmt.format("coarsen", 1), fig_file_fmt.format("coarsen", 1))
    generateVis(input_file_fmt.format("coarsen", 2), fig_file_fmt.format("coarsen", 2))
    generateVis(input_file_fmt.format("further_refine", 0), fig_file_fmt.format("further_refine", 0))
    generateVis(input_file_fmt.format("further_refine", 1), fig_file_fmt.format("further_refine", 1))
    generateVis(input_file_fmt.format("further_coarsen", 1), fig_file_fmt.format("further_coarsen", 1))
    generateVis(input_file_fmt.format("further_coarsen", 2), fig_file_fmt.format("further_coarsen", 2))


def generateVis(input_file, fig_file):
    pid, level, i_coord, j_coord, i_prev, i_next, j_prev, j_next, parent, child0, child1, child2, child3, r_req, c_req \
    = np.loadtxt(input_file, delimiter=',', skiprows=1, unpack=True, dtype=int)

    plt.figure(figsize=(15, 15))
    txt_color = ['b', 'r', 'g', 'y']
    for i in range(len(level)):
        drawPatch(pid[i], level[i], i_coord[i], j_coord[i], i_prev[i], i_next[i], j_prev[i], j_next[i], text_color=txt_color[level[i]])
    plt.gca().set_aspect(1.0)
    plt.tight_layout()
    plt.savefig(fig_file, bbox_inches='tight')
    plt.close()



def drawPatch(pid:int, level:int, i_coord:int, j_coord:int, i_prev:int, i_next:int, j_prev:int, j_next:int, linsty='k-', text_color='b'):
    delta = 0.5 ** level
    x     = i_coord * delta
    y     = j_coord * delta
    plt.plot([x      , x+delta], [y      , y      ], linsty)
    plt.plot([x      , x+delta], [y+delta, y+delta], linsty)
    plt.plot([x      , x      ], [y      , y+delta], linsty)
    plt.plot([x+delta, x+delta], [y      , y+delta], linsty)
    plt.annotate("pid={:d}\ni_nbr=({:d}, {:d})\nj_nbr=({:d}, {:d})".format(pid, i_prev, i_next, j_prev, j_next), \
            (x+0.5*delta, y+0.5*delta), color=text_color, horizontalalignment='center', verticalalignment='center', fontsize=12 * delta)
    



if __name__ == "__main__":
    main()
