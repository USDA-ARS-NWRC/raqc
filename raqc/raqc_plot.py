
import matplotlib.pyplot as plt
from raqc import plotables as pltz
import time
import numpy as np
import scipy

def plot_basic(self, action, file_path_out):
    if action != None:
        # print('this is where matplotlib lives: ', matplotlib.matplotlib_fname())s
        pltz_obj = pltz.Plotables()
        pltz_obj.set_zero_colors(1)
        # print('in plot basic', matplotlib.get_backend())

        plt.close(0)
        fig, axes = plt.subplots(num = 0, nrows = 1, ncols = 2, figsize = (10,4))
        asp_ratio = np.min(self.bins.shape) / np.max(self.bins.shape)
        xedges, yedges = self.xedges, self.yedges
        minx, maxx = min(xedges), max(xedges)
        miny, maxy = min(yedges), max(yedges)


        # Sub1: overall 2D hist
        h = axes[0].imshow(self.bins, origin = 'lower', vmin=0.1, vmax = 1000, cmap = pltz_obj.cmap_choose,
             extent = (minx, maxx, miny, maxy), aspect = ((maxx - minx) / (maxy - miny)))
        cbar = plt.colorbar(h, ax = axes[0])
        cbar.set_label('bin count')
        axes[0].title.set_text('2D histogram')
        axes[0].set_xlabel('early date depth (m)')
        axes[0].set_ylabel('relative delta snow depth')

        # Sub2: clipped outliers
        h = axes[1].imshow(self.outliers_hist_space, origin = 'lower',
            extent = (minx, maxx, miny, maxy), aspect = ((maxx - minx) / (maxy - miny)))
        # axes[1].title.set_text('outlier bins w/mov wind thresh: ' + str(round(threshold_histogram_space[0],2)))
        axes[1].title.set_text('outliers')
        axes[1].set_xlabel('early date depth (m)')
        axes[1].set_ylabel('relative delta snow depth')
        if action == 'show':
            plt.show()
        elif action == 'save':
            plt.savefig(file_path_out, dpi=180)
    else:
        pass
