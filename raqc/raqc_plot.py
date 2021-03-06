
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
        pltz_obj.marks_colors()
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
    # print('type initial ', type(self.bins))
    # img = scipy.misc.toimage(self.bins, high=np.max(self.bins), low=np.min(self.bins), mode='I')
    # img.save('/home/zachuhlmann/projects/data/my16bit.png')
    #
    # # check that you got the same values
    # b = scipy.misc.imread('/home/zachuhlmann/projects/data/my16bit.png')
    # b.dtype
    # # dtype('int32')
    # print('do they equal ', np.array_equal(self.bins, b))
    # # True


# mat = self.trim_extent_nan('mat_diff_norm_nans')
# mat[~self.overlap_nan_trim] = np.nan
#
# # Sub3: Basin snow map
# h = axes[1,0].imshow(mat, origin = 'upper', cmap = pltz_obj.cmap_marks, norm = MidpointNormalize(midpoint = 0))
# axes[1,0].title.set_text('First date snow depth')
# cbar = plt.colorbar(h, ax = axes[1,0])
# cbar.set_label('relative diff (%)')
#
# # Sub4: Basin map of clipped snow
# mat = self.trim_extent_nan('flag_gain_block')
# mat[~self.overlap_nan_trim] = 0
# h = axes[1,1].imshow(mat, origin = 'upper')
# axes[1,1].title.set_text('locations of outliers (n=' + str(np.sum(self.flag_combined )) + ')')
# axes[1,1].set_xlabel('snow depth (m)')
# axes[1,1].set_ylabel('relative delta snow depth')
# self.save_tiff('SJ_multiband2_gain_enforced')
# # self.save_tiff('outliers_map_space', 'Lakes_06_11_05_01_outliers')
