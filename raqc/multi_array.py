from raqc.raqc_plot import plot_basic
import numpy as np
import rasterio as rio
import h5py
import copy
import math
from sklearn.feature_extraction import image
import sys, os
from subprocess import run
from netCDF4 import Dataset
import numpy_groupies as npg
import time
import pandas as pd
import matplotlib

class MultiArrayOverlap(object):
    def __init__(self, file_path_dataset1, file_path_dataset2, file_path_topo, file_out_root, file_name_modifier):
        # First check if user passed already clipped repeat array file paths

        string_match = 'clipped_to'
        self.already_clipped =  (string_match in file_path_dataset1) & (string_match in file_path_dataset2)
        # self.already_clipped2 = 'common_extent.tif' == file_path_dataset2[file_path_dataset2.index('common'):]
        # save file paths needed for clipping

        check_chronology1 = pd.to_datetime(file_path_dataset1.split('/')[-1].split('_')[0][-8:], format = '%Y%m%d')
        check_chronology2 = pd.to_datetime(file_path_dataset2.split('/')[-1].split('_')[0][-8:], format = '%Y%m%d')

        if check_chronology1 < check_chronology2:
            pass
        else:
            sys.exit("Date 1 must occur before Date 2. Exiting program")

        if not self.already_clipped:
            self.file_path_dataset1 = file_path_dataset1
            self.file_path_dataset2 = file_path_dataset2
            self.file_path_topo = file_path_topo
            self.file_out_root = file_out_root

        # Grab arrays and spatial metadata
        with rio.open(file_path_dataset1) as src:
            self.d1 = src
            self.meta = self.d1.profile
            if self.already_clipped:
                mat_clip1 = self.d1.read()  #matrix
                mat_clip1 = mat_clip1[0]
                mat_clip1[np.isnan(mat_clip1)] = -9999
                self.mat_clip1 = mat_clip1
        with rio.open(file_path_dataset2) as src:
            self.d2 = src
            self.meta2 = self.d2.profile
            if self.already_clipped:
                mat_clip2 = self.d2.read()  #matrix
                mat_clip2 = mat_clip2[0]
                mat_clip2[np.isnan(mat_clip2)] = -9999
                self.mat_clip2 = mat_clip2
        if self.already_clipped:
            with rio.open(file_path_topo) as src:
                topo = src
                topo_clip = topo.read()
                topo_clip = topo_clip[0]
                topo_clip[np.isnan(topo_clip)] = -9999
                self.topo_clip = topo_clip

        year1 = file_path_dataset1.split('/')[-1].split('_')[0]
        year2 = file_path_dataset2.split('/')[-1].split('_')[0][-8:]
        if file_out_root[-1] != '/':  #ensure user added forward slash
            file_out_root = file_out_root + '/'
        self.file_path_out_root = '{0}{1}_to_{2}'.format(file_out_root, year1, year2, file_name_modifier)
        self.file_path_out = '{0}_{1}.tif'.format(self.file_path_out_root, file_name_modifier)

    def clip_extent_overlap(self):
        """
        finds overlapping extent of two geotiffs. Saves as attributes clipped versions
        of both matrices extracted from geotiffs, and clipped matrices with -9999 replaced
        with nans (mat_clip1, mat_clip2 and mat_clip_nans of each).
        """
        meta = self.meta  #metadata
        d1 = self.d1
        d2 = self.d2
        #grab basics
        rez = meta['transform'][0]  # spatial resolution
        rez2 = self.meta2['transform'][0]

        #find topo file info
        topo = Dataset(self.file_path_topo)
        x = topo.variables['x'][:]
        y = topo.variables['y'][:]
        topo_extents = [None] * 4
        topo_extents[0], topo_extents[1], topo_extents[2], topo_extents[3], = x.min(), y.min(), x.max(), y.max()
        rez3 = x[1] - x[0]

        # check that resolutions are the same.  Note: if rez not a whole number, it will be rounded and rasters aligned
        if round(rez) == round(rez2):  # janky way to check that all three rez are the same
            pass
        else:
            sys.exit("check that spatial resolution of your two repeat array files are the same \
            must fix and try again")
        if round(rez) == round(rez3):
            topo_rez_same = True
        else:
            print('the resolution of your topo.nc file differs from repeat arrays.  It will be resized to \
            fit the resolution repeat arrays.  your input file will NOT be changed')
            topo_rez_same = False


        # grab bounds of common/overlapping extent and prepare function call for gdal to clip to extent and align
        left_max_bound = max(d1.bounds.left, d2.bounds.left, topo_extents[0])
        bottom_max_bound = max(d1.bounds.bottom, d2.bounds.bottom, topo_extents[1])
        right_min_bound =  min(d1.bounds.right, d2.bounds.right, topo_extents[2])
        top_min_bound = min(d1.bounds.top, d2.bounds.top, topo_extents[3])
        # ensure nothing after decimal - nice whole number, admittedly a float
        left_max_bound = left_max_bound - (left_max_bound % round(rez))
        bottom_max_bound = bottom_max_bound + (round(rez) - bottom_max_bound % round(rez))
        right_min_bound = right_min_bound - (right_min_bound % round(rez))
        top_min_bound = top_min_bound + (round(rez) - top_min_bound % round(rez))


        # file_name_dataset1_te_temp = os.path.splitext(os.path.expanduser(self.file_path_dataset1).split('/')[-1])[0] + '_common_extent.tif'
        # file_name_dataset1_te = self.file_out_root + file_name_dataset1_te_temp
        # file_name_dataset2_te_temp = os.path.splitext(os.path.expanduser(self.file_path_dataset2).split('/')[-1])[0]
        # file_name_dataset2_te = self.file_out_root + file_name_dataset2_te_temp
        file_name_dataset1_te_temp = os.path.splitext(os.path.expanduser(self.file_path_dataset1).split('/')[-1])[0]   #everythong after last / without filename ext (i.e. .tif)
        id_date_start = file_name_dataset1_te_temp.index('2')  #find index of date start in file name i.e. find idx of '2' in 'USCATE2019...'
        file_name_dataset1_te_first = os.path.splitext(file_name_dataset1_te_temp)[0][:id_date_start + 8]
        file_name_dataset1_te_second = os.path.splitext(file_name_dataset1_te_temp)[0][id_date_start:]
        file_name_dataset2_te_temp = os.path.splitext(os.path.expanduser(self.file_path_dataset2).split('/')[-1])[0]
        file_name_dataset2_te_first = os.path.splitext(file_name_dataset2_te_temp)[0][:id_date_start + 8]
        file_name_dataset2_te_second = os.path.splitext(file_name_dataset2_te_temp)[0][id_date_start:]
        file_name_dataset1_te = self.file_out_root + file_name_dataset1_te_first + '_clipped_to_' + file_name_dataset2_te_second + '.tif'
        file_name_dataset2_te = self.file_out_root + file_name_dataset2_te_first + '_clipped_to_' + file_name_dataset1_te_second + '.tif'

        # file_name_dataset2_te = self.file_out_root + os.path.splitext(file_name_dataset1_te_temp)[0][:id_date_start + 8] + '_clipped_to_' +  \
        #         file_name_dataset2_te_temp + '_common_extent.tif'
        file_base_topo_te = os.path.splitext(file_name_dataset1_te_temp)[0][:id_date_start + 8] + '_to_' +  \
                            os.path.splitext(file_name_dataset2_te_temp)[0][id_date_start: id_date_start + 8]
        # file_base_topo_te = self.file_path_dataset1.split('/')[-3] + '_' + self.file_path_dataset1.split('/')[-2]
        file_base_topo_te = self.file_out_root + file_base_topo_te
                #Check if file already exists
        if not (os.path.exists(file_name_dataset1_te)) & (os.path.exists(file_name_dataset2_te) & (os.path.exists(file_base_topo_te + '_dem_common_extent.tif'))):
            # fun gdal command through subprocesses.run to clip and align to commom extent
            print('This will overwrite the "_common_extent.tif" versions of both input files if they are already in exisitence (through this program)' +
                    ' Ensure that file paths "<file_path_dataset1>_common_extent.tif" and "<file_path_dataset2>_common_extent.tif" do not exist or continue to replace them' +
                    ' to proceed type "yes". to exit type "no"')
            while True:
                response = input()
                if response.lower() == 'yes':
                    run_arg1 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format(left_max_bound, bottom_max_bound, right_min_bound, top_min_bound,
                                                                            self.file_path_dataset1, file_name_dataset1_te) + ' -overwrite'
                    # run_arg1 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format(left_max_bound, bottom_max_bound, right_min_bound, top_min_bound,
                    #                                                         self.file_path_dataset1, file_name_dataset1_te) + ' -overwrite'
                    run_arg2 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format(left_max_bound, bottom_max_bound, right_min_bound, top_min_bound,
                                                                            self.file_path_dataset2, file_name_dataset2_te) + ' -overwrite'
                    if topo_rez_same:
                        run_arg3 = 'gdal_translate -of GTiff NETCDF:"{0}":dem {1}'.format(self.file_path_topo, file_base_topo_te + '_dem.tif')
                        run(run_arg3, shell=True)
                    else:
                        run_arg3 = 'gdal_translate -of GTiff -tr {0} {0} NETCDF:"{1}":dem {2}'.format(round(rez), self.file_path_topo, file_base_topo_te + '_dem.tif')
                        run(run_arg3, shell=True)

                    run_arg4 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format(left_max_bound, bottom_max_bound, right_min_bound, top_min_bound,
                                                                            file_base_topo_te + '_dem.tif', file_base_topo_te + '_dem_common_extent.tif -overwrite')

                    run(run_arg1, shell = True)
                    run(run_arg2, shell = True)
                    run(run_arg4, shell = True)
                    run_arg5 = 'rm ' + file_base_topo_te + '_dem.tif'
                    run(run_arg5, shell = True)
                    break
                elif response.lower() == 'no':
                    sys.exit("exiting program")
                    break
                else:
                    print('please answer "yes" or "no"')
        else:
            pass

        #open newly created common extent tifs for use in further analyses
        with rio.open(file_name_dataset1_te) as src:
            d1_te = src
            self.meta = d1_te.profile  # just need one meta data file because it's the same for both
            mat_clip1 = d1_te.read()  #matrix
            mat_clip1 = mat_clip1[0]
        with rio.open(file_name_dataset2_te) as src:
            d2_te = src
            mat_clip2 = d2_te.read()  #matrix
            mat_clip2 = mat_clip2[0]
        with rio.open(file_base_topo_te + '_dem_common_extent.tif') as src:
            topo_te = src
            topo_clip = topo_te.read()
            topo_clip = topo_clip[0]
        # change nans (if present) to -9999
        mat_clip1[np.isnan(mat_clip1)] = -9999
        mat_clip2[np.isnan(mat_clip2)] = -9999
        topo_clip[np.isnan(topo_clip)] = -9999
        self.mat_clip1, self.mat_clip2, self.topo_clip = mat_clip1, mat_clip2, topo_clip

    def mask_advanced(self, name, action, operation, val):
        """
        Adds attributes indicating where no nans present in any input matrices
        and where no nans AND all comparison conditions for each matrice (action, operation, val) are met.

        Arguments
        name: list of strings (1xN), matrix names to base map off of.
        action: Compare or not.  Only two options currently.  Can be expanded in future.
        operation:  (1x2N)list of strings, operation codes to be performed on each matrix
        val: list of floats (1x2N), matching value pairs to operation comparison operators.
        """

        print('Entered mask_advanced')
        keys = {'lt':'<', 'gt':'>'}
        shp = getattr(self, name[0]).shape
        overlap_nan = np.ones(shp, dtype = bool)
        overlap_conditional = np.zeros(shp, dtype = bool)
        extreme_outliers = np.zeros(shp, dtype = bool)
        mat_ct = 0  # initialize count of mats for conditional masking
        for i in range(len(name)):
            mat = getattr(self, name[i])
            # replace nan with -9999 and identify location for output
            if action[i] == 'na':
                action_temp = False
            else:
                action_temp = True
            if np.isnan(mat).any():  # if nans are present and represented by np.nan.
                mat_mask = mat.copy()
                temp_nan = ~np.isnan(mat_mask)
                mat_mask[np.isnan(mat_mask)] = -9999  # set nans to -9999
            elif (mat == -9999).any():  # if nans are present and represented by -9999
                mat_mask = mat.copy()
                temp_nan = mat_mask != -9999  # set nans to -9999
            else:   # no nans present
                mat_mask = mat.copy()
                temp_nan = np.ones(shp, dtype = bool)
            if action_temp == True:
                for j in range(2):
                    id = mat_ct * 2 + j
                    op_str = keys[operation[id]]
                    cmd = 'mat_mask' + op_str + str(val[id])
                    temp = eval(cmd)
                    overlap_conditional = overlap_conditional | temp
                    extreme_outliers = extreme_outliers | (~temp)
                mat_ct += 1
            overlap_nan = overlap_nan & temp_nan  # where conditions of comparison are met and no nans present
            if i == 1:
                nan_to_zero = ~temp_nan_prev & (np.absolute(mat).round(2)==0.0)
                zero_to_nan = zero_prev & ~temp_nan
                zero_and_nan = nan_to_zero | zero_to_nan & ~(~temp_nan_prev & ~temp_nan)
            temp_nan_prev = temp_nan.copy()
            zero_prev = (np.absolute(mat).round(2)==0.0)
        self.overlap_nan = overlap_nan  # where overlap and no nans
        self.flag_zero_and_nan = zero_and_nan
        self.overlap_conditional = overlap_conditional & overlap_nan
        self.flag_extreme_outliers = extreme_outliers & overlap_nan

        #Now that we have mask, make and save a couple arrays
        mat_diff_norm_nans = self.mat_diff_norm.copy()
        mat_diff_norm_nans[~self.overlap_nan] = np.nan
        self.mat_diff_norm_nans = mat_diff_norm_nans

        # find upper bounds and lower bounds of normalized snow depth change for entire basin.  Note: Hard-coded to 50th percentile by trial and error
        temp = self.mat_diff_norm[self.overlap_nan]
        self.lower_bound = round(np.nanpercentile(temp[temp < 0], 50), 3)
        self.upper_bound = round(np.nanpercentile(temp[temp > 0], 50), 3)
        print('Exited mask_advanced')

    def make_diff_mat(self):
        """
        Saves as attribute a normalized difference matrix of the two input tiffs
        and one with nans (mat_diff_norm, mat_diff_norm_nans). difference between \
        two dates is divided by raw depth on date 1 to normalize
        """
        mat_clip1 = self.mat_clip1.copy()
        mat_clip2 = self.mat_clip2.copy()

        self.mat_diff = mat_clip2 - mat_clip1  # raw difference
        self.all_loss = (np.absolute(self.mat_clip1).round(2) > 0.0) & (np.absolute(self.mat_clip2).round(2)==0.0)  #all loss minimal
        mat_clip1[~self.all_loss & (mat_clip1 < 0.25)] = 0.25  # Set snow depths below 0.25m to 0.25m to avoid dividing by zero
        mat_diff_norm = np.round((self.mat_diff / mat_clip1), 2)  #
        self.mat_diff_norm = mat_diff_norm

    def trim_extent_nan(self, name):
        """Used to trim path and rows from array edges with na values.  Returns slimmed down matrix for \
        display purposes and creates attribute of trimmed overlap_nan attribute.

        Args:
            name:    matrix name (string) to access matrix attribute
        Returns:
            np.array:
            **mat_trimmed_nan**: matrix specified by name trimmed to nan extents on all four edges.
        """
        mat_trimmed_nan = getattr(self, name)
        mat = self.overlap_nan
        nrows, ncols = self.overlap_nan.shape[0], self.overlap_nan.shape[1]
        #Now get indices to clip excess NAs
        tmp = []
        for i in range(nrows):
            if any(mat[i,:] == True):
                id = i
                break
        tmp.append(id)
        for i in range(nrows-1, 0, -1):  #-1 because of indexing...
            if any(mat[i,:] == True):
                id = i
                break
        tmp.append(id)
        for i in range(ncols):
            if any(mat[:,i] == True):
                id = i
                break
        tmp.append(id)
        for i in range(ncols-1, 0, -1):  #-1 because of indexing...
            if any(mat[:,i] == True):
                id = i
                break
        tmp.append(id)
        idc = tmp
        mat_trimmed_nan = mat_trimmed_nan[idc[0]:idc[1],idc[2]:idc[3]]
        if ~hasattr(self, 'overlap_nan_trim'):
            self.overlap_nan_trim = self.overlap_nan[idc[0]:idc[1],idc[2]:idc[3]]  # overlap boolean trimmed to nan_extent
        return mat_trimmed_nan

    def save_tiff(self, fname):
        """
        saves matix to geotiff using RasterIO basically. Specify one or more matrices in list of strings
        with attribute names, or leave argv blank to get mat_diff_norm_nans along with all outlier types
        as individual bands in multiband tiff.

        Args:
            fname: filename including path of where to save
            argv:  list of strings of attribute names if don't want default

        """
        tick = time.clock()
        flag_names = self.flag_names.copy()
        flag_names.append('mat_diff_norm_nans')  # 1)Change here and @2 if desire to save single band
        self.meta.update({
            'count': len(flag_names)})

        with rio.open(self.file_path_out, 'w', **self.meta) as dst:
            for id, band in enumerate(flag_names, start = 1):
                try:
                    dst.write_band(id, getattr(self, flag_names[id - 1]))
                except ValueError:
                    mat_temp = getattr(self, flag_names[id - 1])
                    dst.write_band(id, mat_temp.astype('float32'))
        tock = time.clock()
        print('save tiff = ', tock - tick, 'seconds')

    def plot_this(self):
        plot_basic(self)

class PatternFilters():
    def init(self):
        pass
    def mov_wind(self, name, size):
        """
         Very computationally slow moving window base function which adjusts window sizes to fit along
         matrix edges.  Beta version of function. Can add other filter/kernels to moving window calculation
         as needed.  Saves as attribute, pct, which has the proportion of cells/pixels with values > 0 present in each
         moving window centered at target pixel.  pct is the same size as matrix accessed by name

         Args:
            name:  matrix name (string) to access matrix attribute. Matrix must be a boolean.
            size:  moving window size - i.e. size x size.  For example, if size = 5, moving window is a
                    5x5 cell square
        Returns:
            pct:    Proporttion (mislabeled as percentage and abbreviated to pct for easier recognition and code readability) \
                    This is the proportion of cells within moving window that have neighboring cells with values.
                    For example a 5 x 5 window with 5 of the 25 cells having a value, including the target cell, would have \
                    a pct = 0.20.  The target cell would be the center cell at row 3 col 3.
         """

        print('entered histogram moving window')
        tick = time.clock()
        if isinstance(name, str):
            mat = getattr(self, name)
        else:
            mat = name
        nrow, ncol = mat.shape
        base_offset = math.ceil(size/2)
        pct = np.zeros(mat.shape, dtype = float)  # proportion of neighboring pixels in mov_wind
        ct = 0
        for i in range(nrow):
            if i >= base_offset - 1:
                prox_row_edge = nrow - i - 1
                if prox_row_edge >= base_offset - 1:
                    row_idx = np.arange(base_offset * (-1) + 1, base_offset)
                elif prox_row_edge < base_offset - 1:
                    prox_row_edge = nrow - i - 1
                    row_idx = np.arange(prox_row_edge * (-1), base_offset) * (-1)
            elif i < base_offset - 1:
                prox_row_edge = i
                row_idx = np.arange(prox_row_edge * (-1), base_offset)
            for j in range(ncol):
                if j >= base_offset - 1:
                    prox_col_edge = ncol - j - 1
                    if prox_col_edge >= base_offset - 1:  #no window size adjustmentnp.ones(id_dem_bins.shape)
                        col_idx = np.arange(base_offset * (-1) + 1, base_offset)
                    elif prox_col_edge < base_offset - 1:  #at far column edge. adjust window size
                        prox_col_edge = ncol - j - 1
                        col_idx = np.arange(prox_col_edge * (-1), base_offset) * (-1)
                if j < base_offset - 1:
                    prox_col_edge = j
                    col_idx = np.arange(prox_col_edge * (-1), base_offset)

                # Begin the real stuff
                base_row = np.ravel(np.tile(row_idx, (len(col_idx),1)), order = 'F') + i
                base_col = np.ravel(np.tile(col_idx, (len(row_idx),1))) + j
                sub = mat[base_row, base_col]
                pct[i,j] = (np.sum(sub > 0)) / sub.shape[0]
        tock = time.clock()
        print('mov_wind zach version = ', tock - tick, 'seconds')
        return(pct)

    def mov_wind2(self, name, size):
        """
        Orders and orders of magnitude faster moving window function than mov_wind().  Uses numpy bitwise operator
        wrapped in sklearn package to 'stride' across matrix cell by cell.

        Args:
            name:  matrix name (string) to access matrix attribute. Matrix must be a boolean
            size:  moving window size - i.e. size x size.  For example, if size = 5, moving window is a
                       5x5 cell square
        Returns:
            np.array:
                **pct_temp**: Proportion of cells/pixels with values > 0 present in each
                moving window centered at target pixel. Array size same as input array accessed by name
        """

        print('entering map space moving window')
        if isinstance(name, str):
            mat = getattr(self, name)
        else:
            mat = name.copy()
        mat = mat > 0
        base_offset = math.floor(size/2)
        patches = image.extract_patches_2d(mat, (size,size))
        pct_temp = patches.sum(axis = (-1, -2))/(size**2)
        pct_temp = np.reshape(pct_temp, (mat.shape[0] - 2 * base_offset, mat.shape[1] - 2 * base_offset))
        pct = np.zeros(mat.shape)
        pct[base_offset: -base_offset, base_offset : -base_offset] = pct_temp
        return(pct)

class Flags(MultiArrayOverlap, PatternFilters):
    def init(self, file_path_dataset1, file_path_dataset2, file_path_topo, file_out_root, file_name_modifier):
        MultiArrayOverlap.init(self, file_path_dataset1, file_path_dataset2, file_path_topo, file_out_root, file_name_modifier)



    def make_hist(self, name, nbins, thresh, moving_window_size):
        # I don't think an array with nested tuples is computationally efficient.  Find better data structure for the tuple_array
        print('entering make_hist')
        m1, m2 = getattr(self, name[0]), getattr(self, name[1])
        m1_nan, m2_nan = m1[self.overlap_conditional], m2[self.overlap_conditional]
        bins, xedges, yedges = np.histogram2d(np.ravel(m1_nan), np.ravel(m2_nan), nbins)
        self.xedges, self.yedges= xedges, yedges

        # Now find bin edges of overlapping snow depth locations from both dates, and save to self.bin_loc as array of tuples
        xedges = np.delete(xedges, -1)   # remove the last edge
        yedges = np.delete(yedges, -1)
        bins = np.flip(np.rot90(bins,1), 0)  # WTF np.histogram2d.  hack to fix bin mat orientation
        self.bins = bins

        pct = self.mov_wind(bins, moving_window_size)
        flag_spatial_outlier = (pct < thresh[0]) & (bins > 0)
        # flag_bin_ct = (bins < thresh[1]) & (bins > 0)  # ability to filter out bin counts lower than thresh but above zero
        outliers_hist_space = flag_spatial_outlier
        tick = time.clock()
        id = np.where(outliers_hist_space)
        # idx_hist, idy_hist = id[1], id[0]  # column and row id where outliers were found
        idx_bins = np.digitize(m1_nan, xedges) -1  # id of x bin edges.  dims = (N,) array
        idy_bins = np.digitize(m2_nan, yedges) -1  # id of y bin edges
        hist_outliers_temp = np.zeros(idx_bins.shape, dtype = bool)
        for x, y in zip(id[1], id[0]):
                temp = (idx_bins == x) & (idy_bins == y)
                hist_outliers_temp = (idx_bins == x) & (idy_bins == y) | hist_outliers_temp
        hist_outliers = np.zeros(m1.shape, dtype = bool)
        hist_outliers[self.overlap_conditional] = hist_outliers_temp
        self.flag_hist = hist_outliers

        self.outliers_hist_space = outliers_hist_space

        tock = time.clock()
        print('hist2D_with_bins_mapped = ', tock - tick, 'seconds')

    def flag_blocks(self, moving_window_size, neighbor_threshold, snowline_thresh, elevation_band_resolution):
        """
        Finds cells of complete melt, or snow where none existed prior.
        Used to diagnose extreme, unrealistic change that were potentially processed incorrectly

        Args:
            moving_window_size:   size of moving window used to define blocks
            neighbor_threshold:   proportion of neighbors within moving window (including target cell) that have


        """
        # Note below ensures -0.0 and 0 and 0.0 are all discovered and flagged as zeros.
        # Checked variations of '==0.0' and found that could have more decimals or be ==0 with same resultant boolean
        # all_loss = (np.absolute(self.mat_clip1).round(2)!=0.0) & (np.absolute(self.mat_clip2).round(2)==0.0)
        all_gain = (np.absolute(self.mat_clip1).round(2)==0.0) & (np.absolute(self.mat_clip2).round(2) > 0.0)
        all_loss = self.all_loss.copy()
        self.snowline(snowline_thresh, elevation_band_resolution)
        if hasattr(self, 'snowline_elev'):
            pass
        else:
            self.snowline(snowline_thresh, elevation_band_resolution)
        basin_loss = self.overlap_nan & all_loss & (self.topo_clip > self.snowline_elev) #ensures neighbor threshold and overlap, plus from an all_loss cell
        pct = self.mov_wind2(basin_loss, 5)
        self.flag_basin_loss = (pct > 0.39) & all_loss
        basin_gain = self.overlap_nan & all_gain  #ensures neighbor threshold and overlap, plus from an all_gain cell
        pct = self.mov_wind2(basin_gain, 5)
        self.flag_basin_gain = (pct > 0.39) & all_gain

        # add this to basin_gain and basin_loss to fix insufficient histogram flag finder
        if hasattr(self, 'self.bin_loc'):
            self.flag_hist = self.flag_hist & (all_loss | all_gain)
            print('THIS SHOULDNT WORK!')

        self.all_gain = all_gain

    def hypsometry(self, moving_window_size, neighbor_threshold, snowline_thresh, elevation_band_resolution, outlier_percentiles):
        print('entering hypsometry')
        # where bare ground is NOT occuring on both dates - i.e. snow is present in at least one date
        snow_present_mask = ~((np.absolute(self.mat_clip1).round(2)==0.0) & (np.absolute(self.mat_clip2).round(2)==0.0))
        nan_and_snow_present_mask = snow_present_mask & self.overlap_nan  # combine conditional map with snow present
        # Masking bare ground areas because zero change in snow depth will skew distribution from which thresholds are based
        topo_clip_masked = self.topo_clip[nan_and_snow_present_mask]
        mat_diff_norm_masked = self.mat_diff_norm[nan_and_snow_present_mask]
        mat_diff_masked = self.mat_diff[nan_and_snow_present_mask]
        if hasattr(self, 'snowline_elev'):
            pass
        else:
            self.snowline(snowline_thresh, elevation_band_resolution)
        id_dem = np.digitize(topo_clip_masked, self.elevation_edges) -1  #creates bin edge ids.  the '-1' converts to index starting at zero
        id_dem[id_dem == self.elevation_edges.shape[0]]  = self.elevation_edges.shape[0] - 1 #for some there are as many bins as edges.  this smooshes last bin(the max) into second to last bin edge
        id_dem_unique = np.unique(id_dem)  #unique ids
        map_id_dem = np.full(nan_and_snow_present_mask.shape, id_dem_unique[-1] + 1, dtype=int)  # makes nans max(id_dem) + 1
        map_id_dem[nan_and_snow_present_mask] = id_dem
        map_id_dem_overlap = map_id_dem[nan_and_snow_present_mask]

        # initiate arrays used to identify pixels in basin which are outside of specified upper and lower thresholds
        thresh_norm_upper = np.full(id_dem_unique.shape, -9999, dtype = float)
        thresh_norm_lower = np.full(id_dem_unique.shape, -9999, dtype = float)
        thresh_raw_upper = np.full(id_dem_unique.shape, -9999, dtype = float)
        thresh_raw_lower = np.full(id_dem_unique.shape, -9999, dtype = float)
        elevation_count = np.zeros(id_dem_unique.shape, dtype = int)
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            temp = mat_diff_norm_masked
            temp2 = mat_diff_masked
            temp3 = np.ones(temp2.shape, dtype = bool)
            thresh_raw_upper[id] = np.percentile(temp2[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[0])
            thresh_norm_upper[id] = np.percentile(temp[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[1])
            thresh_raw_lower[id] = np.percentile(temp2[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[2])
            thresh_norm_lower[id] = np.percentile(temp[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[3])
            elevation_count[id] = getattr(temp3[map_id_dem_overlap == id_dem_unique2], 'sum')()
            # elevation_std[id] = getattr(mat_diff_norm_masked[map_id_dem_overlap == id_dem_unique2], 'std')()

        # Create dataframe of elevation band satistics on thresholds
        elev_stack = self.elevation_edges[id_dem_unique -1].ravel()
        # Simply preparing the column names in a syntactically shittilly readable format:
        column_names = ['elevation', '{}% depth change (m)', '{}% depth change (normalize)', '{}% depth change (m)', '{}% depth change (normalized)', 'elevation_count']
        column_names_temp = []
        ct = 0
        for id, names in enumerate(column_names):
            if '{}' in names:
                names = names.format(str(outlier_percentiles[ct]))
                ct += 1
            column_names_temp.append(names)
        temp = np.stack((elev_stack, np.around(thresh_raw_upper.ravel(), 3), np.around(thresh_norm_upper.ravel(), 3),
                np.around(thresh_raw_lower.ravel(), 3), np.around(thresh_norm_lower.ravel(), 3), elevation_count.ravel()), axis = -1)
        df = pd.DataFrame(temp, columns = column_names_temp)

        upper_compare_array_norm = np.zeros(nan_and_snow_present_mask.shape)   #this will be a map populated with change threshold at each map location based on elevation thresh
        lower_compare_array_norm = np.zeros(nan_and_snow_present_mask.shape)   #this will be a map populated with change threshold at each map location based on elevation thresh
        upper_compare_array_raw = upper_compare_array_norm.copy()
        lower_compare_array_raw = lower_compare_array_norm.copy()
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            try:
                upper_compare_array_norm[map_id_dem == id_dem_unique2] = thresh_norm_upper[id]
                upper_compare_array_raw[map_id_dem == id_dem_unique2] = thresh_raw_upper[id]
            except IndexError:
                pass
            try:
                lower_compare_array_norm[map_id_dem == id_dem_unique2] = thresh_norm_lower[id]
                lower_compare_array_raw[map_id_dem == id_dem_unique2] = thresh_raw_lower[id]
            except IndexError:
                pass

        elevation_gain = (self.mat_diff_norm > upper_compare_array_norm) & (self.mat_diff > upper_compare_array_raw)
        # elevation_gain[(~nan_and_snow_present_mask) | (self.topo_clip <= self.snowline_elev)] = False
        elevation_gain[~nan_and_snow_present_mask] = False
        pct = self.mov_wind2(elevation_gain, moving_window_size)
        self.flag_elevation_gain = (pct > neighbor_threshold) & elevation_gain

        elevation_loss = self.mat_diff < lower_compare_array_raw
        elevation_loss[~nan_and_snow_present_mask] = False
        pct = self.mov_wind2(elevation_loss, moving_window_size)
        self.flag_elevation_loss = (pct > neighbor_threshold) & elevation_loss

    def snowline(self, snowline_thresh, elevation_band_resolution):
        # use overlap_nan mask for snowline because we want to get average snow per elevation band INCLUDING zero snow depth
        topo_clip_conditional = self.topo_clip[self.overlap_nan]
        min_elev, max_elev = np.min(topo_clip_conditional), np. max(topo_clip_conditional)
        edge_min = min_elev % elevation_band_resolution
        edge_min = min_elev - edge_min
        edge_max = max_elev % elevation_band_resolution
        edge_max = max_elev + (elevation_band_resolution - edge_max)
        self.elevation_edges = np.arange(edge_min, edge_max + 0.1, elevation_band_resolution)
        id_dem = np.digitize(topo_clip_conditional, self.elevation_edges) -1
        id_dem_unique = np.unique(id_dem)
        map_id_dem = np.full(self.overlap_nan.shape, id_dem_unique[-1] + 1, dtype=int)  # makes nans max(id_dem) + 1
        map_id_dem[self.overlap_nan] = id_dem
        snowline_mean = np.full(id_dem_unique.shape, -9999, dtype = 'float')
        snowline_std = np.full(id_dem_unique.shape, -9999, dtype = 'float')
        # use the matrix with the deepest mean basin snow depth to base snowline thresh off of.  Assuming deeper avg basin snow = lower snowline
        if np.mean(self.mat_clip1[self.overlap_nan]) > np.mean(self.mat_clip1[self.overlap_nan]):
            mat_temp = self.mat_clip1
            mat = mat_temp[self.overlap_nan]
            mat_temp[self.overlap_nan & mat_temp ]
        else:
            mat_temp = self.mat_clip2
            mat = mat_temp[self.overlap_nan]
        map_id_dem2_masked = map_id_dem[self.overlap_nan]
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            snowline_mean[id] = getattr(mat[map_id_dem2_masked == id_dem_unique2], 'mean')()
            snowline_std[id] = getattr(mat[map_id_dem2_masked == id_dem_unique2], 'std')()
        id_min = np.min(np.where(snowline_mean > snowline_thresh))
        self.snowline_elev = self.elevation_edges[id_min]  #elevation of estimated snowline

        print(('The snowline was determined to be at {0}m. It was defined as the first elevation band in the basin'
                'with a mean snow depth >= {1}. Elevation bands were in {2}m increments ').
                format(self.snowline_elev, snowline_thresh, elevation_band_resolution))

    def combine_flags(self, names):
        """
        adds up all flag map attributes specified in names (list of strings).  Yields on boolean
        matrix attribute (self.flags_combined) which maps all locations of outliers

        Args:
            names:      list of strings.  names of flags saved as boolean matrice attributes
        """
        flag_names = []
        for name in names:
            if name not in ['basin_block', 'elevation_block']:
                flag_names.append('flag_' + name)
                # flagged = getattr(self, flag_names[i])
            else:
                if name in ['basin_block']:
                    flag_names.append('flag_basin_loss')
                    flag_names.append('flag_basin_gain')
                elif name in ['elevation_block']:
                    flag_names.append('flag_elevation_loss')
                    flag_names.append('flag_elevation_gain')
            self.flag_names = flag_names  #just accumulates all names into a list


    def __repr__(self):
            return ("Main items of use are matrices clipped to each other's extent and maps of outlier flags \
                    Also capable of saving geotiffs and figures")
