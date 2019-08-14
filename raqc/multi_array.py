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

class MultiArrayOverlap(object):
    def __init__(self, file_path_dataset1, file_path_dataset2, file_path_topo, file_out_root, file_name_modifier):
        with rio.open(file_path_dataset1) as src:
            self.d1 = src
            self.meta1 = self.d1.profile
        with rio.open(file_path_dataset2) as src:
            self.d2 = src
            self.meta2 = self.d2.profile
        self.file_path_dataset1 = file_path_dataset1
        self.file_path_dataset2 = file_path_dataset2
        year1 = file_path_dataset1.split('/')[-1].split('_')[0]
        year2 = file_path_dataset2.split('/')[-1].split('_')[0][-8:]
        self.file_path_out = '{0}{1}_to_{2}_{3}.tif'.format(file_out_root, year1, year2, file_name_modifier)
        # topo = h5py.File(file_path_topo, 'r')
        # self.dem = topo['dem']
        self.file_path_topo = file_path_topo
        self.file_out_root = file_out_root

    def clip_extent_overlap(self):
        """
        finds overlapping extent of two geotiffs. Saves as attributes clipped versions
        of both matrices extracted from geotiffs, and clipped matrices with -9999 replaced
        with nans (mat_clip1, mat_clip2 and mat_clip_nans of each).
        """
        meta = self.meta1  #metadata
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
        if (round(rez) == round(rez2)) & (round(rez) == round(rez3)):  # janky way to check that all three rez are the same
            pass
        else:
            sys.exit("check that spatial resolution of your two files are the same")

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

        file_name_dataset1_te = os.path.splitext(self.file_path_dataset1.split('/')[-1])[0] + '_common_extent.tif'
        file_name_dataset1_te = self.file_out_root + file_name_dataset1_te
        file_name_dataset2_te = os.path.splitext(self.file_path_dataset2.split('/')[-1])[0] + '_common_extent.tif'
        file_name_dataset2_te = self.file_out_root + file_name_dataset2_te
        file_base_topo_te = self.file_path_dataset1.split('/')[-3] + '_' + self.file_path_dataset1.split('/')[-2]
        file_base_topo_te = self.file_out_root + file_base_topo_te
        print(file_base_topo_te)
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
                    run_arg2 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format(left_max_bound, bottom_max_bound, right_min_bound, top_min_bound,
                                                                            self.file_path_dataset2, file_name_dataset2_te) + ' -overwrite'
                    run_arg3 = 'gdal_translate -of GTiff NETCDF:"{0}":dem {1}'.format(self.file_path_topo, file_base_topo_te + '_dem.tif')
                    run(run_arg3, shell=True)
                    run_arg4 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format(left_max_bound, bottom_max_bound, right_min_bound, top_min_bound,
                                                                            file_base_topo_te + '_dem.tif', file_base_topo_te + '_dem_common_extent.tif -overwrite')

                    run(run_arg1, shell = True)
                    run(run_arg2, shell = True)
                    run(run_arg4, shell = True)
                    print(run_arg4)
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

        self.mat_clip1_nans, self.mat_clip2_nans, self.topo_clip_nans = mat_clip1.copy(), mat_clip2.copy(), topo_clip.copy()
        mat_clip1, mat_clip2, topo_clip = mat_clip1.copy(), mat_clip2.copy(), topo_clip.copy()
        mat_clip1[np.isnan(mat_clip1)] = -9999
        mat_clip2[np.isnan(mat_clip2)] = -9999
        topo_clip[np.isnan(topo_clip)] = -9999
        self.mat_clip1, self.mat_clip2, self.topo_clip = mat_clip1, mat_clip2, topo_clip

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
        idc = tmp.copy()   #idx to clip [min_row, max_row, min_col, max_col]
        mat_trimmed_nan = mat_trimmed_nan[idc[0]:idc[1],idc[2]:idc[3]]
        if ~hasattr(self, 'overlap_nan_trim'):
            self.overlap_nan_trim = self.overlap_nan[idc[0]:idc[1],idc[2]:idc[3]]  # overlap boolean trimmed to nan_extent
        return mat_trimmed_nan

    def mask_advanced(self, name, action, operation, val):
        """
        Adds attributes indicating where no nans present in any input matrices
        and where no nans AND all comparison conditions for each matrice (action, operation, val) are met.

        Arguments
        name: list of strings (1xN), matrix names to base map off of.
        operation:  (1x2N)list of strings, operation codes to be performed on each matrix
        val: list of floats (1x2N), matching value pairs to operation comparison operators.
        """

        keys = {'lt':'<', 'gt':'>'}
        shp = getattr(self, name[0]).shape
        overlap_nan = np.ones(shp, dtype = bool)
        overlap_conditional = np.ones(shp, dtype = bool)
        mat_ct = 0  # initialize count of mats for conditional masking
        for i in range(len(name)):
            mat = getattr(self, name[i])
            # replace nan with -9999 and identify location for output
            if action[i] == 'na':
                action_temp = False
            else:
                action_temp = True
            if np.isnan(mat).any():  # if nans are present and represented by np.nan
                mat_mask = mat.copy()
                temp_nan = ~np.isnan(mat_mask)
                mat_mask[np.isnan[mat_mask]] = -9999  # set nans to -9999 (if they exist)
            elif (mat == -9999).any():  # if nans are present and represented by -9999
                mat_mask = mat.copy()
                temp_nan = mat_mask != -9999
            else:   # no nans present
                mat_mask = mat.copy()
                temp_nan = np.ones(shp, dtype = bool)
            if action_temp == True:
                for j in range(2):
                    id = mat_ct * 2 + j
                    op_str = keys[operation[id]]
                    cmd = 'mat_mask' + op_str + str(val[id])
                    temp = eval(cmd)
                    overlap_conditional = overlap_conditional & temp
                mat_ct += 1
                overlap_nan = overlap_nan & temp_nan  # where conditions of comparison are met and no nans
        self.overlap_nan = overlap_nan.copy()
        self.overlap_conditional = overlap_conditional & overlap_nan
        temp = self.mat_diff_norm[self.overlap_nan]
        self.lb = round(np.nanpercentile(temp[temp < 0], 50), 3)
        self.ub = round(np.nanpercentile(temp[temp > 0], 50), 3)

    def save_tiff(self, fname):
        """
        saves matix to geotiff using RasterIO basically. Specify one or more matrices in list of strings
        with attribute names, or leave argv blank to get mat_diff_norm_nans along with all outlier types
        as individual bands in multiband tiff.

        Args:
            fname: filename including path of where to save
            argv:  list of strings of attribute names if don't want default

        """
        flag_names = self.flag_names.copy()
        flag_names.append('mat_diff_norm_nans')  # 1)Change here and @2 if desire to save single band
        self.meta.update({
            'count': len(flag_names)})

        print(flag_names)
        print('type ', type(flag_names))
        print(len(flag_names))
        with rio.open(self.file_path_out, 'w', **self.meta) as dst:
            for id, band in enumerate(flag_names, start = 1):
                try:
                    dst.write_band(id, getattr(self, flag_names[id - 1]))# print(mask.shape)
                except ValueError:
                    mat_temp = getattr(self, flag_names[id - 1])
                    dst.write_band(id, mat_temp.astype('float32'))

    def make_diff_mat(self):
        """
        Saves as attribute a normalized difference matrix of the two input tiffs
        and one with nans (mat_diff_norm, mat_diff_norm_nans).
        """
        mat_clip1 = self.mat_clip1.copy()
        mat_clip2 = self.mat_clip2.copy()

        self.mat_diff = mat_clip2 - mat_clip1
        mat_clip1[(mat_clip1 < 0.25) & (mat_clip1 != -9999)] = 0.25  # To avoid dividing by zero
        mat_diff_norm = np.round((self.mat_diff / mat_clip1), 2)
        self.mat_diff_norm = mat_diff_norm.copy()
        mat_diff_norm_nans = mat_diff_norm.copy()
        mat_diff_norm_nans[(mat_clip1 == -9999) | (mat_clip2 == -9999)] = np.nan
        self.mat_diff_norm_nans = mat_diff_norm_nans.copy()

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
            name:  matrix name (string) to access matrix attribute
            size:  moving window size - i.e. size x size.  For example, if size = 5, moving window is a
                    5x5 cell square
         """

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
        return(pct)

    def mov_wind2(self, name, size):
        """
        Orders and orders of magnitude faster moving window function than mov_wind().  Uses numpy bitwise operator
        wrapped in sklearn package to 'stride' across matrix cell by cell.

        Args:
            name:  matrix name (string) to access matrix attribute
            size:  moving window size - i.e. size x size.  For example, if size = 5, moving window is a
                       5x5 cell square
        Returns:
            np.array:
                **pct_temp**: Proportion of cells/pixels with values > 0 present in each
                moving window centered at target pixel. Array size same as input array accessed by name
        """

        if isinstance(name, str):
            mat = getattr(self, name)
        else:
            mat = name.copy()
        mat = mat > 0
        base_offset = math.floor(size/2)
        patches = image.extract_patches_2d(mat, (size,size))
        pct = patches.sum(axis = (-1, -2))/(size**2)
        pct = np.reshape(pct, (mat.shape[0] - 2 * base_offset, mat.shape[1] - 2 * base_offset))
        pct_temp = np.zeros(mat.shape)
        pct_temp[base_offset: -base_offset, base_offset : -base_offset] = pct
        return(pct_temp)

class Flags(MultiArrayOverlap, PatternFilters):
    def init(self, file_path_dataset1, file_path_dataset2, file_path_topo, file_out_root, file_name_modifier):
        MultiArrayOverlap.init(self, file_path_dataset1, file_path_dataset2, file_path_topo, file_out_root, file_name_modifier)

    def hist2d_with_bins_mapped(self, name, nbins):
        """
        basically creates all components necessary to create historgram using np.histogram2d, and saves map locations
        when locations from matrix contributing to pertinent bins on histogram are needed.

        Args:
            name:       names of two matrices (list of strings) used to build 2D histogram, ordered <name x, name y>.
            nbins:      list of number of bins in x and y axis

        """

        # I don't think an array with nested tuples is computationally efficient.  Find better data structure for the tuple_array
        m1, m2 = getattr(self, name[0]), getattr(self, name[1])
        self.mat_shape = m1.shape
        m1_nan, m2_nan = m1[self.overlap_conditional], m2[self.overlap_conditional]
        bins, xedges, yedges = np.histogram2d(np.ravel(m1_nan), np.ravel(m2_nan), nbins)
        # Now find bin edges of overlapping snow depth locations from both dates, and save to self.bin_loc as array of tuples
        xedges = np.delete(xedges, -1)   # remove the last edge
        yedges = np.delete(yedges, -1)
        bins = np.flip(np.rot90(bins,1), 0)  # WTF np.histogram2d.  hack to fix bin mat orientation
        self.bins, self.xedges, self.yedges = bins, xedges, yedges
        # Note: subtract 1 to go from 1 to N to 0 to N - 1 (makes indexing possible below)
        idxb = np.digitize(m1_nan, xedges) -1  # id of x bin edges.  dims = (N,) array
        idyb = np.digitize(m2_nan, yedges) -1  # id of y bin edges
        tuple_array = np.empty((nbins[1], nbins[0]), dtype = object)
        id = np.where(self.overlap_conditional)  # coordinate locations of mat used to generate hist
        idmr, idmc = id[0], id[1]  # idmat row and col
        for i in range(idyb.shape[0]):
                if type(tuple_array[idyb[i], idxb[i]]) != list:  #initiate list if does not exist
                    tuple_array[idyb[i], idxb[i]] = []
                tuple_array[idyb[i], idxb[i]].append([idmr[i], idmc[i]])  #appends map space indices into bin space
        self.bin_loc = tuple_array  #array of tuples containing 0 to N x,y coordinates of overlapping snow map
                                    #locations contributing to 2d histogram bins
    def outliers_hist(self, thresh, moving_window_name, moving_window_size):
        """
        Finds spatial outliers in histogram and bins below a threshold bin count.
        Outputs boolean where these outliers are located in histogram space
        Args:
            thresh:     list of three values - 0. bin count threshold below which bins are flagged.
                        1.  Currently Unused, but areas where complete melt occurs.
                        2. proportion of neighbors, self.pct, which have values.

        """

        pct = self.mov_wind(moving_window_name, moving_window_size)
        flag_spatial_outlier = (pct < thresh[0]) & (self.bins > 0)
        flag_bin_ct = (self.bins < thresh[1]) & (self.bins > 0)  # ability to filter out bin counts lower than thresh but above zero
        flag = (flag_spatial_outlier | flag_bin_ct)
        self.outliers_hist_space = flag
        self.hist_to_map_space()  # unpack hisogram spacconsider adding density option to np.histogram2de outliers to geographic space locations

    def hist_to_map_space(self):
        """
        unpacks histogram bins onto their contributing map locations (self.outliers_hist_space).
        Data type is a list of x,y coordinate pairs
        """
        hist_outliers = np.zeros(self.mat_shape, dtype = int)
        idarr = np.where(self.outliers_hist_space)
        for i in range(len(idarr[0])):  # iterate through all bins with tuples
            loc_tuple = self.bin_loc[idarr[0][i], idarr[1][i]]
            for j in range(len(loc_tuple)):  # iterate through each tuple within each bin
                pair = loc_tuple[j]
                hist_outliers[pair[0], pair[1]] = 1
        self.flag_hist = hist_outliers  # unpacked map-space locations of outliers

    def flag_blocks(self, moving_window_size, neighbor_threshold):
        """
        Finds spatially continuous blocks or areas of complete melt, or snow where none existed prior.
        Used to diagnose tiles of extreme, unrealistic change that were potentially processed incorrectly as tiles

        Args:
            moving_window_size:   size of moving window used to define blocks
            neighbor_threshold:   proportion of neighbors within moving window (including target cell) that have


        """
        all_loss = 1 * (self.mat_clip1 != 0) & (self.mat_clip2 == 0)  #lost everything
        all_gain = 1 * (self.mat_clip1 == 0) & (self.mat_clip2 != 0)  #gained everything
        loss_outliers = self.mat_diff_norm < self.lb
        gain_outliers = self.mat_diff_norm > self.ub
        flag_loss_block = all_loss & loss_outliers
        flag_gain_block = all_gain & gain_outliers
        pct = self.mov_wind2(flag_loss_block, 5)
        flag_loss_block = (pct > neighbor_threshold) & self.overlap_conditional & all_loss
        self.flag_loss_block = flag_loss_block.copy()
        pct = self.mov_wind2(flag_gain_block, 5)
        flag_gain_block = (pct >neighbor_threshold) & self.overlap_conditional & all_gain
        self.flag_gain_block = flag_gain_block.copy()

    def hypsometry(self):
        bare_ground_mask = (np.absolute(self.mat_clip1).round(2)==0.0) & (np.absolute(self.mat_clip2).round(2)==0.0)
        bare_ground_mask = (~bare_ground_mask)
        overlap_bare_ground = bare_ground_mask & self.overlap_conditional
        self.overlap_bare_ground = overlap_bare_ground
        topo_clip_masked = self.topo_clip[overlap_bare_ground]
        mat_diff_norm_masked = self.mat_diff_norm[overlap_bare_ground]

        self.snowline()

        # min_elev, max_elev = np.min(topo_clip_masked), np. max(topo_clip_masked)
        # min_elev_round, max_elev_round = math.ceil(min_elev/100)*100, math.floor(max_elev/100)*100
        # elevation_edges = np.arange(min_elev_round, max_elev_round, 200)
        # elevation_edges[0], elevation_edges[-1] = min_elev, max_elev
        map_id_dem = np.digitize(topo_clip_masked, self.elevation_edges) -1  #creates bin edge ids.  the '-1' converts to index starting at zero
        map_id_dem[map_id_dem == self.elevation_edges.shape[0]]  = self.elevation_edges.shape[0] - 1 #for some there are as many bins as edges.  this smooshes last bin(the max) into second to last bin edge
        id_dem = np.unique(map_id_dem)  #unique ids
        map_id_dem2 = np.full(overlap_bare_ground.shape, id_dem[-1] + 1, dtype=int)  # makes nans max(id_dem) + 1
        map_id_dem2[overlap_bare_ground] = map_id_dem

        map_id_dem2_overlap = map_id_dem2[overlap_bare_ground]
        elevation_std = np.full(id_dem.shape, -9999, dtype = float)
        elevation_mean = np.full(id_dem.shape, -9999, dtype = float)
        for id, id_dem2 in enumerate(id_dem):
            elevation_std[id] = getattr(mat_diff_norm_masked[map_id_dem2_overlap == id_dem2], 'std')()
            elevation_mean[id] = getattr(mat_diff_norm_masked[map_id_dem2_overlap == id_dem2], 'mean')()
        num_std = 2
        thresh_upper = elevation_mean + elevation_std * num_std
        thresh_lower = elevation_mean - elevation_std * num_std
        # thresh_upper = mean + std * num_std
        # thresh_lower = mean - std * num_std
        # print(thresh_upper)
        # print(elevation_edges)
        # print(thresh_upper)
        # print(thresh_lower)

        upper_compare_array = np.zeros(overlap_bare_ground.shape)   #this will be a map populated with change threshold at each map location based on elevation thresh
        lower_compare_array = np.zeros(overlap_bare_ground.shape)   #this will be a map populated with change threshold at each map location based on elevation thresh
        for id, id_dem in enumerate(id_dem):
            try:
                upper_compare_array[map_id_dem2 == id_dem] = thresh_upper[id]
            except IndexError:
                pass
            try:
                lower_compare_array[map_id_dem2 == id_dem] = thresh_lower[id]
            except IndexError:
                pass



        self.flag_elevation_gain = self.mat_diff_norm > upper_compare_array
        self.flag_elevation_gain[(~overlap_bare_ground) | (self.topo_clip <= self.snowline_elev)] = False
        self.flag_elevation_loss = self.mat_diff_norm < lower_compare_array
        self.flag_elevation_loss[(~overlap_bare_ground) | (self.topo_clip <= self.snowline_elev)] = False
    def snowline(self):
        topo_clip_conditional = self.topo_clip[self.overlap_conditional]
        min_elev, max_elev = np.min(topo_clip_conditional), np. max(topo_clip_conditional)
        dem_resolution = 50
        edge_min = min_elev % dem_resolution
        edge_min = min_elev - edge_min
        edge_max = max_elev % dem_resolution
        edge_max = max_elev + (dem_resolution - edge_max)
        self.elevation_edges = np.arange(edge_min, edge_max + 0.1, dem_resolution)
        print('elevation min is: ', min_elev)
        map_id_dem = np.digitize(topo_clip_conditional, self.elevation_edges) -1
        id_dem = np.unique(map_id_dem)
        map_id_dem2 = np.full(self.overlap_conditional.shape, id_dem[-1] + 1, dtype=int)  # makes nans max(id_dem) + 1
        map_id_dem2[self.overlap_conditional] = map_id_dem
        snowline_mean = np.full(id_dem.shape, -9999, dtype = 'float')
        mat_clip2_mask = self.mat_clip2[self.overlap_conditional]
        map_id_dem2_mask = map_id_dem2[self.overlap_conditional]
        for id, id_dem2 in enumerate(id_dem):
            snowline_mean[id] = getattr(mat_clip2_mask[map_id_dem2_mask == id_dem2], 'mean')()
        id_min = np.min(np.where(snowline_mean > 0.40))
        print(snowline_mean)
        print(self.elevation_edges)
        print(self.elevation_edges[id_min])
        self.snowline_elev = self.elevation_edges[id_min]

    def combine_flags(self, names):
        """
        adds up all flag map attributes specified in names (list of strings).  Yields on boolean
        matrix attribute (self.flags_combined) which maps all locations of outliers

        Args:
            names:      list of strings.  names of flags saved as boolean matrice attributes
        """
        flag_names = []
        for i in range(len(names)):
            flag_names.append('flag_' + names[i])
            flagged = getattr(self, flag_names[i])
            if i == 0:
                flag_combined = flagged
            else:
                flag_combined = flag_combined | flagged
        self.flag_combined = flag_combined
        self.flag_names = flag_names  #just accumulates all names into a list


    def __repr__(self):
            return ("Main items of use are matrices clipped to each other's extent and maps of outlier flags \
                    Also capable of saving geotiffs and figures")
