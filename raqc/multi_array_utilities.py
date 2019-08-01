import rasterio as rio
import numpy as np
import copy
import math
from sklearn.feature_extraction import image

class MultiArrayOverlap(object):
    def __init__(self, fp_d1, fp_d2, fp_out):
        with rio.open(fp_d1) as src:
            self.d1 = src
            self.meta1 = self.d1.profile
            mat1 = self.d1.read()  #matrix
            self.mat1 = mat1[0]
        with rio.open(fp_d2) as src:
            self.d2 = src
            self.meta2 = self.d2.profile
            mat2 = self.d2.read()  #matrix
            self.mat2 = mat2[0]
        self.fp_out = fp_out

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
        bounds = [d1.bounds.left, d1.bounds.bottom, d1.bounds.right, d1.bounds.top, \
                d2.bounds.left, d2.bounds.bottom, d2.bounds.right, d2.bounds.top]
        rez = meta['transform'][0]  # spatial resolution
        # minimum overlapping extent
        left_max_bound = max(bounds[0], bounds[4])
        bottom_max_bound = max(bounds[1], bounds[5])
        right_min_bound =  min(bounds[2], bounds[6])
        top_min_bound = min(bounds[3], bounds[7])
        self.top_max_bound = top_min_bound
        self.left_max_bound = left_max_bound
        for d in range(2):  # datasets
            # below loop makes list of line and sample georaphic coordinates (column and row)
            for r in range(2):  #row or column (i.e. samples or lines)
                num_lines_or_samps = math.ceil((bounds[3 + d * 4 - r] - bounds[1 + d * 4 - r])/rez)
                coords = []
                for c in range(0, num_lines_or_samps):
                    coords.append(bounds[1 + d * 4 - r] + rez * c)
                if r == 0:  # row and data 1
                    del coords[0]
                    coords.append(bounds[1 + d * 4 - r] + rez * (c + 1))
                    row = coords.copy()
                    num_samps = num_lines_or_samps
                elif r == 1:  # col and data 1:
                    col = coords.copy()
            # get indices of minimum overlapping extent
            ids = [None] * 4
            ids[0:4] = [num_samps - row.index(top_min_bound) - 1, num_samps - row.index(bottom_max_bound + rez) -1, \
                    col.index(left_max_bound), col.index(right_min_bound - rez)]
            #subset both datasets --> mat_clip[num]
            if d == 0:
                mat = self.mat1
                mat_clip1 = mat[ids[0]:ids[1], ids[2]:ids[3]]
            elif d == 1:
                mat = self.mat2
                mat_clip2 = mat[ids[0]:ids[1], ids[2]:ids[3]]
        # calculations on np arrays fail with np.nan values --> save one of each mat with np.nans and -9999
        self.mat_clip1_nans, self.mat_clip2_nans = mat_clip1.copy(), mat_clip2.copy()
        mat_clip1, mat_clip2 = mat_clip1.copy(), mat_clip2.copy()
        mat_clip1[np.isnan(mat_clip1)] = -9999
        mat_clip2[np.isnan(mat_clip2)] = -9999
        self.mat_clip1, self.mat_clip2 = mat_clip1, mat_clip2

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

    def save_tiff(self, name, fname, *argv):
        """
        saves matix to geotiff using RasterIO basically. Specify one or more matrices in list of strings
        with attribute names, or leave argv blank to get mat_diff_norm_nans along with all outlier types
        as individual bands in multiband tiff.

        Args:
            fname: filename including path of where to save
            argv:  list of strings of attribute names if don't want default

        """

        fn_out = self.fp_out + fname + '.tif'
        if len(argv) > 0:  # if a specific band is specified
            name = argv[0]
            dims = getattr(self, name).shape
            count = 1
        else:
            name = ['flag_loss_block', 'flag_gain_block', 'flag_hist', 'mat_diff_norm_nans']
            dims = getattr(self, name[0]).shape
            count = len(name)

        meta_new = copy.deepcopy(self.meta1)
        aff = copy.deepcopy(self.d1.transform)
        new_aff = rio.Affine(aff.a, aff.b, self.left_max_bound, aff.d, aff.e, self.top_max_bound)
        meta_new.update({
            'height':dims[0],
            'width': dims[1],
            'transform': new_aff,
            'count': count})

        if count == 1:
            with rio.open(fn_out, 'w', **meta_new) as dst:
                mat_temp = getattr(self,name)
                try:
                    dst.write(mat_temp,1)# print(mask.shape)
                except ValueError:
                    dst.write(mat_temp.astype('float32'),1)
        if count > 1:
            with rio.open(fn_out, 'w', **meta_new) as dst:
                for id, band in enumerate(name, start = 1):
                    try:
                        dst.write_band(id, getattr(self, name[id - 1]))# print(mask.shape)
                    except ValueError:
                        mat_temp = getattr(self, name[id - 1])
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
                    if prox_col_edge >= base_offset - 1:  #no window size adjustment
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
    def init(self, fp_d1, fp_d2, fp_out):
        MultiArrayOverlap.init(self, fp_d1, fp_d2, fp_out)

    def hist2d_with_bins_mapped(self, name, nbins):
        """
        basically creates all components necessary to create historgram using np.histogram2d, and saves map locations
        in x and y list at each histogram space cell.  These are unpacked in hist_to_map_space() method later.  Useful
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

    def combine_flags(self, names):
        """
        adds up all flag map attributes specified in names (list of strings).  Yields on boolean
        matrix attribute (self.flags_combined) which maps all locations of outliers

        Args:
            names:      list of strings.  names of flags saved as boolean matrice attributes
        """
        for i in range(len(names)):
            flag = 'flag_' + names[i]
            flagged = getattr(self, flag)
            if i == 0:
                flag_combined = flagged
            else:
                flag_combined = flag_combined | flagged
        self.flag_combined = flag_combined


    def __repr__(self):
            return ("Main items of use are matrices clipped to each other's extent and maps of outlier flags \
                    Also capable of saving geotiffs and figures")
