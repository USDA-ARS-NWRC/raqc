from raqc.raqc_plot import plot_basic
import numpy as np
import rasterio as rio
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
from memory_profiler import profile

class MultiArrayOverlap(object):
    def __init__(self, file_path_dataset1, file_path_dataset2, file_path_topo, file_out_root, basin, file_name_modifier, elevation_band_resolution):

        # Ensure that dataset 1 and dataset2 are in chronological order
        self.date1_string = file_path_dataset1.split('/')[-1].split('_')[0][-8:]
        self.date2_string = file_path_dataset2.split('/')[-1].split('_')[0][-8:]
        check_chronology1 = pd.to_datetime(file_path_dataset1.split('/')[-1].split('_')[0][-8:], format = '%Y%m%d')
        check_chronology2 = pd.to_datetime(file_path_dataset2.split('/')[-1].split('_')[0][-8:], format = '%Y%m%d')

        if check_chronology1 < check_chronology2:
            pass
        else:
            sys.exit("Date 1 must occur before Date 2. Please switch Date 1 with Date 2 in UserConfit. Exiting program")

        # Make subdirectory --> file_out_root/basin
        file_out_basin = os.path.join(file_out_root, basin)
        if not os.path.exists(file_out_basin):
            os.makedirs(file_out_basin)

        # Make file paths
        year1 = file_path_dataset1.split('/')[-1].split('_')[0]
        year2 = file_path_dataset2.split('/')[-1].split('_')[0][-8:]
        file_base = '{0}_to_{1}'.format(year1, year2)
        self.file_path_out_base = os.path.join(file_out_basin, file_base)
        self.file_path_out_tif = '{0}_{1}.tif'.format(self.file_path_out_base, file_name_modifier)
        self.file_path_out_backup_config = '{0}_{1}'.format(self.file_path_out_base, 'raqc_backup_config.ini')
        self.file_path_out_csv = '{0}_raqc.csv'.format(self.file_path_out_base)
        path_log_file = '{0}_memory_usage.log'.format(self.file_path_out_base)
        # log_file = open(path_log_file, 'w+')

        # check if user passed clipped files in config file
        string_match = 'clipped_to'
        dem_clipped = self.file_path_out_base + '_dem_common_extent.tif'
        self.already_clipped =  (string_match in file_path_dataset1) & (string_match in file_path_dataset2) & (os.path.isfile(dem_clipped))

        if not self.already_clipped:
            self.file_path_dataset1 = file_path_dataset1
            self.file_path_dataset2 = file_path_dataset2
            self.file_path_topo = file_path_topo
            self.file_out_basin = file_out_basin

        # Grab arrays and spatial metadata
        with rio.open(file_path_dataset1) as src:
            self.d1 = src
            self.meta = self.d1.profile
            if self.already_clipped:
                mat_clip1 = self.d1.read()  #matrix
                mat_clip1 = mat_clip1[0]
                mat_clip1[np.isnan(mat_clip1)] = -9999
                self.mat_clip1 = self.get16bit(mat_clip1)
        with rio.open(file_path_dataset2) as src:
            self.d2 = src
            self.meta2 = self.d2.profile
            if self.already_clipped:
                mat_clip2 = self.d2.read()  #matrix
                mat_clip2 = mat_clip2[0]
                mat_clip2[np.isnan(mat_clip2)] = -9999
                self.mat_clip2 = self.get16bit(mat_clip2)

        if self.already_clipped:
            with rio.open(file_path_topo) as src:
                topo = src
                dem_clip = topo.read()
                dem_clip = dem_clip[0]
                dem_clip[np.isnan(dem_clip)] = -9999
                self.dem_clip = dem_clip
                # ensure user_specified DEM resolution is compatible with uint8 i.e. not too fine
                self.check_DEM_resolution()

        # we need this for later
        self.elevation_band_resolution = elevation_band_resolution

    # @profile
    def clip_extent_overlap(self, remove_clipped_files):
        """
        finds overlapping extent of two geotiffs. Saves as attributes clipped versions
        of both matrices extracted from geotiffs, Also clips and outputs DEM and Vegetation
        bands from topo.nc file.
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
            sys.exit("check that spatial resolution of your two repeat array files are the same' \
            'must fix and try again")
        if round(rez) == round(rez3):
            topo_rez_same = True
        else:
            print('the resolution of your topo.nc file differs from repeat arrays.  It will be resized to' \
            'fit the resolution repeat arrays.  your input file will NOT be changed')
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

        # Create file paths and names
        file_name_dataset1_te_temp = os.path.splitext(os.path.expanduser(self.file_path_dataset1).split('/')[-1])[0]   #everythong after last / without filename ext (i.e. .tif)
        id_date_start = file_name_dataset1_te_temp.index('2')  #find index of date start in file name i.e. find idx of '2' in 'USCATE2019...'
        file_name_dataset1_te_first = os.path.splitext(file_name_dataset1_te_temp)[0][:id_date_start + 8]
        file_name_dataset1_te_second = os.path.splitext(file_name_dataset1_te_temp)[0][id_date_start:]
        file_name_dataset2_te_temp = os.path.splitext(os.path.expanduser(self.file_path_dataset2).split('/')[-1])[0]
        file_name_dataset2_te_first = os.path.splitext(file_name_dataset2_te_temp)[0][:id_date_start + 8]
        file_name_dataset2_te_second = os.path.splitext(file_name_dataset2_te_temp)[0][id_date_start:]
        file_name_dataset1_te = os.path.join(self.file_out_basin, file_name_dataset1_te_first + '_clipped_to_' + file_name_dataset2_te_second + '.tif')
        file_name_dataset2_te = os.path.join(self.file_out_basin, file_name_dataset2_te_first + '_clipped_to_' + file_name_dataset1_te_second + '.tif')

        # list of clipped files that are created in below code, and deleted upon user specification in UserConfig
        # the deleting of files was moved towards end of code to free up memory
        self.remove_clipped_files = remove_clipped_files
        self.new_file_list = [file_name_dataset1_te, file_name_dataset2_te, self.file_path_out_base + '_dem_common_extent.tif', self.file_path_out_base + '_veg_height_common_extent.tif']
                #Check if file already exists

        if not (os.path.exists(file_name_dataset1_te)) & (os.path.exists(file_name_dataset2_te) & (os.path.exists(self.file_path_out_base + '_dem_common_extent.tif'))):
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
                        run_arg3 = 'gdal_translate -of GTiff NETCDF:"{0}":dem {1}'.format(self.file_path_topo, self.file_path_out_base + '_dem.tif')
                        run(run_arg3, shell=True)
                        run_arg3b = 'gdal_translate -of GTiff NETCDF:"{0}":veg_height {1}'.format(self.file_path_topo, self.file_path_out_base + '_veg_height.tif')
                        run(run_arg3b, shell=True)
                    else:
                        run_arg3 = 'gdal_translate -of GTiff -tr {0} {0} NETCDF:"{1}":dem {2}'.format(round(rez), self.file_path_topo, self.file_path_out_base + '_dem.tif')
                        print('Running DEM')
                        run(run_arg3, shell=True)
                        run_arg3b = 'gdal_translate -of GTiff -tr {0} {0} NETCDF:"{1}":veg_height {2}'.format(round(rez), self.file_path_topo, self.file_path_out_base + '_veg_height.tif')
                        print('Running VEG')
                        run(run_arg3b, shell=True)


                    run_arg4 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format(left_max_bound, bottom_max_bound, right_min_bound, top_min_bound,
                                                                            self.file_path_out_base + '_dem.tif', self.file_path_out_base + '_dem_common_extent.tif -overwrite')
                    run_arg4b = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format(left_max_bound, bottom_max_bound, right_min_bound, top_min_bound,
                                                                            self.file_path_out_base + '_veg_height.tif', self.file_path_out_base + '_veg_height_common_extent.tif -overwrite')

                    run(run_arg1, shell = True)
                    run(run_arg2, shell = True)
                    print('running dem_common_extent')
                    run(run_arg4, shell = True)
                    print('running veg_height_common_extent')
                    run(run_arg4b, shell = True)
                    # remove unneeded files
                    run_arg5 = 'rm ' + self.file_path_out_base + '_dem.tif'
                    run_arg5b = 'rm ' + self.file_path_out_base + '_veg_height.tif'
                    run(run_arg5, shell = True)
                    run(run_arg5b, shell = True)
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
            self.mat_clip1_orig = mat_clip1
            mat_clip1[np.isnan(mat_clip1)] = -9999
        with rio.open(file_name_dataset2_te) as src:
            d2_te = src
            mat_clip2 = d2_te.read()  #matrix
            mat_clip2 = mat_clip2[0]
            self.mat_clip2_orig = mat_clip2
            mat_clip2[np.isnan(mat_clip2)] = -9999
        with rio.open(self.file_path_out_base + '_dem_common_extent.tif') as src:
            topo_te = src
            dem_clip = topo_te.read()  #not sure why this pulls the dem when there are logs of
            dem_clip = dem_clip[0]
            dem_clip[np.isnan(dem_clip)] = -9999
            self.dem_clip = dem_clip
            # ensure user_specified DEM resolution is compatible with uint8 i.e. not too fine
            self.check_DEM_resolution()

        self.mat_clip1 = self.get16bit(mat_clip1)
        self.mat_clip2 = self.get16bit(mat_clip2)

    # @profile
    def mask_basic(self):

        #Create master dictionary (key)
        operators = {'less_than':'<', 'greater_than':'>'}

        config_to_mat_object = {'date1' : 'mat_clip1', 'date2' : 'mat_clip2',
                                'difference_normalized' : 'mat_diff_norm',
                                'difference' : 'mat_diff', 'date1_nan' : 'date1_nan'}

        mat_object_to_tif =     {'mat_clip1' : 'depth_{0}'.format(self.date1_string),
                                'mat_clip2' : 'depth_{0}'.format(self.date2_string),
                                'mat_diff_norm' : 'difference_normalized',
                                'mat_diff' : 'difference'}

        #COMBINE dictionaries into NESTED dictionary
        self.keys_master = {'operators' : operators,
                            'config_to_mat_object' : config_to_mat_object,
                            'mat_object_to_tif' : mat_object_to_tif}


        date1_zero = np.absolute(self.mat_clip1) == 0
        date2_zero = np.absolute(self.mat_clip2) == 0

        self.mask_overlap_nan = ~((self.mat_clip1 == -9999) | (self.mat_clip2 == -9999))  # where overlap and no nans

        # zero_and_nan flag
        nan_to_zero = (self.mat_clip1 == -9999) & date2_zero   # m2_nans and m1_zeros
        zero_to_nan = date1_zero & (self.mat_clip2 == -9999)                               # m2_zeros and m1_nans
        self.flag_zero_and_nan = nan_to_zero | zero_to_nan

        mat_diff_norm_nans = self.mat_diff_norm.copy()
        mat_diff_norm_nans[~self.mask_overlap_nan] = np.nan
        self.mat_diff_norm_nans = mat_diff_norm_nans

        snow_present_mask = ~(date1_zero & date2_zero)  #snow present on at least one date
        self.mask_nan_snow_present = snow_present_mask & self.mask_overlap_nan  # combine nan-free mask with snow present mask

        self.all_gain = date1_zero & (np.absolute(self.mat_clip2) > 0)

    def mask_advanced(self, name, action, operation, value):
        """
        Adds new masks to object:
        1) overlap_nans: no nans present in any input matrices
        2) flag_zero_and_nans: nans in one date and zeros in another
        3) flag_extreme_outliers: all comparison conditions for each matrice (action, operation, val) are met
        4) overlap_conditional: where flag_extreme_outliers AND overlap_nans occur

        Arguments
        name: list of strings (1xN), matrix names to base map off of.
        action: Compare or not.  Only two options currently.  Can be expanded in future.
        operation:  (1x2N)list of strings, operation codes to be performed on each matrix
        value: list of floats (1x2N), matching value pairs to operation comparison operators.
        """

        print('Entered mask_advanced')


        shp = getattr(self, self.keys_master['config_to_mat_object'][name[0]]).shape
        # mask_overlap_conditional = np.zeros(shp, dtype = bool)
        for j in range(2):
            if j == 0:
                mask_overlap_conditional = np.ones(shp, dtype = bool)
            id = 2 * j
            mat_str = 'self.{0}'.format(self.keys_master['config_to_mat_object'][name[j]])
            op_str = self.keys_master['operators'][operation[id]]
            op_str2 = self.keys_master['operators'][operation[id + 1]]
            val = str(value[id])
            val2 = str(value[id + 1])
            cmd = '({0} {1} {2}) & ({3} {4} {5})'.format(mat_str, op_str, val, mat_str, op_str2, val2)
            print(cmd)
            temp = eval(cmd)
            mask_overlap_conditional = mask_overlap_conditional & temp
        self.mask_overlap_conditional = mask_overlap_conditional & self.mask_overlap_nan

        # shp = getattr(self, self.keys_master['config_to_mat_object'][name[0]]).shape
        # mask_overlap_nan = np.ones(shp, dtype = bool)
        # mask_overlap_conditional = np.zeros(shp, dtype = bool)
        # extreme_outliers = np.zeros(shp, dtype = bool)
        # mat_ct = 0  # initialize count of mats for conditional masking
        # for i in range(len(name)):
        #     mat = getattr(self, self.keys_master['config_to_mat_object'][name[i]])
        #     # replace nan with -9999 and identify location for output
        #     # if action[i] == 'na':
        #     #     action_temp = False
        #     # else:
        #     #     action_temp = True
        #     if (mat == -9999).any():  # if nans are present and represented by -9999
        #         mat_mask = mat.copy()
        #         temp_nan = mat_mask != -9999  # set nans to -9999
        #         print('In -9999')
        #     else:   # no nans present
        #         mat_mask = mat.copy()
        #         temp_nan = np.ones(shp, dtype = bool)
        #
        #     if action_temp == True:
        #         for j in range(2):
        #             id = mat_ct * 2 + j
        #             op_str = self.keys_master['operators'][operation[id]]
        #             cmd = 'mat_mask' + op_str + str(value[id])
        #             temp = eval(cmd)
        #             mask_overlap_conditional = mask_overlap_conditional | temp
        #             extreme_outliers = extreme_outliers | (~temp)
        #         mat_ct += 1
        #     if i == 1:
        #         # nan_to_zero = ~temp_nan_prev & (np.absolute(mat).round(2)==0.0)   # m2_nans and m1_zeros
        #         nan_to_zero = ~temp_nan_prev & (np.absolute(mat) == 0)   # m2_nans and m1_zeros
        #         zero_to_nan = zero_prev & ~temp_nan                               # m2_zeros and m1_nans
        #         zero_and_nan = nan_to_zero | zero_to_nan
        #     mask_overlap_nan = mask_overlap_nan & temp_nan  # where conditions of comparison are met and no nans present
        #     temp_nan_prev = temp_nan.copy()
        #     # zero_prev = (np.absolute(mat).round(2)==0.0)
        #     zero_prev = np.absolute(mat) == 0
        # self.mask_overlap_nan = mask_overlap_nan  # where overlap and no nans
        # self.flag_zero_and_nan = zero_and_nan
        # self.mask_overlap_conditional = mask_overlap_conditional & mask_overlap_nan
        # self.flag_extreme_outliers = extreme_outliers & mask_overlap_nan
        print('Exited mask_advanced')
    # @profile
    def make_diff_mat(self):
        """
        Saves as attribute a normalized difference matrix of the two input tiffs
        and one with nans (mat_diff_norm, mat_diff_norm_nans). difference between \
        two dates is divided by raw depth on date 1 to normalize.
        """
        mat_clip1 = self.mat_clip1.copy()
        mat_clip2 = self.mat_clip2.copy()
        self.all_loss = (np.absolute(mat_clip1) > 0) & (np.absolute(self.mat_clip2) == 0)  #all loss minimal
        mat_diff = mat_clip2 - mat_clip1  # raw difference
        # self.self.all_loss = (np.absolute(self.mat_clip1).round(2) > 0.0) & (np.absolute(self.mat_clip2).round(2)==0.0)  #all loss minimal
        mat_clip1[~self.all_loss & (mat_clip1 < 25)] = 25  # Set snow depths below 0.25m to 0.25m to avoid dividing by zero
        mat_diff_norm = np.round((mat_diff / mat_clip1), 2)  #
        self.mat_diff_norm = np.ndarray.astype(mat_diff_norm, np.float16)
        self.mat_diff = np.ndarray.astype(mat_diff, np.float16)

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
        mat = self.mask_overlap_nan
        nrows, ncols = self.mask_overlap_nan.shape[0], self.mask_overlap_nan.shape[1]
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
        if ~hasattr(self, 'mask_overlap_nan_trim'):
            self.mask_overlap_nan_trim = self.mask_overlap_nan[idc[0]:idc[1],idc[2]:idc[3]]  # overlap boolean trimmed to nan_extent
        return mat_trimmed_nan
    # @profile
    def save_tiff(self, fname, flags, include_arrays, include_masks):
        """
        saves matix to geotiff using RasterIO basically. Specify one or more matrices in list of strings
        with attribute names, or leave argv blank to get mat_diff_norm_nans along with all outlier types
        as individual bands in multiband tiff.

        Args:
            fname: filename including path of where to save
            flags: string. flags to include as bands in tif
            include_arrays: arrays (i.e. difference) to include as bands in tif
            include_masks: arrays (i.e. overlap_nan) to include as bands in tif

        """

        tick = time.clock()
        # now that calculations complete (i.e. self.mat_diff > 1), set -9999 to nan
        self.mat_diff[~self.mask_nan_snow_present] = np.nan
        self.mat_diff_norm[~self.mask_nan_snow_present] = np.nan
        # invert mask >>  pixels with NO nans both dates to AT LEAST ONE nan
        # self.mask_overlap_nan = ~self.mask_overlap_nan

    # PREPARE bands to be saved to tif
        # split certain flags into 'loss' and 'gain'
        flag_names = []
        for flag in flags:
            if flag not in ['basin_block', 'elevation_block', 'tree']:
                flag_names.append('flag_' + flag)
                # flagged = getattr(self, flag_flags[i])
            else:
                if flag in ['basin_block']:
                    flag_names.extend(('flag_basin_loss', 'flag_basin_gain'))
                elif flag in ['elevation_block']:
                    flag_names.extend(('flag_elevation_loss', 'flag_elevation_gain'))
                elif flag in ['tree']:
                    flag_names.extend(('flag_tree_loss', 'flag_tree_gain'))

        # append arrays and masks to flags list to save to tiff
        if include_masks != None:
            for mask in include_masks:
                flag_names.append('mask_' + mask)

        if include_arrays != None:
            for array in include_arrays:
                flag_names.append(self.keys_master['config_to_mat_object'][array])  # 1)Change here and @2 if desire to save single band

        # finally, change abbreviated object names to verbose, intuitive names
        band_names = self.apply_dict(flag_names, self.keys_master, 'mat_object_to_tif')
        # update metadata to reflect new band count
        self.meta.update({
            'count': len(flag_names)})

        # Write new file
        with rio.open(self.file_path_out_tif, 'w', **self.meta) as dst:
            for id, band in enumerate(flag_names, start = 1):
                try:
                    dst.write_band(id, getattr(self, flag_names[id - 1]))
                    dst.set_band_description(id, band_names[id - 1])
                except ValueError:  # Rasterio has no float16  >> cast to float32
                    mat_temp = getattr(self, flag_names[id - 1])
                    dst.write_band(id, mat_temp.astype('float32'))
                    dst.set_band_description(id, band_names[id - 1])
        tock = time.clock()

        # Now all is complete, DELETE clipped files from clip_extent_overlap()
        # Upon user specification in UserConfig,
        if self.remove_clipped_files == True:
            for file in self.new_file_list:
                run('rm ' + file, shell = True)

        print('save tiff = ', tock - tick, 'seconds')

    def plot_this(self):
        plot_basic(self)

    def check_DEM_resolution(self):
        """
        Ensures that DEM resolution user specified on UserConfig can be partitioned into uint8 datatype
        """
        min_elev, max_elev = np.min(self.dem_clip), np.max(self.dem_clip)
        num_elev_bins = math.ceil((max_elev - min_elev) / self.elevation_band_resolution)
        min_elev_band_rez = math.ceil((max_elev - min_elev) / 254)
        if num_elev_bins > 254:
            print('In the interest of saving memory, please lower (make more coarse)' \
            'your elevation band resolution' \
            'the number of elevation bins must not exceed 254 ' \
            'i.e. (max elevation - min elevation) / elevation band resolution must not exceed 254)' \
            'Enter a new resolution ---> (Must be no finer than{0})'.format(min_elev_band_rez))

            while True:
                response = input()
                try:
                    response = float(response)
                except ValueError:
                    print('must enter a float or integer')
                else:
                    if response > min_elev_band_rez:
                        self.min_elev_band_rez = response
                        print('your new elevation_band_resolution will be: {}. Note that this will NOT be reflected on your backup_config.ini file'.format(response))
                        break
                    else:
                        print('Value still too fine. Enter a new resolution ---> must be no finer than{0})'.format(min_elev_band_rez))

    def get16bit(self, array):
        """
        Converts array into numpy 16 bit integer
        """
        id_nans = array == -9999
        array_cm = np.round(array,2) * 100
        array_cm = np.ndarray.astype(array_cm, np.int16)
        array_cm[id_nans] = -9999
        return(array_cm)

    def apply_dict(self, original_list, dictionary, nested_key):
        keyed_list = []
        for val in original_list:
            try:
                keyed_list.append(dictionary[nested_key][val])
            except KeyError:
                keyed_list.append(val)
        return(keyed_list)

class PatternFilters():
    def init(self):
        pass
    # @profile
    def mov_wind(self, name, size):
        """
         Slow moving window base function which adjusts window sizes to fit along
         matrix edge, as opposed to mov_wind2 which is faster, but trims output (pct) to ~ (M - size) x (N - size),
         if original matrix accessec with 'name' was M x N.
         Beta version of function. Can add other filter/kernels to moving window calculation
         as needed.  Returns pct, which has the proportion of cells/pixels with values > 0 present in each
         moving window centered at target pixel.

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

    # @profile
    def mov_wind2(self, name, size):
        """
        Orders and orders of magnitude faster moving window function than mov_wind().  Uses numpy bitwise operator
        wrapped in sklearn package to 'stride' across matrix cell by cell. Trims output (pct) to ~ (M - size) X (N - size)
        if original matrix accessed by 'name' was M x N.

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
        pct_temp = np.ndarray.astype(patches.sum(axis = (-1, -2))/(size**2), np.float16)
        del patches
        pct_temp = np.reshape(pct_temp, (mat.shape[0] - 2 * base_offset, mat.shape[1] - 2 * base_offset))
        pct = np.zeros(mat.shape, dtype = np.float16)
        pct[base_offset: -base_offset, base_offset : -base_offset] = pct_temp
        return(pct)

class Flags(MultiArrayOverlap, PatternFilters):
    def init(self, file_path_dataset1, file_path_dataset2, file_path_topo, file_out_root, basin, file_name_modifier):
        MultiArrayOverlap.init(self, file_path_dataset1, file_path_dataset2, file_path_topo, file_out_root, basin, file_name_modifier)
    # @profile
    def make_histogram(self, name, nbins, thresh, moving_window_size):
        """
        Creates histogram and finds outliers.  Outlier detection is pretty crude and could benefit from more sophisticated
        2D kernel detecting change. Saves as a flag, self.flag_hist, to object as an attribute which locates histogram outliers

        Args:
            name:  matrix name (string) to access matrix attribute.
            nbins:  1 x 2 list.  Value one is the number of bins in x and Value two is number of bins in y direction
            thresh:  1 x 2 list.  Value one is the proportion of cells within moving window (i.e. 'pct') that must be
                        present to retain in flag_hist_outliers. Value 2 is the minimum bin count to be retained as
                        flag_hist_outlier
            moving_window_size:  Integer.  Size of side of square in pixels of kernel/filter used in moving window.
        """
        # I don't think an array with nested tuples is computationally efficient.  Find better data structure for the tuple_array
        print('entering make_histogram')

        m1 = getattr(self, self.keys_master['config_to_mat_object'][name[0]])
        m2 = getattr(self, self.keys_master['config_to_mat_object'][name[1]])
        m1_nan, m2_nan = m1[self.mask_overlap_conditional], m2[self.mask_overlap_conditional]
        print('m2: ', self.keys_master['config_to_mat_object'][name[1]])
        print(np.sum(np.isnan(m2_nan)))
        bins, xedges, yedges = np.histogram2d(np.ravel(m1_nan), np.ravel(m2_nan), nbins)
        self.xedges, self.yedges= xedges, yedges

        # Now find bin edges of overlapping snow depth locations from both dates, and save to self.bin_loc as array of tuples
        xedges = np.delete(xedges, -1)   # remove the last edge
        yedges = np.delete(yedges, -1)
        bins = np.flip(np.rot90(bins,1), 0)  # WTF np.histogram2d.  hack to fix bin mat orientation
        self.bins = bins
        pct = self.mov_wind(bins, moving_window_size)
        flag_spatial_outlier = (pct < thresh[0]) & (bins > 0)
        print('num of hist outliers ', np.sum(flag_spatial_outlier))
        # flag_bin_ct = (bins < thresh[1]) & (bins > 0)  # ability to filter out bin counts lower than thresh but above zero
        outliers_hist_space = flag_spatial_outlier
        self.outliers_hist_space = outliers_hist_space
        tick = time.clock()
        id = np.where(outliers_hist_space)
        # idx_hist, idy_hist = id[1], id[0]  # column and row id where outliers were found
        idx_bins = np.digitize(m1_nan, xedges) -1  # id of x bin edges.  dims = (N,) array
        idy_bins = np.digitize(m2_nan, yedges) -1  # id of y bin edges
        hist_outliers_temp = np.zeros(idx_bins.shape, dtype = bool)
        for x, y in zip(id[1], id[0]):
                temp = (idx_bins == x) & (idy_bins == y)
                hist_outliers_temp = (idx_bins == x) & (idy_bins == y) | hist_outliers_temp
                # print('hist outliers temp ', np.sum(hist_outliers_temp))
        hist_outliers = np.zeros(m1.shape, dtype = bool)
        hist_outliers[self.mask_overlap_conditional] = hist_outliers_temp
        self.flag_histogram = hist_outliers

        tock = time.clock()
        print('hist2D_with_bins_mapped = ', tock - tick, 'seconds')

    def flag_basin_blocks(self, apply_moving_window, moving_window_size, neighbor_threshold, snowline_threshold):
        """
        Finds cells of complete melt or snow where none existed prior.
        Apply moving window to remove scattered, individual cells and flag larger blocks of cells.
        Used to diagnose extreme, unrealistic change that were potentially processed incorrectly
        Saves flag_basin_gain and flag_basin_loss to object.

        Args:
            moving_window_size:   size of moving window used to define blocks
            neighbor_threshold:   proportion of neighbors within moving window (including target cell) that have
            snowline_threshold:   mean depth of snow in elevation band used to determine snowline

        """
        # Note below ensures -0.0 and 0 and 0.0 are all discovered and flagged as zeros.
        # Checked variations of '==0.0' and found that could have more decimals or be ==0 with same resultant boolean
        # all_gain = (np.absolute(self.mat_clip1).round(2)==0.0) & (np.absolute(self.mat_clip2).round(2) > 0.0)
        # self.all_gain = (np.absolute(self.mat_clip1) == 0) & (np.absolute(self.mat_clip2) > 0)

        if hasattr(self, 'snowline_elev'):
            pass
        else:
            self.snowline(snowline_threshold)

        basin_loss = self.mask_overlap_nan & self.all_loss & (self.dem_clip > self.snowline_elev) #ensures neighbor threshold and overlap, plus from an all_loss cell
        basin_gain = self.mask_overlap_nan & self.all_gain  #ensures neighbor threshold and overlap, plus from an all_gain cell

        if apply_moving_window:
            pct = self.mov_wind2(basin_loss, moving_window_size)
            self.flag_basin_loss = (pct > neighbor_threshold) & self.all_loss
            pct = self.mov_wind2(basin_gain, moving_window_size)
            self.flag_basin_gain = (pct > neighbor_threshold) & self.all_gain
        else:
            self.flag_basin_loss = basin_loss
            self.flag_basin_gain = basin_gain

    # @profile
    def flag_elevation_blocks(self, apply_moving_window, moving_window_size, neighbor_threshold, snowline_threshold, outlier_percentiles,
                    elevation_thresholding):
        """
        More potential for precision than flag_basin_blocks function.  Finds outliers in relation to their elevation bands for snow gain and loss.
        Saves as attributes to object flag_elevation_gain and flag_elevation_loss.  By default, elevation gain finds pixels where BOTH raw snow gained AND
        normalized snow gained were greater than outlier_percentiles.  Elevation loss thresholds ONLY on raw snow lost as relative snow lost
        is -1 or -100% for much of the lower and mid elevations in the melt season, making relative snow loss unreliable.  Both flags are further
        filtered with mov_wind2 to ensure neighboring pixels had similiar behavior.  This is a safeguard against outliers.
        Args:
            moving_window_size:  Same as flag_basin_blocks
            neighbor_threshold:  Same as flag_basin_blocks
            snowline_threshold:  Same as flag_basin_blocks
            outlier_percentiles:  list of four values (raw gain upper, raw loss upper, normalized gain lower, normalized loss lower)
                                    Percentiles used to threshold each elevation band.  i.e. 95 in (95,80,10,10) is the raw gain upper,
                                    which means anything greater than the 95th percentile of raw snow gain in each elevatin band will
                                    be flagged as an outlier.
        """
        print('entering flag_elevation_blocks')
        # Masking bare ground areas because zero change in snow depth will skew distribution from which thresholds are based
        dem_clip_masked = self.dem_clip[self.mask_nan_snow_present]
        mat_diff_norm_masked = self.mat_diff_norm[self.mask_nan_snow_present]
        mat_diff_masked = self.mat_diff[self.mask_nan_snow_present]

        # Will need self.elevation_edges from snowline() if hypsometry has not been run yet
        if hasattr(self, 'snowline_elev'):
            pass
        else:
            self.snowline(snowline_threshold)

        id_dem = np.digitize(dem_clip_masked, self.elevation_edges) -1  #creates bin edge ids.  the '-1' converts to index starting at zero
        id_dem[id_dem == self.elevation_edges.shape[0]]  = self.elevation_edges.shape[0] - 1 #for some there are as many bins as edges.  this smooshes last bin(the max) into second to last bin edge
        id_dem_unique = np.unique(id_dem)  #unique ids
        map_id_dem = np.full(self.mask_nan_snow_present.shape, id_dem_unique[-1] + 1, dtype = np.uint8)  # makes nans max(id_dem) + 1
        map_id_dem[self.mask_nan_snow_present] = id_dem
        map_id_dem_overlap = map_id_dem[self.mask_nan_snow_present]

        # Find threshold values per elevation band - 1D array
        thresh_upper_norm = np.zeros(id_dem_unique.shape, dtype = np.float16)
        thresh_lower_norm = np.zeros(id_dem_unique.shape, dtype = np.float16)
        thresh_upper_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)
        thresh_lower_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)
        elevation_count = np.zeros(id_dem_unique.shape, dtype = np.int16)
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            temp = mat_diff_norm_masked
            temp2 = mat_diff_masked
            temp3 = np.ones(temp2.shape, dtype = bool)
            thresh_upper_raw[id] = np.percentile(temp2[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[0])
            thresh_upper_norm[id] = np.percentile(temp[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[1])
            thresh_lower_raw[id] = np.percentile(temp2[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[2])
            thresh_lower_norm[id] = np.percentile(temp[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[3])
            elevation_count[id] = getattr(temp3[map_id_dem_overlap == id_dem_unique2], 'sum')()
            # elevation_std[id] = getattr(mat_diff_norm_masked[map_id_dem_overlap == id_dem_unique2], 'std')()

        # Place threshold values in appropriate elevation bin onto map - 2D array.  Used to find elevation based outliers
        thresh_upper_norm_array = np.zeros(self.mask_nan_snow_present.shape, dtype=np.float16)
        thresh_lower_norm_array = thresh_upper_norm_array.copy()
        thresh_upper_raw_array = thresh_upper_norm_array.copy()
        thresh_lower_raw_array = thresh_upper_norm_array.copy()
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            try:
                thresh_upper_norm_array[map_id_dem == id_dem_unique2] = thresh_upper_norm[id]
                thresh_upper_raw_array[map_id_dem == id_dem_unique2] = thresh_upper_raw[id]
            except IndexError:
                pass
            try:
                thresh_lower_norm_array[map_id_dem == id_dem_unique2] = thresh_lower_norm[id]
                thresh_lower_raw_array[map_id_dem == id_dem_unique2] = thresh_lower_raw[id]
            except IndexError:
                pass

        # Combine outliers from mat_diff, mat_diff_norm or both mats accoring to UserConfig
        # Dictionary to translate values from UserConfig

        keys_local = {'loss' : {'operator' : 'less', 'flag' : 'flag_elevation_loss',
                        'mat_diff_norm' : thresh_lower_norm_array, 'mat_diff' : thresh_lower_raw_array},
                'gain' : {'operator' : 'greater', 'flag' : 'flag_elevation_gain',
                        'mat_diff_norm' : thresh_upper_norm_array, 'mat_diff' : thresh_upper_raw_array}}

        flag_options = ['loss', 'gain']
        shape_temp = getattr(np, 'shape')(getattr(self, 'mat_diff'))

        for i in range(len(elevation_thresholding)):
            temp_out_init = np.ones(shape = shape_temp, dtype = bool)
            elevation_flag_name = flag_options[i]
            for mat_name in elevation_thresholding[i]:
                mat_name = self.keys_master['config_to_mat_object'][mat_name]
                diff_mat = getattr(self, mat_name)
                elevation_thresh_array = keys_local[elevation_flag_name][mat_name]  # yields thresh_..._array
                temp_out = getattr(np, keys_local[elevation_flag_name]['operator'])(diff_mat, elevation_thresh_array) & temp_out_init
                temp_out_init = temp_out.copy()
            temp_out_init[~self.mask_nan_snow_present] = False

            # Moving Window:
            # Finds pixels idenfied as outliers (temp_out_init) which have a minimum number of neighbor outliers within moving window
            # Note: the '& temp_out_init' ensures that ONLY pixels originally classified as outliers are kept
            if apply_moving_window:
                pct = self.mov_wind2(temp_out_init, moving_window_size)
                temp_out_init = (pct > neighbor_threshold) & temp_out_init
                setattr(self, keys_local[elevation_flag_name]['flag'], temp_out_init.copy())
            # NO Moving Window:
            else:
                setattr(self, keys_local[elevation_flag_name]['flag'], temp_out_init.copy())

        # Save dataframe of elevation band satistics on thresholds
        elev_stack = self.elevation_edges[id_dem_unique -1].ravel()
        # Simply preparing the column names in a syntactically shittilly readable format:
        column_names = ['elevation', '{}% change (m)', '{}% change (norm)', '{}% change (m)', '{}% change (norm)', 'elevation_count']
        column_names_temp = []
        ct = 0
        for id, names in enumerate(column_names):
            if '{}' in names:
                names = names.format(str(outlier_percentiles[ct]))
                ct += 1
            column_names_temp.append(names)

        temp = np.stack((self.elevation_edges[id_dem_unique], np.around(thresh_upper_raw.ravel(), 3), np.around(thresh_upper_norm.ravel(), 3),
                np.around(thresh_lower_raw.ravel(), 3), np.around(thresh_lower_norm.ravel(), 3), elevation_count.ravel()), axis = -1)
        df = pd.DataFrame(temp, columns = column_names_temp)
        df.to_csv(path_or_buf = self.file_path_out_csv, index=False)

    # @profile
    def snowline(self, snowline_threshold):
        """
        Finds the snowline based on the snowline_threshold, or the user specified minimum snow depth.  The lowest elevation band
        with a mean snow depth at the snowline threshold is set as the snowline.  This value is later used to discard outliers below
        snowline.
        Args:
            snowline:       explained above
        """
        # use overlap_nan mask for snowline because we want to get average snow per elevation band INCLUDING zero snow depth
        dem_clip_conditional = self.dem_clip[self.mask_overlap_nan]
        min_elev, max_elev = np.min(dem_clip_conditional), np. max(dem_clip_conditional)

        edge_min = min_elev % self.elevation_band_resolution
        edge_min = min_elev - edge_min
        edge_max = max_elev % self.elevation_band_resolution
        edge_max = max_elev + (self.elevation_band_resolution - edge_max)
        self.elevation_edges = np.arange(edge_min, edge_max + 0.1, self.elevation_band_resolution)
        id_dem = np.digitize(dem_clip_conditional, self.elevation_edges) -1
        id_dem = np.ndarray.astype(id_dem, np.uint8)
        id_dem_unique = np.unique(id_dem)
        map_id_dem = np.full(self.mask_overlap_nan.shape, id_dem_unique[-1] + 1, dtype=np.uint8)  # makes nans max(id_dem) + 1
        map_id_dem[self.mask_overlap_nan] = id_dem
        snowline_mean = np.full(id_dem_unique.shape, -9999, dtype = 'float')
        snowline_std = np.full(id_dem_unique.shape, -9999, dtype = 'float')
        # use the matrix with the deepest mean basin snow depth to base snowline thresh off of.  Assuming deeper avg basin snow = lower snowline
        if np.mean(self.mat_clip1[self.mask_overlap_nan]) > np.mean(self.mat_clip1[self.mask_overlap_nan]):
            mat_temp = self.mat_clip1
            mat = mat_temp[self.mask_overlap_nan]
            mat_temp[self.mask_overlap_nan & mat_temp ]
        else:
            mat_temp = self.mat_clip2
            mat = mat_temp[self.mask_overlap_nan]
        map_id_dem2_masked = map_id_dem[self.mask_overlap_nan]
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            snowline_mean[id] = getattr(mat[map_id_dem2_masked == id_dem_unique2], 'mean')()
            snowline_std[id] = getattr(mat[map_id_dem2_masked == id_dem_unique2], 'std')()
        id_min = np.min(np.where(snowline_mean > snowline_threshold))
        self.snowline_elev = self.elevation_edges[id_min]  #elevation of estimated snowline

        with rio.open(self.file_path_out_base + '_veg_height_common_extent.tif') as src:
            topo_te = src
            veg_height_clip = topo_te.read()  #not sure why this pulls the dem when there are logs of
            self.veg_height_clip = veg_height_clip[0]
        self.veg_presence = self.veg_height_clip > 5

        print(('The snowline was determined to be at {0}m. It was defined as the first elevation band in the basin'
                'with a mean snow depth >= {1}. Elevation bands were in {2}m increments ').
                format(self.snowline_elev, snowline_threshold, self.elevation_band_resolution))

    def flag_tree_blocks(self, logic):

        """
        These are areas with EITHER flag_elevation_gain/loss OR flag_basin_bain/loss with trees present.
        Tree presense is determined from topo.nc vegetation band.
        """

        key = {'loss' : {'basin' : 'flag_basin_loss', 'elevation' : 'flag_elevation_loss', 'flag_tree_name' : 'flag_tree_loss'},
                'gain' : {'basin' : 'flag_basin_gain', 'elevation' : 'flag_elevation_gain', 'flag_tree_name' : 'flag_tree_gain'}}

        flag_options = ['loss', 'gain']

        # get basin and elevation flags if requested
        for i, logic in enumerate(logic):
            tree_flag_name = key[flag_options[i]]['flag_tree_name']
            if 'temp_basin' not in locals():
                try:
                    temp_basin = getattr(self, key[flag_options[i]]['basin'])
                except AttributeError:
                    pass
            if 'temp_elevation' not in locals():
                try:
                    temp_elevation = getattr(self, key[flag_options[i]]['elevation'])
                except AttributeError:
                    pass

            if logic == 'or':
                setattr(self, tree_flag_name, (temp_basin | temp_elevation) & self.veg_presence)
            elif logic == 'and':
                setattr(self, tree_flag_name, (temp_basin & temp_elevation) & self.veg_presence)
            elif logic == 'basin':
                setattr(self, tree_flag_name, temp_basin & self.veg_presence)
            elif logic == 'elevation':
                setattr(self, tree_flag_name, temp_elevation & self.veg_presence)

                # self.flag_tree_loss = (self.flag_basin_loss | self.flag_elevation_loss) & self.veg_presence
        # self.flag_tree_gain = (self.flag_basin_gain | self.flag_elevation_gain) & self.veg_presence
    def __repr__(self):
            return ("Main items of use are matrices clipped to each other's extent and maps of outlier flags \
                    Also capable of saving geotiffs and figures")
