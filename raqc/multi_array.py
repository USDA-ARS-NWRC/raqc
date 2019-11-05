from raqc.raqc_plot import plot_basic
import numpy as np
import rasterio as rio
import copy
import math
from sklearn.feature_extraction import image
import sys, os
from subprocess import run
from netCDF4 import Dataset
import time
import pandas as pd
import json
import affine
from memory_profiler import profile
from .utilities import prep_coords, get_elevation_bins, check_DEM_resolution, \
                        evenly_divisible_extents, get16bit, update_meta_from_json, \
                        apply_dict, create_clipped_file_names
from tabulate import tabulate
import pandas as pd

class MultiArrayOverlap(object):
    def __init__(self, file_path_dataset1, file_path_dataset2, file_path_topo,
                file_out_root, basin, file_name_modifier,
                elevation_band_resolution):
        """
        Initiate self and add attributes needed throughout RAQC run.
        Ensure dates are in chronological order.
        Check if input files have already been clipped. Note: User can either
        provide input file paths that were already clipped,
        i.e. USCATE20180528_to_20180423_SUPERsnow_depth.tif, or use file paths
        of original files, and this will be checked in clip_overlap_extent()

        Args:
            file_path_dataset1:             date1 file path
            file_path_dataset2:             date2 file path
            file_path_topo:                 topo.nc file path
            file_out_root:                  root path to join with basin and
                                            file_name_modifier
            basin:                          Tuolumne, SanJoaquin, etc.
            file_name_modifier:             Links backup_config to file
            elevation_band_resolution:      see .utilities get_elevation_bins
        """

        # Ensure that dataset 1 and dataset2 are in chronological order
        self.date1_string = file_path_dataset1.split('/')[-1].split('_')[0][-8:]
        self.date2_string = file_path_dataset2.split('/')[-1].split('_')[0][-8:]
        check_chronology1 = pd.to_datetime(file_path_dataset1.split('/')[-1]  \
                            .split('_')[0][-8:], format = '%Y%m%d')
        check_chronology2 = pd.to_datetime(file_path_dataset2.split('/')[-1] \
                            .split('_')[0][-8:], format = '%Y%m%d')

        if check_chronology1 < check_chronology2:
            pass
        else:
            sys.exit('Date 1 must occur before Date 2. Please switch Date 1'
                        '\n with Date 2 in UserConfig. Exiting program')

        # Make subdirectory --> file_out_root/basin
        # file_out_basin = os.path.join(file_out_root, basin)
        basin_abbr = file_path_dataset1.split('/')[-1].split('_')[0][:-8]  #basin abbreviation
        year1 = file_path_dataset1.split('/')[-1].split('_')[0][-8:]
        year2 = file_path_dataset2.split('/')[-1].split('_')[0][-8:]
        self.file_path_out_base = os.path.join(file_out_root, basin, year1 + '_' + year2)

        if not os.path.exists(self.file_path_out_base):
            os.makedirs(self.file_path_out_base)

        # Make file paths
        file_name_base = '{0}_{1}'.format(basin_abbr + year1, year2)
        self.file_name_base = os.path.join(self.file_path_out_base, file_name_base)
        self.file_path_out_tif_flags = '{0}_{1}_flags.tif'.format(self.file_name_base, file_name_modifier)
        self.file_path_out_tif_arrays = '{0}_{1}_arrays.tif'.format(self.file_name_base, file_name_modifier)
        self.file_path_out_backup_config = '{0}_{1}_raqc_backup_config.ini'.format(self.file_name_base, file_name_modifier)
        if os.path.exists(self.file_path_out_backup_config):
            print('\n{0}\nThe file: {1}'
                    '\nalready exists,indicating that raqc was previously'
                    '\nrun with a user config including the same input files'
                    '\nand file_name_modifier, "{2}".'
                    '\n\nThis will overwrite files from'
                    '\nprevious run if in existence.'
                    '\n\ntype    "proceed"   to proceed as is'
                    '\nor type   "exit"      to exit and change user config' \
                    .format('-'*60, self.file_path_out_backup_config, file_name_modifier))
            while True:
                response = input('\n')
                if response.lower() == 'proceed':
                    break
                elif response.lower() == 'exit':
                    os.sys.exit('\nexiting\n' + '-'*60)
                else:
                    print('\nresponse not recognized\n'
                    '\ntype    "proceed"   to proceed as is'
                    '\nor      "exit"      to exit and fix user config')
                    pass

        self.file_path_out_csv = '{0}_raqc.csv'.format(self.file_name_base)
        self.elevation_band_resolution = elevation_band_resolution
        self.file_path_out_json = '{0}_metadata.txt'.format(self.file_name_base)
        self.file_path_out_histogram = '{0}_{1}_2D_histogram.png'.format(self.file_name_base, file_name_modifier)
        path_log_file = '{0}_memory_usage.log'.format(self.file_name_base)

        # log_file = open(path_log_file, 'w+')

        # check if user decided to pass clipped files in config file
        # Note: NOT ROBUST for user input error!
        # i.e. if wrong basin ([file][basin]) in UserConfig, problems will emerge
        string_match = 'clipped_to'
        # this will yield 0, 1 or 2, coding for which files were passed
        check_for_clipped =  (string_match in file_path_dataset1) + \
                                (string_match in file_path_dataset2)

        if check_for_clipped == 0:
            self.already_clipped = False
        elif check_for_clipped == 2:
            self.already_clipped = True
        elif check_for_clipped == 1:
            print('it appears one snow date is clipped already and one is not'
                '\n please ensure that both are either clipped or the original \n')
            sys.exit('program will exit for user to fix problem ---')

        # creates file paths for clippeds snow date and topo.nc-derived files
        self.file_path_date1_te, self.file_path_date2_te = \
            create_clipped_file_names(self.file_path_out_base,
                                        file_path_dataset1, file_path_dataset2)
        # clipped (common_extent) topo derived files must also be in directory
        self.file_path_dem_te = self.file_name_base + '_dem_common_extent.tif'
        self.file_path_veg_te = self.file_name_base + '_veg_height_common_extent.tif'

        if self.already_clipped:
            if not (os.path.isfile(self.file_path_dem_te) & \
                    os.path.isfile(self.file_path_veg_te)):
                print(('{0}_dem_common_extent and \n{0}_veg_height_common_extent'
                '\nmust exist to pass clipped files directly in UserConfig').format(self.file_name_base))

                sys.exit('Exiting ---- Ensure all clipped files present or simply'
                '\npass original snow depth and topo files\n')
            else:
                self.file_path_date1_clipped = file_path_dataset1
                self.file_path_date2_clipped = file_path_dataset2
                self.file_name_dem_clipped = self.file_path_dem_te

        # check if clipped files already exist before clipping
        # i.e. user passsed orig file in UserConfig, but clipped files EXIST
        if not self.already_clipped:
            # check for existence of clipped snow files
            if os.path.isfile(self.file_path_date1_te) & os.path.isfile \
                              (self.file_path_date2_te):
                print('\nFiles passed in UserConfig were original snow dept files \
                \nhowever clipped files were detected in file path.\
                \nNo new files need to be created; existing clipped files will \
                \nbe used\n')
                # check for existence of clipped dem and veg from topo.nc
                if (os.path.isfile(self.file_path_dem_te) & \
                    os.path.isfile(self.file_path_veg_te)):
                    topo_clip_present = True
                else:
                    topo_clip_present = False
                    print(('However: \
                    \n{0}_dem_common_extent and \
                    \n{0}_veg_height_common_extent \
                    \nmust exist in order to use clipped files. \
                    \nRun will proceed, and original snow depth and topo files \
                    \nwill be clipped (original files will be retained)\n').
                    format(self.file_name_base))
                # set file paths and indicate if clipped using self.already_clipped
                if topo_clip_present:
                    self.file_path_date1_clipped = self.file_path_date1_te
                    self.file_path_date2_clipped = self.file_path_date2_te
                    self.file_name_dem_clipped = self.file_path_dem_te
                    self.already_clipped = True
                if not topo_clip_present:
                    self.file_path_dataset1 = file_path_dataset1
                    self.file_path_dataset2 = file_path_dataset2
                    self.file_path_topo = file_path_topo

    # @profile
    def clip_extent_overlap(self, remove_clipped_files):
        """
        finds overlapping extent of the input files (Geotiffs).  Adds clipped
        matrices of input files as attributes, and additionally outputs geotiffs
        of each (both snow dates, DEM, and Vegetation).

        Arguments:
            remove_clipped_files:   UserConfig option to delete clipped file
        """

        topo_rez_same, extents_same, min_extents, rez13 = prep_coords( \
                self.file_path_dataset1, self.file_path_dataset2, \
                self.file_path_topo, 'dem')

        # Save JSON
        # save metadata of original date2 to json txt file
        # date1 or date2 should both work?  Vestige from when date mattered..
        with rio.open(self.file_path_dataset2) as src:
            meta2 = src.profile

        with open(self.file_path_out_json, 'w') as outfile:
            json_dict = dict({k:v for k, v in meta2.items() if k != 'crs'})
            # convert crs to integer.
            # crs object is not "serializable" for saving with json.dump
            # will be reincarnated as object using epsg code later
            crs_object = rio.crs.CRS.to_epsg(meta2['crs'])
            json_dict.update({'crs' : crs_object})
            json.dump(json_dict, outfile)

        file_path_dem_te = self.file_name_base + '_dem_common_extent.tif'
        file_path_veg_te = self.file_name_base + '_veg_height_common_extent.tif'
        file_path_dem = self.file_name_base + '_dem.tif'
        file_path_veg = self.file_name_base + '_veg.tif'

        # list of clipped files that are created in below code
        # files deleted upon UserConfig preference
        self.remove_clipped_files = remove_clipped_files
        self.new_file_list = [self.file_path_date1_te, self.file_path_date2_te, \
                                file_path_dem_te, file_path_veg_te]

        # Create strings to run as os.run commands

        # Pull veg and dem from topo.nc and save as .tif
        if topo_rez_same:
            run_arg1 = 'gdal_translate -of GTiff NETCDF:"{0}":dem {1}'.format \
                        (self.file_path_topo, file_path_dem)

            run_arg1b = 'gdal_translate -of GTiff NETCDF:"{0}":veg_height {1}' \
                        .format(self.file_path_topo, file_path_veg)

        # If spatial resolution of DEM differs from snow
        # Pull veg and dem from topo.nc, rescale and save as .tif
        else:
            print(('The resolution of your topo.nc file differs from repeat arrays'
            '\nIt will be resized to fit the resolution repeat arrays.'
            '\nTopo spatial resolution = {0}    &   repeat arrays resolution = {1}'
            '\n Your input files will NOT be changed{2}\n') \
            .format(rez13[2], rez[0], '--'*60))

            run_arg1 = 'gdal_translate -of GTiff -tr {0} {0} NETCDF:"{1}":dem {2}' \
                        .format(round(rez[0]), self.file_path_topo, \
                        file_path_dem)

            run_arg1b ='gdal_translate -of GTiff -tr {0} {0} NETCDF:"{1}":veg_height {2}' \
                        .format(round(rez[0]), self.file_path_topo, \
                        file_path_veg)

        # START Clipping
        if not extents_same:
            # if date1, date2 and topo have different extents  ---> clip
            run_arg2 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format \
                        (*min_extents, self.file_path_dataset1, \
                        self.file_path_date1_te) + ' -overwrite'

            run_arg3 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format \
                        (*min_extents, self.file_path_dataset2, \
                        self.file_path_date2_te) + ' -overwrite'

            run_arg4 = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format \
                        (*min_extents, file_path_dem, \
                        self.file_path_dem_te + ' -overwrite')

            run_arg4b = 'gdalwarp -te {0} {1} {2} {3} {4} {5}'.format( \
                        *min_extents, file_path_veg, \
                        self.file_path_veg_te + ' -overwrite')

        else:
            # if date1, date2 and topo.nc are the same extents, don't waste time
            # just copy
            print('Note: the extents of date1 and date2 were the same and \
            Did NOT need clipping.  File copied and renamed as clipped \
            but is the same')

            run_arg2 = 'cp {} {}'.format(self.file_path_dataset1,
                        self.file_path_date1_te)
            run_arg3 = 'cp {} {}'.format(self.file_path_dataset2,
                        self.file_path_date2_te)
            run_arg4 = 'cp {} {} {}'.format(file_path_dem, self.file_path_dem_te, '-overwrite')
            run_arg4b = 'cp {} {} {}'.format(file_path_veg, self.file_path_veg_te, '-overwrite')

        run(run_arg1, shell=True)
        run(run_arg1b, shell=True)
        run(run_arg2, shell = True)
        run(run_arg3, shell = True)
        run(run_arg4, shell = True)
        run(run_arg4b, shell = True)

        # remove unneeded files
        run_arg5 = 'rm ' + file_path_veg
        run_arg5b = 'rm ' + file_path_veg
        run(run_arg5, shell = True)
        run(run_arg5b, shell = True)

        self.file_path_date1_clipped = self.file_path_date1_te
        self.file_path_date2_clipped = self.file_path_date2_te
        self.file_name_dem_clipped = self.file_path_dem_te

    # @profile
    def mask_basic(self):
        """
        Creates boolean masks and flags needed for multiple methods within RAQC, like
        'mask_overlap_nan', and adds as attributes.  Also creates nested dictionary
        used throughout RAQC to translate verbose config file names from .ini
        file into more parsimonious abbreviations used in code

        Outputs (attributes):
            mask_overlap_nans:       mask - no nans present in any input matrices
            flag_zero_and_nans:      flag - nans in one date and zeros in another
                                        nans set to -9999
            mask_nan_snow_present:   mask - snow is present in both dates, and no nans
            all_gain:                mask from zero snow in date1 to snow of any amount
                                        in date2.
        """
        #Create master dictionary (key)
        operators = {'less_than':'<', 'greater_than':'>'}

        config_to_mat_object = {'date1' : 'mat_clip1', 'date2' : 'mat_clip2',
                                'difference_normalized' : 'mat_diff_norm',
                                'difference' : 'mat_diff', 'date1_nan' : 'date1_nan',
                                'mat_diff_flags_to_median' : 'mat_diff_flags_to_median',
                                'median_elevation' : 'median_elevation'}

        mat_object_to_tif =     {'mat_clip1' : 'depth_{0}'.format(self.date1_string),
                                'mat_clip2' : 'depth_{0}'.format(self.date2_string),
                                'mat_diff_norm' : 'difference_normalized',
                                'mat_diff' : 'difference',
                                'median_elevation' : 'median_diff_elev'}

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

        # mat_diff_norm_nans = self.mat_diff_norm.copy()
        # mat_diff_norm_nans[~self.mask_overlap_nan] = np.nan
        # self.mat_diff_norm_nans = mat_diff_norm_nans

        snow_present_mask = ~(date1_zero & date2_zero)  #snow present on at least one date
        self.mask_nan_snow_present = snow_present_mask & self.mask_overlap_nan  # combine nan-free mask with snow present mask

        self.all_gain = date1_zero & (np.absolute(self.mat_clip2) > 0)

    def mask_advanced(self, name, action, operation, value):
        """
        Creates mask with no nans and no outliers.  Useful for histogram 2D as
        outliers ruin the visualtion.

        Arguments
            name: list of strings (1xN), matrix names to base map off of.
            action: Compare or not.  Only two options currently.  Can be expanded in future.
            operation:  (1x2N)list of strings, operation codes to be performed on each matrix
            value: list of floats (1x2N), matching value pairs to operation comparison operators.

        Output (attribute):
            mask_overlap_conditional:    mask - where no outliers of nans occur
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
            temp = eval(cmd)
            mask_overlap_conditional = mask_overlap_conditional & temp
        self.mask_overlap_conditional = mask_overlap_conditional & self.mask_overlap_nan

        print('Exited mask_advanced')

    # @profile
    def make_diff_mat(self):
        """
        Saves as attribute a normalized difference matrix of the two input tiffs.
        (Difference divided by date 1 raw depth).

        Outputs (attributes):
            all_loss:       mask - indicating snow of any amount in date1 to zero snow on date2
            mat_diff:       matrix - date1 depth - date2 depth
            mat_diff_norm:  matrix - mat_diff / date1
        """
        with rio.open(self.file_path_date1_clipped) as src:
            d1_te = src
            mat_clip1 = d1_te.read()  #matrix
            mat_clip1 = mat_clip1[0]
            mat_clip1[np.isnan(mat_clip1)] = -9999
            mat_clip1 = get16bit(mat_clip1)
            self.mat_clip1 = mat_clip1.copy()
        with rio.open(self.file_path_date2_clipped) as src:
            self.d2_te = src
            self.meta2_te = self.d2_te.profile
            mat_clip2 = self.d2_te.read()  #matrix
            mat_clip2 = mat_clip2[0]
            mat_clip2[np.isnan(mat_clip2)] = -9999
            mat_clip2 = get16bit(mat_clip2)
            self.mat_clip2 = mat_clip2.copy()
        with rio.open(self.file_name_dem_clipped) as src:
            topo_te = src
            dem_clip = topo_te.read()  #not sure why this pulls the dem when there are logs of
            dem_clip = dem_clip[0]
            dem_clip[np.isnan(dem_clip)] = -9999
            self.dem_clip = dem_clip.copy()
            # ensure user_specified DEM resolution is compatible with uint8 i.e. not too fine
            check_DEM_resolution(self.dem_clip, self.elevation_band_resolution)

        self.all_loss = (np.absolute(mat_clip1) > 0) & (np.absolute(mat_clip2) == 0)  #all loss minimal
        mat_diff = mat_clip2 - mat_clip1  # raw difference
        # self.self.all_loss = (np.absolute(self.mat_clip1).round(2) > 0.0) & (np.absolute(self.mat_clip2).round(2)==0.0)  #all loss minimal
        mat_clip1[~self.all_loss & (mat_clip1 < 25)] = 25  # Set snow depths below 0.25m to 0.25m to avoid dividing by zero
        mat_diff_norm = np.round((mat_diff / mat_clip1), 2)  #
        self.mat_diff_norm = np.ndarray.astype(mat_diff_norm, np.float16)
        self.mat_diff = np.ndarray.astype(mat_diff, np.float16)

    # @profile
    def get_buffer(self):
        """
        Returns numbers of cells to buffer tiff to match original size, extents, etc.

        Outputs:
            buffer:     dictionary with buffer values left, right, bottom, top
        """


        # get extents and resolution from json and determine if originally clipped
        d_orig = self.derive_dataset('d2_te')
        if not self.extents_same:
            #If disjoint bounds  --> find num cols and rows to buffer with nans N S E W

            bounds_date2_te = [None] * 4
            bounds_date2_te[0], bounds_date2_te[1] = self.d2_te.bounds.left, self.d2_te.bounds.bottom
            bounds_date2_te[2], bounds_date2_te[3] = self.d2_te.bounds.right, self.d2_te.bounds.top

            rez = d_orig['resolution']
            bounds_date2 = [None] * 4
            bounds_date2[0] = evenly_divisible_extents(d_orig['left'], rez)
            bounds_date2[2] = evenly_divisible_extents(d_orig['right'], rez)
            bounds_date2[1] = evenly_divisible_extents(d_orig['bottom'], rez)
            bounds_date2[3] = evenly_divisible_extents(d_orig['top'], rez)

            buffer={}
            # Notice '*-1'.  Those ensure subsetting flag array into nan array
            # buffers -<buffer> on right and bottom, and +<buffer> top and left
            buffer.update({'left' : round((bounds_date2[0] - bounds_date2_te[0]) / rez) * -1})
            buffer.update({'bottom' : round((bounds_date2[1] - bounds_date2_te[1]) / rez)})
            buffer.update({'right' : round((bounds_date2[2] - bounds_date2_te[2]) / rez) * -1})
            buffer.update({'top' : round((bounds_date2[3] - bounds_date2_te[3]) / rez)})
            for k, v in buffer.items():
                if v == 0:
                    buffer[k] = None
            self.buffer = buffer
            print(buffer)
    def save_tiff(self, fname, flags, include_arrays, include_masks):
        """
        Saves up to two geotiffs using RasterIO basically.  One tiff will be the
        matrices of floats, and the second the masks and flags - booleans (uint8).
        Bands to output will be specified in UserConfig

        Args:
            fname: filename including path of where to save
            flags: string. flags to include as bands in tif
            include_arrays: arrays (i.e. difference) to include as bands in tif
            include_masks: arrays (i.e. overlap_nan) to include as bands in tif

        Outputs:
            file_path_out_tiff_flags:   single or multibanded array which may
                                        include any flag or mask
            file_path_out_tiff_arrays:  single or multibanded array which may
                                        include mat_diff and snow depth matrices
                                        for example
        """

        tick = time.clock()

        # Save all arrays as float32
        # convert to float from int to set NaNs back
        self.mat_clip1 = np.ndarray.astype(self.mat_clip1, np.float32)
        self.mat_clip2 = np.ndarray.astype(self.mat_clip2, np.float32)
        self.median_elevation = np.ndarray.astype(self.median_elevation, np.float32)

        # now that calculations complete (i.e. self.mat_diff > 1), set -9999 to nan
        self.mat_diff[~self.mask_nan_snow_present] = np.nan
        self.mat_diff_norm[~self.mask_nan_snow_present] = np.nan
        self.mat_clip1[self.mat_clip1 == -9999] = np.nan
        self.mat_clip2[self.mat_clip2 == -9999] = np.nan
        self.median_elevation[~self.mask_nan_snow_present] = np.nan
        # invert mask >>  pixels with NO nans both dates to AT LEAST ONE nan
        self.mask_overlap_nan = ~self.mask_overlap_nan

        flag_names = self.get_flag_names(flags)

        # append masks to flags list to save to tiff
        if include_masks != None:
            for mask in include_masks:
                flag_names.append('mask_' + mask)

        # finally, change abbreviated object names to verbose, intuitive names
        band_names = apply_dict(flag_names, self.keys_master, 'mat_object_to_tif')

        self.get_buffer()
        if not self.extents_same:
            buffer = self.buffer
            # buffer flag arrays with nans to fit original date2 array shape
            # nan = <uint> 255
            for id, band in enumerate(flag_names):
                flag_buffer = np.full(self.orig_shape, 255, dtype = 'uint8')
                mat_temp = getattr(self, band)
                flag_buffer[buffer['top'] : buffer['bottom'], buffer['left'] : buffer['right']] = mat_temp
                setattr(self, band, flag_buffer)

            # grab json with original metadata and format some key value pairs
            meta_orig = update_meta_from_json(self.file_path_out_json)
            # update clipped metadata with that of original - extents, resolutition, etc.
            self.meta2_te.update(meta_orig)

        # upate metadata to include number of bands (flags) and uint8 dtype
        self.meta2_te.update({
            'count': len(flag_names),
            'dtype': 'uint8',
            'nodata': 255})

        # Write new file
        with rio.open(self.file_path_out_tif_flags, 'w', **self.meta2_te) as dst:
            for id, band in enumerate(flag_names, start = 1):
                try:
                    dst.write_band(id, getattr(self, flag_names[id - 1]))
                    dst.set_band_description(id, band_names[id - 1])
                except ValueError:  # Rasterio has no float16  >> cast to float32
                    mat_temp = getattr(self, flag_names[id - 1])
                    dst.write_band(id, mat_temp.astype('uint8'))
                    dst.set_band_description(id, band_names[id - 1])

        if include_arrays != None:
            array_names = []
            for array in include_arrays:
                array_names.append(self.keys_master['config_to_mat_object'][array])  # 1)Change here and @2 if desire to save single band

            if not self.extents_same:
                buffer = self.buffer
                for id, band in enumerate(array_names):
                    array_buffer = np.full(self.orig_shape, -9999, dtype = 'float32')
                    mat_temp = getattr(self, band)
                    array_buffer[buffer['top'] : buffer['bottom'], \
                                buffer['left'] : buffer['right']] = mat_temp
                    setattr(self, band, array_buffer)

            # finally, change abbreviated object names to verbose, intuitive names
            band_names = apply_dict(array_names, self.keys_master, 'mat_object_to_tif')

            # update metadata to reflect new band count
            self.meta2_te.update({
                'count': len(array_names),
                'dtype': 'float32',
                'nodata': -9999})

            with rio.open(self.file_path_out_tif_arrays, 'w', **self.meta2_te) as dst:
                for id, band in enumerate(array_names, start = 1):
                    print('band ', band)
                    # try:
                    #     print('which arrays: ', array_names[id - 1])
                    #     dst.write_band(id, getattr(self, array_names[id - 1]))
                    #     dst.set_band_description(id, band_names[id - 1])
                    # except ValueError:  # Rasterio has no float16  >> cast to float32
                    mat_temp = getattr(self, array_names[id - 1])
                    dst.write_band(id, mat_temp.astype('float32'))
                    dst.set_band_description(id, band_names[id - 1])

        self.flag_names = flag_names
        self.array_names = array_names

        # Now all is complete, DELETE clipped files from clip_extent_overlap()
        # Upon user specification in UserConfig,
        try:
            if self.remove_clipped_files == True:
                for file in self.new_file_list:
                    run('rm ' + file, shell = True)
        except AttributeError:  #occurs if user passed clipped files through config
            pass

        tock = time.clock()
        print('save tiff = ', round(tock - tick, 2), 'seconds')

    def get_flag_names(self, flags):
        """
        PREPARE bands to be saved to tif
        To simplify UserConfig, options were 'basin_block' 'elevation_block' and/or 'tree'
        Each of these values yield 'loss' and 'gain' flags for each.
        Below code simply parses the three user config values into loss and gain
        flag names

        Arguments:
            flags:  flag names
        Outputs:
            flag_names:     list of flag names
        """
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
        return(flag_names)

    def plot_this(self, action):
        plot_basic(self, action, self.file_path_out_histogram)

    def derive_dataset(self, dataset_clipped_name):
        """
        Unpacks, formats original metadata from json file and returns
        Also determines self.extents_same

        Outputs:
            orig_shape:         dimensions of original date2 input tif
            extents_same:       boolean - was original file clipped
            orig_extents_rez:   extents and resolution of orig date2 input tif
        """

        orig_extents_rez = {}
        with open(self.file_path_out_json) as json_file:
            meta_orig = json.load(json_file)

        temp_affine = meta_orig['transform'][:6]
        rez = temp_affine[0]
        left = temp_affine[2]
        top = temp_affine[5]
        right = left + meta_orig['width'] * rez
        bottom = top - meta_orig['height'] * rez

        # load clipped dataset
        meta_clip = getattr(self, dataset_clipped_name)

        # determine if original extents were same
        self.extents_same = (meta_clip.bounds.left == left) & \
                            (meta_clip.bounds.right == right) & \
                            (meta_clip.bounds.top == top) & \
                            (meta_clip.bounds.bottom == bottom)
        orig_extents_rez.update({'left' : left, 'right' : right, 'top' : top, \
                        'bottom' : bottom, 'resolution' : rez})

        # save size of original array as tuple to get shape
        orig_shape = []
        orig_shape.extend([round((top - bottom)/rez), round((right - left)/rez)])
        self.orig_shape = tuple(orig_shape)
        return(orig_extents_rez)

class PatternFilters():
    def init(self):
        pass
    # @profile
    def mov_wind(self, name, size):
        """
         Moving window operation which adjusts window sizes to fit along
         matrix edges. As opposed to mov_wind2 which is faster, but trims yields
         an output matrix (pct) smaller than the input matrix.
         From Input Size = M x N ---> Output Size = (M - size) x (N - size).
         Beta version of function. Can add other filter/kernels to moving window
         calculation as needed.

         Args:
            name:  matrix name (string) to access matrix attribute. Matrix must
                    be a boolean.
            size:  moving window size - i.e. size x size.  For example, if size = 5,
                    moving window is a 5x5 cell square.
        Returns:
            pct:    Proportion of cells within moving window that have neighboring
                    cells with values.
                    For example a 5 x 5 window with 5 of 25 cells having a value,
                    target cell included, would have a pct = 5/25 = 0.20.
                    The target cell is at the center cell at row 3 col 3 in this case.
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
                #NOTE:  this can be CUSTOMIZED! Instead of pct, calculate other metric
                pct[i,j] = (np.sum(sub > 0)) / sub.shape[0]
        tock = time.clock()
        print('mov_wind zach version = ', round(tock - tick, 2), 'seconds')
        return(pct)

    # @profile
    def mov_wind2(self, name, size):
        """
        Orders and orders of magnitude faster moving window function than mov_wind().
        Uses numpy bitwise operator wrapped in sklearn package to 'stride' across
        matrix cell by cell. Trims output (pct) size from input size of M x N to
        (M - size) X (N - size).

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
    def init(self, file_path_dataset1, file_path_dataset2, file_path_topo,
            file_out_root, basin, file_name_modifier, elevation_band_resolution):
        """
        Protozoic attempt of use of inheritance
        """
        MultiArrayOverlap.init(self, file_path_dataset1, file_path_dataset2,
                                file_path_topo, file_out_root, basin,
                                file_name_modifier)

    # @profile
    def make_histogram(self, name, nbins, thresh, moving_window_size):
        """
        Creates 2D histogram and finds outliers.  Outlier detection is pretty crude
        and could benefit from more sophisticated 2D kernel detecting change.
        Saves outlier flags as attribute ---> self.flag_hist.

        Args:
            name:               matrix name (string) to access matrix attribute.
            nbins:              1 x 2 list.  First value is the number of x axis bins
                                Second value is the number of y axis bins
            thresh:             1 x 2 list.  First value is threshold of the proportion
                                of cells within moving window (i.e. 'pct') required to qualify
                                as a flag. If pct < thresh, flag = True
                                The second value is the minimum bin count of outlier
                                in histogram space. i.e. user may want only bins with
                                counts > 10 for instance.
            moving_window_size:  Integer.  Size of side of square in pixels of
                                kernel/filter used in moving window.

        Outputs:
            flag_histogram:     flags in map space where meeting x, y values of flags
                                from histogram space (i.e. self.outliers_hist_space)
        """
        # I don't think an array with nested tuples is computationally efficient.  Find better data structure for the tuple_array
        print('entering make_histogram')

        m1 = getattr(self, self.keys_master['config_to_mat_object'][name[0]])
        m2 = getattr(self, self.keys_master['config_to_mat_object'][name[1]])
        m1_nan, m2_nan = m1[self.mask_overlap_conditional], m2[self.mask_overlap_conditional]
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
        hist_outliers = np.zeros(m1.shape, dtype = bool)
        hist_outliers[self.mask_overlap_conditional] = hist_outliers_temp
        self.flag_histogram = hist_outliers

        tock = time.clock()
        print('hist2D_with_bins_mapped = ', round(tock - tick, 2), 'seconds')

    def flag_basin_blocks(self, apply_moving_window, moving_window_size, \
                            neighbor_threshold, snowline_threshold):
        """
        Finds cells of complete melt or snow where none existed prior.
        Apply moving window to remove scattered and isolated cells, ineffect
        highlighting larger blocks of cells.  Intended to diagnose extreme, unrealistic
        change that were potentially processed incorrectly by ASO.

        Args:
            apply_moving_window:  Boolean.  Moving window is optional.
            moving_window_size:   size of moving window used to define blocks
            neighbor_threshold:   proportion of neighbors within moving window (including target cell) that have
            snowline_threshold:   mean depth of snow in elevation band used to determine snowline

        Outputs:
            flag_basin_gain:      all_gain blocks
            flag_basin_loss:      all_loss_blocks

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
    def flag_elevation_blocks(self, apply_moving_window, moving_window_size,
            neighbor_threshold, snowline_threshold, outlier_percentiles,
            elevation_thresholding):
        """
        More potential for precision than flag_basin_blocks function.  Finds outliers
        in relation to their elevation bands for snow gain and loss and adds as attributed
        By default in CoreConfig, elevation gain finds pixels where BOTH raw snow gained AND
        normalized snow gained were greater than outlier_percentiles.
        Alternatively, Elevation loss uses ONLY raw snow lost as relative snow lost
        can be total  (i.e. -1 or -100%) for much of the lower and mid elevations during the melt season,
        making relative snow loss an unreliable indicator.  Apply moving window to
        remove scattered and isolated cells, ineffect highlighting larger blocks of cells.
        Elevation band outlier statistics saved to DataFrame, which can be output
        as CSV per UserConfig request.

        Args:
            apply_moving_window: Boolean. Moving window is optional
            moving_window_size:  Same as flag_basin_blocks
            neighbor_threshold:  Same as flag_basin_blocks
            snowline_threshold:  Same as flag_basin_blocks
            outlier_percentiles:  list of four values (raw gain upper, raw loss upper, normalized gain lower, normalized loss lower)
                                    Percentiles used to threshold each elevation band.  i.e. 95 in (95,80,10,10) is the raw gain upper,
                                    which means anything greater than the 95th percentile of raw snow gain in each elevatin band will
                                    be flagged as an outlier.
            elevation_thresholding:  string. A one or two item list indicating whether to
                                     use diff and/or diff_norm to threshold the
                                     elevation flags
        Output:
            flag_elevation_loss:  attribute
            flag_elevation_gain   attribute
        """
        print('entering flag_elevation_blocks')

        # Masking bare ground areas because zero change in snow depth will skew
        # distribution from which thresholds are based
        mask = copy.deepcopy(self.mask_overlap_nan)
        mat_diff_norm_masked = self.mat_diff_norm[mask]
        mat_diff_masked = self.mat_diff[mask]

        # Will need self.elevation_edges from snowline() if hypsometry has not
        # been run yet
        if hasattr(self, 'snowline_elev'):
            pass
        else:
            self.snowline(snowline_threshold)

        map_id_dem, id_dem_unique, elevation_edges = \
            get_elevation_bins(self.dem_clip, mask, \
                                self.elevation_band_resolution)

        map_id_dem_overlap = map_id_dem[mask]

        # Find threshold values per elevation band - 1D array
        # Initiate numpy 1D arrays for these elevation bin statistics:
            # 1) upper, lower and median outlier thresholds for snow depth
            # 2) upper and lower outlier thresholds for normalized snow depth
            # 3) elevation_count = bins per elevation band
        thresh_upper_norm = np.zeros(id_dem_unique.shape, dtype = np.float16)
        thresh_lower_norm = np.zeros(id_dem_unique.shape, dtype = np.float16)
        thresh_upper_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)
        thresh_lower_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)
        elevation_count = np.zeros(id_dem_unique.shape, dtype = np.int64)
        median_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)
        temp = mat_diff_norm_masked
        temp2 = mat_diff_masked
        temp3 = np.ones(temp2.shape, dtype = bool)
        # save bin statistics per elevation bin to a numpy 1D Array i.e. list
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            thresh_upper_raw[id] = np.percentile(temp2[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[0])
            thresh_upper_norm[id] = np.percentile(temp[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[1])
            thresh_lower_raw[id] = np.percentile(temp2[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[2])
            thresh_lower_norm[id] = np.percentile(temp[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[3])
            median_raw[id] = np.percentile(temp2[map_id_dem_overlap == id_dem_unique2], 50)
            elevation_count[id] = getattr(temp3[map_id_dem_overlap == id_dem_unique2], 'sum')()
            # elevation_std[id] = getattr(mat_diff_norm_masked[map_id_dem_overlap == id_dem_unique2], 'std')()

        # Place threshold values onto map in appropriate elevation bin
        # Used to find elevation based outliers
        thresh_upper_norm_array = np.zeros(mask.shape, dtype=np.float16)
        thresh_lower_norm_array = thresh_upper_norm_array.copy()
        thresh_upper_raw_array = thresh_upper_norm_array.copy()
        thresh_lower_raw_array = thresh_upper_norm_array.copy()
        thresh_median_raw_array = thresh_upper_norm_array.copy()
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            id_bin = map_id_dem == id_dem_unique2
            try:
                thresh_upper_norm_array[id_bin] = thresh_upper_norm[id]
                thresh_upper_raw_array[id_bin] = thresh_upper_raw[id]
                thresh_lower_norm_array[id_bin] = thresh_lower_norm[id]
                thresh_lower_raw_array[id_bin] = thresh_lower_raw[id]
                thresh_median_raw_array[id_bin] = median_raw[id]
            except IndexError as e:
                print(e)
        # save as attribute for plotting purposes
        self.median_elevation = thresh_median_raw_array

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
                # this translates UserConfig name to name (string) used here:
                # i.e. difference = mat_diff and vice/versa
                mat_name = self.keys_master['config_to_mat_object'][mat_name]
                diff_mat = getattr(self, mat_name)
                elevation_thresh_array = keys_local[elevation_flag_name][mat_name]  # yields thresh_..._array
                temp_out = getattr(np, keys_local[elevation_flag_name]['operator'])(diff_mat, elevation_thresh_array) & temp_out_init
                temp_out_init = temp_out.copy()
            temp_out_init[~mask] = False

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
        # Simply preparing the column names in a syntactically shittilly readable format:
        column_names = ['elevation', '{}% change (m)', '{}% change (norm)', '{}% change (m)', '{}% change (norm)', '50% change (m)', 'elevation_count']
        column_names_temp = []
        ct = 0
        for names in column_names:
            if '{}' in names:
                names = names.format(str(outlier_percentiles[ct]))
                ct += 1
            column_names_temp.append(names)

        temp = np.stack((elevation_edges[id_dem_unique], thresh_upper_raw.ravel(), thresh_upper_norm.ravel(),
                thresh_lower_raw.ravel(), thresh_lower_norm.ravel(), median_raw.ravel(), elevation_count.ravel()), axis = -1)
        temp = np.around(temp, 2)
        df = pd.DataFrame(temp, columns = column_names_temp)
        df.to_csv(path_or_buf = self.file_path_out_csv, index=False)

    # @profile
    def snowline(self, snowline_threshold):
        """
        Finds the snowline based on the snowline_threshold. The lowest elevation
        band with a mean snow depth >= the snowline threshold is set as the snowline.
        This value is later used to discard outliers below snowline.
        Additionaly, tree presence is determined from the topo.nc file and saved
        as attribute.
        Args:
            snowline_threshold:     Minimum average snow depth within elevation band,
                                    which defines snowline_elev.  In Centimeters
        Output:
            snowline_elev:      Lowest elevation where snowline_threshold is
                                first encountered
            veg_presence:       cells with vegetation height > 5m from the topo.nc file.
        """

        map_id_dem, id_dem_unique, elevation_edges = \
            get_elevation_bins(self.dem_clip, self.mask_overlap_nan, \
                                self.elevation_band_resolution)

        # initiate lists (<numpy arrays>) used to determine snowline
        snowline_mean = np.full(id_dem_unique.shape, -9999, dtype = 'float')
        # use the matrix with the deepest mean basin snow depth to determine
        # snowline thresh,  assuming deeper avg basin snow = lower snowline
        if np.mean(self.mat_clip1[self.mask_overlap_nan]) > np.mean(self.mat_clip1[self.mask_overlap_nan]):
            #this uses date1 to find snowline
            mat_temp = self.mat_clip1
            mat = mat_temp[self.mask_overlap_nan]
        else:
            #this uses date2 to find snowline
            mat_temp = self.mat_clip2
            mat = mat_temp[self.mask_overlap_nan]
        # Calculate mean for pixels with no nan (nans create errors)
        # in each of the elevation bins
        map_id_dem2_masked = map_id_dem[self.mask_overlap_nan]
        print(id_dem_unique)
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            snowline_mean[id] = getattr(mat[map_id_dem2_masked == id_dem_unique2], 'mean')()
        # Find SNOWLINE:  first elevation where snowline occurs
        id_min = np.min(np.where(snowline_mean > snowline_threshold))
        self.snowline_elev = elevation_edges[id_min]  #elevation of estimated snowline

        # Open veg layer from topo.nc and identify pixels with veg present (veg_height > 5)
        with rio.open(self.file_path_veg_te) as src:
            topo_te = src
            veg_height_clip = topo_te.read()  #not sure why this pulls the dem when there are logs of
            self.veg_height_clip = veg_height_clip[0]
        self.veg_presence = self.veg_height_clip > 5

        print(('The snowline was determined to be at {0}m.'
                '\nIt was defined as the first elevation band in the basin'
                '\nwith a mean snow depth >= {1}.'
                ' \nElevation bands were in {2}m increments ').
                format(self.snowline_elev, snowline_threshold, self.elevation_band_resolution))

    def flag_tree_blocks(self, logic):

        """
        These are cells with EITHER flag_elevation_gain/loss AND/OR
        flag_basin_bain/loss with trees present.
        Tree presense is determined from topo.nc vegetation band.

        Args:
            logic:  tree_loss and tree_gain UserConfig values i.e. use elevation,
                    basin, and, or
        """

        key = {'loss' : {'basin' : 'flag_basin_loss', 'elevation' : 'flag_elevation_loss', 'flag_tree_name' : 'flag_tree_loss'},
                'gain' : {'basin' : 'flag_basin_gain', 'elevation' : 'flag_elevation_gain', 'flag_tree_name' : 'flag_tree_gain'}}

        flag_options = ['loss', 'gain']

        # get basin and elevation flags if requested
        for i, logic in enumerate(logic):
            tree_flag_name = key[flag_options[i]]['flag_tree_name']
            temp_basin = getattr(self, key[flag_options[i]]['basin'])
            temp_elevation = getattr(self, key[flag_options[i]]['elevation'])

            if logic == 'or':
                setattr(self, tree_flag_name, (temp_basin | temp_elevation) & self.veg_presence)
            elif logic == 'and':
                setattr(self, tree_flag_name, (temp_basin & temp_elevation) & self.veg_presence)
            elif logic == 'basin':
                setattr(self, tree_flag_name, temp_basin & self.veg_presence)
            elif logic == 'elevation':
                setattr(self, tree_flag_name, temp_elevation & self.veg_presence)

    def stats_report(self, flags):
        map_id_dem, id_dem_unique, elevation_edges = \
            get_elevation_bins(self.dem_clip, self.mask_nan_snow_present, \
                                self.elevation_band_resolution)

        flags = self.get_flag_names(flags)
        mat_diff = self.mat_diff.copy()
        delta, cell_count, pct_coverage = [], [], []
        for flag_name in flags:
            row = []
            flag = getattr(self, flag_name)
            mask_temp = self.mask_nan_snow_present & ~flag
            mat_diff_clip = mat_diff[mask_temp]
            map_id_dem_clip = map_id_dem[mask_temp]

            median_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)
            # save bin statistics per elevation bin to a numpy 1D Array i.e. list
            for id, id_dem_unique2 in enumerate(id_dem_unique):
                median_raw[id] = np.percentile(mat_diff_clip[map_id_dem_clip == \
                                                            id_dem_unique2], 50)

            # Place threshold values onto map in appropriate elevation bin
            # Used to find elevation based outliers
            thresh_median_array = np.zeros(mat_diff.shape, dtype=np.float16)
            for id, id_dem_unique2 in enumerate(id_dem_unique):
                id_bin = map_id_dem == id_dem_unique2
                try:
                    thresh_median_array[id_bin] = median_raw[id]
                except IndexError as e:
                    print(e)

            mat_diff_flags_to_median = mat_diff.copy()
            mat_diff_flags_to_median[flag] = thresh_median_array[flag]
            sum1 = np.sum(np.ndarray.astype(mat_diff[self.mask_nan_snow_present], np.double))
            sum2 = np.sum(np.ndarray.astype(mat_diff_flags_to_median[self.mask_nan_snow_present], np.double))
            delta_temp = round(100 * ((sum1 - sum2) / sum1), 1)
            delta.append('{}%'.format(delta_temp))
            # total cells in flag
            cell_count_temp = np.sum(flag)
            cell_count.append(cell_count_temp)
            pct_coverage_temp = cell_count_temp / np.sum(self.mask_nan_snow_present)
            pct_coverage_temp = round((pct_coverage_temp * 100),1)
            pct_coverage.append('{}%'.format(pct_coverage_temp))
        print('\n\n')
        df = pd.DataFrame({'flag' : flags, 'cell count' : cell_count, 'percent coverage' : pct_coverage, '\u0394' : delta})
        df = df[['flag', 'cell count', 'percent coverage', '\u0394']]
        print(tabulate(df, headers='keys', colalign=["left", "left", "right", "left", "right"], tablefmt = "github"))
        print('\n\n')
