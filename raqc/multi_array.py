import logging
# solution to almost eliminating Rastrio logging to log file
logging.getLogger('rasterio').setLevel(logging.WARNING)
from inicheck.output import print_config_report
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
                        apply_dict, create_clipped_file_names, format_date, \
                        snowline, return_snow_files, \
                        debug_fctn
from tabulate import tabulate
import pandas as pd
import datetime
from . import plotables as pltz
import matplotlib.pyplot as plt
import coloredlogs

class MultiArrayOverlap(object):
    def __init__(self, file_path_dataset1, file_path_dataset2, file_path_topo,
                file_out_root, file_path_snownc, basin, file_name_modifier,
                elevation_band_resolution):
    # def __init__(self):
        """
        Initiate self and add attributes needed throughout RAQC run.
        Ensure dates are in chronological order.
        Check if input files have already been clipped. Note: User can either
        provide input file paths that were already clipped,
        i.e. USCATE20180528_to_20180423_SUPERsnow_depth.tif, or use file paths
        of original files, and this will be checked in clip_overlap_extent()
        Functionality was added later that automatically finds clipped files
        (based off of file name), and uses them if present

        Args:
            file_path_dataset1:             date1 file path
            file_path_dataset2:             date2 file path
            file_path_topo:                 topo.nc file path
            file_out_root:                  root path to join with basin and
                                            file_name_modifier
            basin:                          Tuolumne, SanJoaquin, etc.
            file_name_modifier:             Links backup_config to file
            elevation_band_resolution:      see .utilities get_elevation_bins
            file_path_snow_nc:              file path containing snow.nc files
                                            from model outputs
        """
        # 1) GET DATE STRINGS
        # Get date strings AND if lidar flight(.tif):
        # ensure that dataset 1 and dataset2 are in chronological order
        if os.path.splitext(file_path_dataset1)[1] == '.tif':
            self.date1_string = file_path_dataset1.split('/')[-1].split('_')[0][-8:]
            self.date2_string = file_path_dataset2.split('/')[-1].split('_')[0][-8:]
            check_chronology1 = pd.to_datetime(self.date1_string, format = '%Y%m%d')
            check_chronology2 = pd.to_datetime(self.date2_string, format = '%Y%m%d')

            if check_chronology1 < check_chronology2:
                pass
            else:
                sys.exit('Date 1 must occur before Date 2. Please switch Date 1'
                            '\n with Date 2 in UserConfig. Exiting program')

        # if date1 is a snow.nc file
        else:
            self.date1_string = file_path_dateset1.split('/')[-2][3:]
            self.date2_string = file_path_dataset2.split('/')[-1].split('_')[0][-8:]

        # 2) CREATE FILEPATHS
        # Make subdirectory --> file_out_root/basin
        # file_out_basin = os.path.join(file_out_root, basin)
        # first_flight: Change basin_abbrev to file_path_dataset2
        basin_abbr = file_path_dataset2.split('/')[-1].split('_')[0][:-8]  #basin abbreviation
        self.file_path_out_base = os.path.join(file_out_root, basin, self.date1_string + '_' + self.date2_string)

        # Create directory for all output files
        if not os.path.exists(self.file_path_out_base):
            os.makedirs(self.file_path_out_base)

        # Make file paths
        file_name_base = '{0}_{1}'.format(basin_abbr + self.date1_string, self.date2_string)
        self.file_name_base = os.path.join(self.file_path_out_base, file_name_base)
        self.file_path_out_backup_config = '{0}_{1}_raqc_backup_config.ini'.format(self.file_name_base, file_name_modifier)

        # If backup_config of same name in existence, inform user that files
        # have previously been generated under same name.
        # Confirm that they want to overwrite these files.
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
                    sys.exit('\nexiting\n' + '-'*60)
                else:
                    print('\nresponse not recognized\n'
                    '\ntype    "proceed"   to proceed as is'
                    '\nor      "exit"      to exit and fix user config')
                    pass

        # ...continue making file paths
        self.file_path_snownc = file_path_snownc
        self.file_path_out_tif_flags = '{0}_{1}_flags.tif'.format(self.file_name_base, file_name_modifier)
        self.file_path_out_tif_arrays = '{0}_{1}_arrays.tif'.format(self.file_name_base, file_name_modifier)
        self.file_path_out_csv = '{0}_raqc.csv'.format(self.file_name_base)
        self.elevation_band_resolution = elevation_band_resolution
        self.file_path_out_json = '{0}_metadata.txt'.format(self.file_name_base)
        self.file_path_out_histogram = '{0}_{1}_2D_histogram.png'.format(self.file_name_base, file_name_modifier)

        # 3) LOGGING - set up logging
        self.log_level = 'debug'
        self.log_level = self.log_level.upper()
        level = logging.getLevelName(self.log_level)

        # add a custom format for logging
        fmt = "%(levelname)s: %(msg)s"

        # Always write log file to <output>/log.txt
        log_file = self.file_name_base + '_log.txt'
        self.log = logging.getLogger(__name__)

        # Log to file, no screen output.
        logging.basicConfig(filename=log_file, filemode='w+',
                                            level=self.log_level,
                                            format=fmt)

        # colored screen logging
        coloredlogs.install(logger=self.log, level=self.log_level,
                                            fmt=fmt)

        # 4) CLIPPED FILES in CONFIG? check if clipped files passed to config
        # Note: NOT ROBUST for user input error!
        # i.e. if wrong basin ([file][basin]) in UserConfig, problems may emerge

        # yields 0, 1 or 2, coding for which files were passed
        string_match = 'clipped_to'
        check_for_clipped =  (string_match in file_path_dataset1) + \
                                (string_match in file_path_dataset2)
        # date1 and date2 were not clipped
        if check_for_clipped == 0:
            self.already_clipped = False
        # date 1 and date2 were clipped
        elif check_for_clipped == 2:
            self.already_clipped = True
        # only one date was clipped
        elif check_for_clipped == 1:
            print('it appears one snow date is clipped already and one is not'
                '\n please ensure that both are either clipped or the original \n')
            sys.exit('program will exit for user to fix problem ---')

        # clipped (common_extent) topo derived files must also be in directory
        self.file_path_dem_te = self.file_name_base + '_dem_common_extent.tif'
        self.file_path_veg_te = self.file_name_base + '_veg_height_common_extent.tif'
        self.file_path_date1_te = file_path_dataset1
        self.file_path_date2_te = file_path_dataset2

        # check if clipped veg and dem files exist in base folder
        if self.already_clipped:
            if not (os.path.isfile(self.file_path_dem_te) & \
                    os.path.isfile(self.file_path_veg_te)):
                print(('{0}_dem_common_extent and \n{0}_veg_height_common_extent'
                '\nmust exist to pass clipped files directly in UserConfig').format(self.file_name_base))

                sys.exit('Exiting ---- Ensure all clipped files present or simply'
                '\npass original snow depth and topo files\n')
            else:
                pass

        # 5) CLIPPED FILES EXIST but not passed in Config
        # i.e.orig file passed in UserConfig, but clipped exist in directory
        log_msg1 = '\nFiles passed in UserConfig were original \
                \nsnow depth files however clipped files were detected in \
                \nfile path.  No new files need to be created; existing  \
                \nclipped files will be used\n'
        log_msg2 = '\nHowever: \
                \n{0}_dem_common_extent AND \
                \n{0}_veg_height_common_extent \
                \nmust exist in order to use clipped files. \
                \nRun will proceed. Topo files will be rescaled and clipped \
                \n(original files will be retained) to match extents and rez \
                \nof clipped snow files passed in UserConfig\n'. \
                format(self.file_name_base)
        # User did not passed clipped files, but below code block automatically
        # detects the existence of clipped files based on file name
        if not self.already_clipped:
            # creates file paths for clipped snow date and topo.nc-derived files
            # check for clipped snow files even if not passed in UserConfig
            self.file_path_date1_te, self.file_path_date2_te = \
                create_clipped_file_names(self.file_path_out_base,
                                        file_path_dataset1, file_path_dataset2)

            if os.path.isfile(file_path_date1_te) & os.path.isfile \
                              (file_path_date2_te):
                self.log.info(log_msg1)

                # check for existence of clipped dem and veg from topo.ncexi
                if (os.path.isfile(self.file_path_dem_te) & \
                    os.path.isfile(self.file_path_veg_te)):
                    self.already_clipped = True

                # while clipped snow files were present, topo-derived files
                # were not (i.e. dem and veg clipped). This will set
                # the already_clipped attribute to False and via cli.py, run
                # clip_extent_overlay method to clip topo derived files
                else:
                    self.log.info(log_msg2)

        # Unncessary else statement included to explicitly comment logic:
        # Clipped files were not passed to UserConfig and indeed
        # they did not exist in run directory
        else:
            pass

        # These paths only needed for clip_extent_overlap, but setting here
        # makes codeset more parsimonious
        self.file_path_dataset1 = file_path_dataset1
        self.file_path_dataset2 = file_path_dataset2
        self.file_path_topo = file_path_topo

    debug_fctn()


    # @profile
    def clip_extent_overlap(self, remove_clipped_files):
        """
        finds overlapping extent of the input files (Geotiffs).  Adds clipped
        matrices of input files as attributes, and additionally outputs geotiffs
        of each (both snow dates, DEM, and Vegetation).

        Arguments:
            remove_clipped_files:   UserConfig option to delete clipped file
        """

        # 1) A few tasks
        # a. Save original metadata to json formated txt file
        #     date1 or date2 should both work?  Vestige from when date mattered..
        # b. get information on extents and resolution
        # Zach check warning from log here WARNING %s in %s
        with rio.open(self.file_path_dataset2) as src:
            meta2 = src.profile

        with open(self.file_path_out_json, 'w') as outfile:
            json_dict = dict({k:v for k, v in meta2.items() if k != 'crs'})
            # meta2 is a dictionary with metadata, however the 'crs' key
            # yields an object: CRS.from_epsg(<epsg#>).  Below function
            # rio.crs.CRS.to_epsg converts crs object to epsg int i.e.= 32611
            # Required because crs object is not "serializable" for json.dump.
            # obj will be reincarnated with epsg code later to save with Rasterio
            crs_object = rio.crs.CRS.to_epsg(meta2['crs'])
            json_dict.update({'crs' : crs_object})
            json.dump(json_dict, outfile)

        # topo_rez_same: topo spatial resolution same as snow files,
        # extents_same:   if snow files have matching spatial resolutions,
        # min_extents: minimum overlapping extents
        # rez13: resolution of all three
        topo_rez_same, extents_same, min_extents, rez13 = prep_coords( \
                self.file_path_dataset1, self.file_path_dataset2, \
                self.file_path_topo, 'dem')

        # 2) Create strings to prepare files using OS.run

        # list of clipped files that are created in below code
        # to be deleted upon UserConfig preference
        self.remove_clipped_files = remove_clipped_files
        self.new_file_list = [self.file_path_date1_te, self.file_path_date2_te, \
                                self.file_path_dem_te, self.file_path_veg_te]

        # prepare log messages
        log_msg1 = '\nThe resolution of your topo.nc file differs from repeat arrays' \
            '\nIt will be resized to fit the resolution repeat arrays.' \
            '\nTopo spatial resolution = {0}    &   repeat arrays resolution = {1}' \
            '\n Your input files will NOT be changed{2}\n' \
                                                .format(rez13[2], rez13[0], '--'*60)
        log_msg2 = '\nNote: the extents of date1 and date2 were the same and' \
                    '\nDid NOT need clipping.  File copied and renamed as clipped' \
                    '\nbut is the same\n'

        # prepare topo substrings for topo NetCDF
        # same target rez (flag -tr in gdalwarp) so no resample needed
        if topo_rez_same:
            tr_substring1 = '-of GTiff NETCDF:"{0}":dem {1}'.format \
                        (self.file_path_topo, self.file_path_dem_te)

            tr_substring1b = '-of GTiff NETCDF:"{0}":veg_height {1}' \
                        .format(self.file_path_topo, self.file_path_veg_te)


        # If spatial resolution of topo differs from snow files
        # resample veg and dem from topo.nc to match snow rez,
        else:
            self.log.info(log_msg1)

            tr_substring1 = '-tr {0} {0} -of GTiff NETCDF:"{1}":dem {2}' \
                        .format(round(rez13[0]), self.file_path_topo, \
                        self.file_path_dem_te)

            tr_substring1b ='-tr {0} {0} -of GTiff NETCDF:"{1}":veg_height {2}' \
                        .format(round(rez13[0]), self.file_path_topo, \
                        self.file_path_veg_te)

        # START Clipping
        if not extents_same:
            # if date1, date2 and topo have different extents  ---> clip
            run_arg1 = 'gdalwarp -te {0} {1} {2} {3} {4} {5} -overwrite'.format \
                        (*min_extents, self.file_path_dataset1, \
                        self.file_path_date1_te)

            run_arg2 = 'gdalwarp -te {0} {1} {2} {3} {4} {5} -overwrite'.format \
                        (*min_extents, self.file_path_dataset2, \
                        self.file_path_date2_te)

            run_arg3 = 'gdalwarp -te {0} {1} {2} {3} {4} -overwrite'.format \
                        (*min_extents, tr_substring1)

            run_arg3b = 'gdalwarp -te {0} {1} {2} {3} {4} -overwrite'.format( \
                        *min_extents, tr_substring1b)

        else:
            # if snow files and topo.nc are the same extents,
            # just copy snow and translate topo. no need to clip or resample
            self.log.info(log_msg2)

            run_arg1 = 'cp {} {}'.format(self.file_path_dataset1,
                        self.file_path_date1_te)
            run_arg2 = 'cp {} {}'.format(self.file_path_dataset2,
                        self.file_path_date2_te)
            run_arg3 = 'gdal_translate {}'.format(tr_substring1b)
            run_arg3b = 'gdal_translate {}'.format(tr_substring1b)

        # 3) Now run all commands
        run(run_arg1, shell = True)
        run(run_arg2, shell = True)
        run(run_arg3, shell = True)
        run(run_arg3b, shell = True)

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

        self.log.debug('Entered mask_advanced')

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

        self.log.debug('Exited mask_advanced')

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
        with rio.open(self.file_path_date1_te) as src:
            d1_te = src
            mat_clip1 = d1_te.read()  #matrix
            mat_clip1 = mat_clip1[0]
            # If NaN is represented as np.nan, change to -9999. Will be
            # returned to np.nan prior to saving tiff, once analysis is complete
            mat_clip1[np.isnan(mat_clip1)] = -9999
            mat_clip1 = get16bit(mat_clip1)
            self.mat_clip1 = mat_clip1.copy()
        with rio.open(self.file_path_date2_te) as src:
            self.d2_te = src
            self.meta2_te = self.d2_te.profile
            mat_clip2 = self.d2_te.read()  #matrix
            mat_clip2 = mat_clip2[0]
            # If NaN is represented as np.nan, change to -9999. Will be
            # returned to np.nan prior to saving tiff, once analysis is complete
            mat_clip2[np.isnan(mat_clip2)] = -9999
            mat_clip2 = get16bit(mat_clip2)
            self.mat_clip2 = mat_clip2.copy()
        with rio.open(self.file_path_dem_te) as src:
            topo_te = src
            dem_clip = topo_te.read()  #not sure why this pulls the dem when there are logs of
            dem_clip = dem_clip[0]
            # If NaN is represented as np.nan, change to -9999. Will be
            # returned to np.nan prior to saving tiff, once analysis is complete
            dem_clip[np.isnan(dem_clip)] = -9999
            self.dem_clip = dem_clip.copy()
            # ensure user_specified DEM resolution is compatible with uint8 i.e. not too fine
            elevation_band_rez = check_DEM_resolution(self.dem_clip,
                                                self.elevation_band_resolution)
        self.elevation_band_resolution = elevation_band_rez
        self.all_loss = (np.absolute(mat_clip1) > 0) & (np.absolute(mat_clip2) == 0)  #all loss minimal
        mat_diff = mat_clip2 - mat_clip1  # raw difference
        # self.self.all_loss = (np.absolute(self.mat_clip1).round(2) > 0.0) & (np.absolute(self.mat_clip2).round(2)==0.0)  #all loss minimal
        mat_clip1[~self.all_loss & (mat_clip1 < 25)] = 25  # Set snow depths below 0.25m to 0.25m to avoid dividing by zero
        mat_diff_norm = np.round((mat_diff / mat_clip1), 2)  #
        self.mat_diff_norm = np.ndarray.astype(mat_diff_norm, np.float16)
        self.mat_diff = np.ndarray.astype(mat_diff, np.float16)

    def determine_basin_change(self, band):
        """
        First pass (nod to Mark for the term "First Pass" for example "I have a P I,
        here is my first pass at retaining my dignity - I'll shoot 17 consecutive
        shots 2.5 ft from basket until I win") at determining if basin lost or
        gained snow.  This short protocol takes two snow.nc files.

        Returns:
            gaining:       True if basin gained snow.  False if basin lost snow.
                            this will be used later to shed a noisy flag
        """
        # Zach check warning from log here WARNING %s in %s
        file_path_snownc1, file_path_snownc2 = \
                        return_snow_files(self.file_path_snownc, \
                                        self.date1_string, self.date2_string)

        snownc1_file_open_rio = 'netcdf:{0}:{1}'.format(file_path_snownc1, band)
        snownc2_file_open_rio = 'netcdf:{0}:{1}'.format(file_path_snownc2, band)
        topo_file_open_rio = 'netcdf:{0}:{1}'.format(self.file_path_topo, 'mask')

        date1 = file_path_snownc1.split('/')[-1][3:11]
        date2 = file_path_snownc2.split('/')[-1][3:11]
        file_path_out = os.path.join(self.file_name_base, \
                    'RAQC_modelled_basin_diff_{}_to_{}.png'.format(date1, date2))

        with rio.open(snownc1_file_open_rio) as src:
            snownc1_obj = src
            meta = snownc1_obj.profile
            snownc1 = snownc1_obj.read()

        with rio.open(snownc2_file_open_rio) as src:
            snownc2_obj = src
            snownc2 = snownc2_obj.read()

        with rio.open(topo_file_open_rio) as src:
            mask_obj = src
            mask = mask_obj.read()

        # Get statistics on snow depth change
        # snownc1 and snownc2 are now the arrays of specified band (thickness)
        snownc1 = snownc1[0]
        snownc2 = snownc2[0]
        # basin mask array
        mask = mask[0]
        mask = np.ndarray.astype(mask, np.bool)
        # only interested in pixels with snow on at least one day
        both_zeros = (np.absolute(snownc1) ==0) & (np.absolute(snownc2) == 0)
        zeros_and_mask = mask & ~both_zeros
        # snow property change (thickness)
        diff = snownc2 - snownc1
        # only within mask where snow present
        diff_clipped_to_mask = diff[zeros_and_mask]

        rez = meta['transform'][0]
        # basin-wide change
        basin_total_change = np.sum(diff_clipped_to_mask) * (rez ** 2)
        # number of cells - to get area, multply by cell size
        total_pixels_in_mask = diff_clipped_to_mask.shape[0]
        # avg change per pixel
        basin_avg_change = round((basin_total_change / total_pixels_in_mask) / (rez ** 2), 2)

        # determine if basin-wide snow depth is gaining or losing
        if basin_total_change > 0:
            gaining = True
        else:
            gaining = False

        cbar_string = '\u0394 thickness (m)'
        suptitle_string = '\u0394 snow thickness (m): snow.nc run{}_to_{}'. \
                            format(date1, date2)

        # temp dates to pass into log message
        temp_date1 = ''.join(file_path_snownc1.split('/')[-2])
        temp_date2 = ''.join(file_path_snownc2.split('/')[-2])

        # turn True or False into string = 'gaining' or 'losing' for log message
        gain_loss = 'gaining' * gaining + 'losing' * (not gaining)
        log_message = '\nTotal basin_difference in depth ("thickness")'\
                        '\ncalculated between'\
                        '\n<file_path_snownc>/{0}/snow.nc and '\
                        '\n<file_path_snownc>/{1}/snow.nc is {2}m.' \
                        '\nThe average change in pixels where depth changed, ' \
                        '\ni.e. where snow was present in either of the two dates,' \
                        '\nwas {3} m.' \
                        '\nAs such the basin is considered to be "{4}.' \
                        '\nFlags will be determined accordingly\n'.format \
                        (temp_date1, temp_date2, str(int(round(basin_total_change, 0))), \
                        str(round(basin_avg_change,2)), gain_loss)
        self.log.info(log_message)

        # gaining = False
        self.gaining = gaining

        # # plot and save
        # basic_plot(diff, zeros_and_mask, cbar_string, suptitle_string, file_path_out)

    def get_buffer(self):
        """
        Returns numbers of cells to buffer clipped tiff to match size and
        extents of original file.

        Outputs:
            buffer:     dictionary with buffer values left, right, bottom, top
                        in index positions not UTMs
        """


        # get extents and resolution from json and determine if originally clipped
        # Zach consider using spatialnc.get_topo for saving this data instead
        self.derive_dataset('d2_te')
        #If disjoint bounds  --> find num cols and rows to buffer with nans N S E W
        #   Note: self.extents_same comes from derive_dataset
        if not self.extents_same:
            # clipped bounds
            bounds_date2_te = [None] * 4
            bounds_date2_te[0], bounds_date2_te[1] = self.d2_te.bounds.left, self.d2_te.bounds.bottom
            bounds_date2_te[2], bounds_date2_te[3] = self.d2_te.bounds.right, self.d2_te.bounds.top

            rez = self.orig_extents_rez['resolution']
            bounds_date2 = [None] * 4
            # Round bounds (extents) to even numbers in multiples of rounded rez
            #   For instance, bounds 2049m and 2024m with rez = 50m
            #   convert to 2050m and 2000m respectively
            bounds_date2[0] = evenly_divisible_extents \
                                        (self.orig_extents_rez['left'], rez)
            bounds_date2[2] = evenly_divisible_extents \
                                        (self.orig_extents_rez['right'], rez)
            bounds_date2[1] = evenly_divisible_extents \
                                        (self.orig_extents_rez['bottom'], rez)
            bounds_date2[3] = evenly_divisible_extents \
                                        (self.orig_extents_rez['top'], rez)

            buffer={}
            # Notice '*-1'.  Those ensure subsetting flag array into nan array
            # buffers -<buffer> on right and bottom, and +<buffer> top and left
            buffer.update({'left' : round((bounds_date2[0] - bounds_date2_te[0]) / rez) * -1})
            buffer.update({'bottom' : round((bounds_date2[1] - bounds_date2_te[1]) / rez)})
            buffer.update({'right' : round((bounds_date2[2] - bounds_date2_te[2]) / rez) * -1})
            buffer.update({'top' : round((bounds_date2[3] - bounds_date2_te[3]) / rez)})

            # Replace zeros with None if they exist
            for k, v in buffer.items():
                if v == 0:
                    buffer[k] = None
            self.buffer = buffer
    def save_tiff(self, flag_attribute_names, include_arrays, \
                    resampling_percentiles):
        """
        Saves up to two geotiffs using RasterIO basically.  One tiff will be the
        matrices of floats, and the second the masks and flags - booleans (uint8).
        Bands to output will be specified in UserConfig

        Args:
            flags: string. flags to include as bands in tif
            include_arrays: arrays (i.e. difference) to include as bands in tif
        Outputs:
            file_path_out_tiff_flags:   single or multibanded array which may
                                        include any flag or mask
            file_path_out_tiff_arrays:  single or multibanded array which may
                                        include mat_diff and snow depth matrices
                                        for example
        """

        tick = time.clock()

        # 1) SET dtypes and NaN representation in prepartion for saving to tiff
        # Save all arrays as float32
        # convert to float from int to set NaNs back
        self.mat_clip1 = np.ndarray.astype(self.mat_clip1, np.float32)
        self.mat_clip2 = np.ndarray.astype(self.mat_clip2, np.float32)
        self.median_elevation = np.ndarray.astype(self.median_elevation, np.float32)

        # now that flags and masks are calculated, set -9999 back to np.nan
        self.mat_diff[~self.mask_nan_snow_present] = np.nan
        self.mat_diff_norm[~self.mask_nan_snow_present] = np.nan
        self.mat_clip1[self.mat_clip1 == -9999] = np.nan
        self.mat_clip2[self.mat_clip2 == -9999] = np.nan
        self.median_elevation[~self.mask_nan_snow_present] = np.nan

        # finally, change abbreviated attribute names to intuitive band names
        band_names = apply_dict(flag_attribute_names, self.keys_master, 'mat_object_to_tif')

        # 4) RESTORE clipped arrays/flags to original size, extent etc:

        # if input tif files were less than desired 50m output
        meta2_clip = getattr(self, 'meta2_te')
        rez_clip = meta2_clip['transform'][0]
        if round(rez_clip) != 50:
            self.log.debug('rez not 50m')
            flag_dtype = 'float32'
            fill_val = np.nan
            not_50m = True
        else:
            flag_dtype = 'uint8'
            fill_val = 255
            not_50m = False

        # First determine if buffering necessary
        self.get_buffer()
        if not self.extents_same:
            # find number of rows and columns to buffer with NaNs
            buffer = self.buffer
            # buffer flag arrays with nans to fit original date2 array shape
            # nan = <uint> 255
            for id, band in enumerate(flag_attribute_names):
                flag_buffer = np.full(self.orig_shape, fill_val, dtype = flag_dtype)
                mat_temp = getattr(self, band)
                if not_50m:
                    mat_temp = mat_temp.astype(flag_dtype)
                flag_buffer[buffer['top'] : buffer['bottom'], buffer['left'] : buffer['right']] = mat_temp
                setattr(self, band, flag_buffer)

            # Open JSON with original metadata and restore some key value pairs
            # that required changes when saving to json txt file
            meta_orig = update_meta_from_json(self.file_path_out_json)

            # update clipped metadata with that of original - extents, resolutition, etc.
            self.meta2_te.update(meta_orig)

        # upate metadata to include number of bands (flags) and uint8 dtype
        self.meta2_te.update({
            'count': len(flag_attribute_names),
            'dtype': flag_dtype,
            'nodata': fill_val})

        # 5) WRITE Flags.tif
        with rio.open(self.file_path_out_tif_flags, 'w', **self.meta2_te) as dst:
            for id, band in enumerate(flag_attribute_names, start = 1):
                try:
                    mat_temp = getattr(self, flag_attribute_names[id - 1])
                    dst.write_band(id, mat_temp.astype(flag_dtype))
                    dst.set_band_description(id, band_names[id - 1])
                except ValueError:  # Rasterio has no float16  >> cast to float32
                    self.log.debug('Try except in save_tiff: if triggered, investigate')
                    mat_temp = getattr(self, flag_attribute_names[id - 1])
                    dst.write_band(id, mat_temp.astype(flag_dtype))
                    dst.set_band_description(id, band_names[id - 1])

        # 5a) if images need rescaling to 50m
        # reference readme.md for details on 50m flag tif
        if not_50m:
            self.rescale_to_50m(resampling_percentiles)

        # REPEAT steps 3) and 4) for saving arrays i.e. diff and diff_norm

        # 4) RESTORE clipped arrays/flags to original size, extent etc:
        # First determine number of rows and columns to buffer with NaNs
        if include_arrays != None:
            array_names = []
            for array in include_arrays:
                array_names.append(self.keys_master['config_to_mat_object'][array])  # 1)Change here and @2 if desire to save single band

            # First determine if buffering necessary
            if not self.extents_same:
                # find number of rows and columns to buffer with NaNs
                buffer = self.buffer
                # buffer arrays with nans to fit original date2 array shape
                # nan = <uint> 255
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

            # 5 Write Arrays.tif
            with rio.open(self.file_path_out_tif_arrays, 'w', **self.meta2_te) as dst:
                for id, band in enumerate(array_names, start = 1):
                    mat_temp = getattr(self, array_names[id - 1])
                    dst.write_band(id, mat_temp.astype('float32'))
                    dst.set_band_description(id, band_names[id - 1])

        # 7) CLEANUP. Now all is complete, DELETE clipped files from
        # clip_extent_overlap() upon user specification in UserConfig,
        try:
            if self.remove_clipped_files == True:
                for file in self.new_file_list:
                    run('rm ' + file, shell = True)
        except AttributeError:  #occurs if user passed clipped files through config
            pass

        tock = time.clock()
        log_message = 'save tiff = {} seconds'.format(round(tock - tick, 2))
        self.log.debug(log_message)

    def rescale_to_50m(self, resampling_percentiles):
        """
        Final step in resampling finer spatial resolution flags to 50m.
        Takes 50m flag.tif and saves multi-banded tif based off thresholds.
        If [0.1 0.2 0.3] or 10, 20 and 30% were passed to function, then output
        tif will have 3 bands for each of those percentages.
        For example a pct = 0.1 (10%) will flag 50m pixels that contained >=10%
        of the original finer resolution flag tif from which it was resampled.
        Simple theoretical example, if resampling a 5m flag.tif to a 50m:
        The 50m contains 100 5m pixels (10X10 grid).  For a given 50m pixel, if
        10 composing 50m pixels were flags, then the 50m pixel would be flagged
        at the 10% band but NOT the 20% band.  If only 9 pixels out of 100 were
        flagged, then the flag would be false for that 50m pixel.
        """

        # Create file paths
        fp_temp_base = os.path.splitext(self.file_path_out_tif_flags)[-2]
        fp_percent = '{}_resampled_percent.tif'.format(fp_temp_base)
        fp_thresh = '{}_resampled_thresh_flags.tif'.format(fp_temp_base)

        # resample from finer resolution to 50m
        cmd = 'gdalwarp -r average -overwrite -tr 50 50 {0} {1}' \
                .format(self.file_path_out_tif_flags, fp_percent)
        # run gdal command through shell
        run(cmd, shell = True)

        # Note flag is dtype float32 since it's a percentage
        with rio.open(fp_percent) as src:
            rio_obj = src
            meta = rio_obj.profile
            arr = rio_obj.read()[:]
            # rio.obj.descriptions are from set_band_descriptions method of
            # rasterio. They are hacky substitutes for variable names in netCDFs
            # BUT can be viewed in QGIS and potentially other programs
            band_desc = list(rio_obj.descriptions)

        # stack thresholded bands into multilayered np array (temp_stack)
        for pct in resampling_percentiles:
            # sums all flags (elevation gain/loss, basin gain/loss, etc)
            # into one array with total number of flags at each pixel location
            temp_arr = np.sum((arr > pct) * 1, axis = 0)
            # adds one more dimension to array to become (1,n,m) where n = rows
            # and m = cols
            temp_arr = temp_arr[np.newaxis]
            # stack each pct thresh band for each iter of loop
            if 'temp_stack' not in locals():
                temp_stack = np.ndarray.astype(temp_arr, 'uint8')
            else:
                temp_stack = np.concatenate \
                    ((temp_stack, np.ndarray.astype(temp_arr, 'uint8')), axis = 0)

        # create strings for band descriptions
        pct_str = ['{}% thresh'.format(int(pct * 100)) \
                                    for pct in resampling_percentiles]

        # update metada
        meta.update({'count':len(resampling_percentiles),
                    'dtype':'uint8',
                    'nodata':255})

        # write array to tif
        with rio.open(fp_thresh, 'w', **meta) as dst:
            for i in range(temp_stack.shape[0]):
                dst.set_band_description(i + 1, '{}'.format(pct_str[i]))
            dst.write(temp_stack)

        # with rio.open(fp_percent, 'w', **meta) as dst:
        #     for i in range(temp_stack.shape[0]):
        #         dst.set_band_description(i + 1, '{}'.format(band_desc[i]))
        #     dst.write(temp_stack)


    def format_flag_names(self, flags, prepend):
        """
        Maps flags from UserConfig names to attribute names and vice versa.
        Also, expands UserConfig flag name 'basin' and 'elevation' into
        loss and gain flags for both
        - i.e. dict['basin'] = 'flag_basin_loss' and 'flag_basin_gain'

        Args:
            flags:      string list of UserConfig or attribute flag names
            prepend:    <boolean> True to prepend flags with 'flag_'
                        i.e. dict['histogram'] = 'flag_histogram'.
                        False will map the reverse, from 'flag_<name>' to 'flag'
        Returns:
            flag_names:  list of mapped flag names
        """

        flag_dict = {'prepend_flag':{'histogram':'flag_histogram',
                                'zero_and_nan':'flag_zero_and_nan',
                                'basin':['flag_basin_gain', 'flag_basin_loss'],
                                'elevation':['flag_elevation_gain',
                                            'flag_elevation_loss']},
                    'remove_flag':{'flag_histogram':'histogram',
                                    'flag_zero_and_nan':'zero_and_nan',
                                    'flag_basin_gain':'basin_gain',
                                    'flag_basin_loss':'basin_loss',
                                    'flag_elevation_gain':'elevation_gain',
                                    'flag_elevation_loss':'elevation_loss'}}
        if prepend is True:
            action_key = 'prepend_flag'
        else:
            action_key = 'remove_flag'

        flag_names = []
        for flag_name in flags:
            temp_flag = flag_dict[action_key][flag_name]
            if type(temp_flag) is list:
                [flag_names.append(name) for name in temp_flag]
            else:
                flag_names.append(temp_flag)

        return(flag_names)

    def plot_hist(self, action):
        if action != None:
            pltz_obj = pltz.Plotables()
            pltz_obj.set_zero_colors(1)
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
                plt.savefig(self.file_path_out_histogram, dpi=180)
        else:
            pass

    def derive_dataset(self, dataset_clipped_name):
        """
        Unpacks, formats original metadata from json file and returns
        Also determines self.extents_same
        Zach consider outputting orig_extents_rez

        Inputs:
            dataset_clipped_name:   rio object name
        Outputs (i.e. self.):
            orig_shape:         dimensions of original date2 input tif
            extents_same:       boolean - was original file clipped
        Returns:
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
        # Zach  do we really need to save self.d2_te or can we just save a text file?
        meta_clip = getattr(self, dataset_clipped_name)

        # determine if original extents were same
        self.extents_same = (meta_clip.bounds.left == left) & \
                            (meta_clip.bounds.right == right) & \
                            (meta_clip.bounds.top == top) & \
                            (meta_clip.bounds.bottom == bottom)

        # dictionary of extents of original file
        orig_extents_rez.update({'left' : left, 'right' : right, 'top' : top, \
                        'bottom' : bottom, 'resolution' : rez})

        # save size of original array as tuple to get shape
        # Zach consider adding orig shape to json or as a spatialnc.get_topo_stats
        orig_shape = []
        orig_shape.extend([round((top - bottom)/rez), round((right - left)/rez)])
        self.orig_shape = tuple(orig_shape)
        self.orig_extents_rez = orig_extents_rez

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

        self.log.debug('entered histogram moving window')
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
        log_message = 'mov_wind_zach_version = {} seconds'.format \
                                            (round(tock-tick, 2))
        self.log.debug(log_message)
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

        self.log.debug('entering map space moving window')
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
    # def init(self, file_path_dataset1, file_path_dataset2, file_path_topo,
    #         file_out_root, basin, file_name_modifier, elevation_band_resolution,
    #         file_path_snownc):
    def init(self):
        """
        Protozoic attempt of use of inheritance
        """
        # MultiArrayOverlap.init(self, file_path_dataset1, file_path_dataset2,
        #                         file_path_topo, file_out_root, basin,
        #                         file_name_modifier, file_path_snownc)

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
        self.log.debug('entering make_histogram')

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
        log_message = 'hist2D_with_bins_mapped = {} seconds'.format \
                                        (round(tock - tick, 2))
        self.log.debug(log_message)

    # @profile
    def flag_basin(self, apply_moving_window, moving_window_size, \
                            neighbor_threshold, snowline_threshold,
                            gaining):
        """
        Finds cells of complete melt or snow where none existed prior.
        Apply moving window to remove scattered and isolated cells, ineffect
        highlighting larger blocks of cells.  Intended to diagnose extreme, unrealistic
        change that were potentially processed incorrectly by ASO.

        Args:
            apply_moving_window:  Boolean.  Moving window is optional.
            moving_window_size:   size of moving window used to define blocks
            neighbor_threshold:   proportion of neighbors within moving window (including target cell) that have
            snowline_threshold:   mean depth of snow (cm) in elevation band used to determine snowline
            gaining:              True if basin gaining snow, False otherwise

        Outputs:
            flag_basin_gain:      all_gain blocks
            flag_basin_loss:      all_loss_blocks
            veg_present:          from topo.nc vegetation band.
                                    where veg height > 5m

        """
        # Note below ensures -0.0 and 0 and 0.0 are all discovered and flagged as zeros.
        # Checked variations of '==0.0' and found that could have more decimals or be ==0 with same resultant boolean
        # all_gain = (np.absolute(self.mat_clip1).round(2)==0.0) & (np.absolute(self.mat_clip2).round(2) > 0.0)
        # self.all_gain = (np.absolute(self.mat_clip1) == 0) & (np.absolute(self.mat_clip2) > 0)

        if hasattr(self, 'snowline_elev'):
            pass
        else:
            self.snowline_elev = snowline(self.dem_clip, self.mask_overlap_nan,
            self.elevation_band_resolution, self.mat_clip1, self.mat_clip2,
            snowline_threshold)

        # Create flag_basin_gain or flag_basin_loss depending on if basin is
        # gaining or losing overall.

        # If/else statement creates initial boolean array for flag and strings
        # to further refine flag and set to attributes depending on basin change.
        if gaining:
            # no nans either date (mask_overlap_nan)
            # and where snow completely melted (all_loss)
            # don't include loss below snowline
            basin_flag = self.mask_overlap_nan & self.all_loss & \
                                    (self.dem_clip > self.snowline_elev)
            basin_string = 'loss'
            attribute_string = 'all_loss'
            flag_attribute_string = 'flag_basin_loss'
        else:
            # no nans either date (mask_overlap_nan)
            # and where snow was begat from bare ground (all_gain)
            basin_flag = self.mask_overlap_nan & self.all_gain
            basin_string = 'gain'
            attribute_string = 'all_gain'
            flag_attribute_string = 'flag_basin_gain'

        if apply_moving_window:
            # moving window removes stray and isolated flagged pixels
            pct = self.mov_wind2(basin_flag, moving_window_size)
            basin_flag = (pct > neighbor_threshold) & \
                                    getattr(self, attribute_string)
        else:
            # set flag as attribute
            setattr(self, flag_attribute_string, basin_flag)

        log_msg1 = '\nThe snowline was determined to be at {0}m.' \
                    '\nIt was defined as the first elevation band in the basin' \
                    '\nwith a mean snow depth >= {1} cm.' \
                    ' \nElevation bands were in {2}m increments\n'. \
                    format(self.snowline_elev, snowline_threshold, \
                            self.elevation_band_resolution)
        self.log.info(log_msg1)

    # @profile
    def flag_elevation(self, apply_moving_window, moving_window_size,
            neighbor_threshold, snowline_threshold, outlier_percentiles,
            elev_flag_only_veg):
        """
        More potential for precision than flag_basin function.  Finds outliers
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
            moving_window_size:  Same as flag_basin
            neighbor_threshold:  Same as flag_basin
            snowline_threshold:  Same as flag_basin
            outlier_percentiles:  list of four values (raw gain upper, raw loss upper, normalized gain lower, normalized loss lower)
                                    Percentiles used to threshold each elevation band.  i.e. 95 in (95,80,10,10) is the raw gain upper,
                                    which means anything greater than the 95th percentile of raw snow gain in each elevatin band will
                                    be flagged as an outlier.
            elev_flag_only_veg:   True means only pixels with veg_height > 5m will be flagged

        Output:
            flag_elevation_loss:  attribute
            flag_elevation_gain   attribute
        """
        self.log.debug('entering flag_elevation')
        tick = time.clock()

        # 1) Grab necessary matrices

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
            self.snowline_elev = snowline(snowline_threshold)

        # Read docstring for this function for detailed info.
        # In short: map_id_dem.shape = mask.shape, the shape of the clipped array.
        #                              it is an array that consists of indices
        #                               corresponding to elevation bands
        map_id_dem, id_dem_unique, elevation_edges = \
            get_elevation_bins(self.dem_clip, mask, \
                                self.elevation_band_resolution)

        map_id_dem_overlap = map_id_dem[mask]

        # 2) Find elevation band thresholds
        # Find threshold values per elevation band - 1D array
            # a) upper, lower and median outlier thresholds for snow depth
            # b) upper and lower outlier thresholds for normalized snow depth
            # c) elevation_count = bins per elevation band
        # Initiate numpy 1D arrays for these elevation bin statistics:
        thresh_upper_norm = np.zeros(id_dem_unique.shape, dtype = np.float16)
        thresh_lower_norm = np.zeros(id_dem_unique.shape, dtype = np.float16)
        thresh_upper_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)
        thresh_lower_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)
        elevation_count = np.zeros(id_dem_unique.shape, dtype = np.int64)
        median_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)
        temp = np.ones(mat_diff_norm_masked.shape, dtype = bool)
        # save bin statistics per elevation band to a numpy 1D Array i.e. list
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            # for each elevation band find outliers in snow depth distribution:
            # ex) at 1850-1900m elev band, find 5th percentile (thresh_lower) and
            # 95th (thresh_upper) of snow depth in m (raw) and normalized (norm)
            # These arrays will be N X 1 where N = number of elevation bins
            thresh_upper_raw[id] = np.percentile(mat_diff_masked[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[0])
            thresh_upper_norm[id] = np.percentile(mat_diff_norm_masked[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[1])
            thresh_lower_raw[id] = np.percentile(mat_diff_masked[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[2])
            thresh_lower_norm[id] = np.percentile(mat_diff_norm_masked[map_id_dem_overlap == id_dem_unique2], outlier_percentiles[3])
            median_raw[id] = np.percentile(mat_diff_masked[map_id_dem_overlap == id_dem_unique2], 50)
            elevation_count[id] = getattr(temp[map_id_dem_overlap == id_dem_unique2], 'sum')()

        # 3) Place elevatin bin indices on clipped map space
        # Place threshold values onto map in appropriate elevation bin
        # The result of this codeblock ending after try/except statement will
        # be arrays corresponding to the DEM with identical values in pixels
        # within the same elevation band.  For instance the thresh_upper_raw_array
        # may have 2.1 for all pixels within the 2150 - 2200m elevation bins.
        # These arrays will not be output or saved but used to find elevation
        # based outliers like this:
        # temp = mat_diff_norm > thresh_upper_norm_array
        # temp2 = mat_diff > thresh_upper_raw_array
        # flag_elevation_gain = temp & temp2
        # Note that flag_elevation_gain and flag_elevation_loss are pixels
        # where both raw snow depth and normalized depth are above or below
        # for both gain and loss
        thresh_upper_norm_array = np.zeros(mask.shape, dtype=np.float16)
        thresh_lower_norm_array = thresh_upper_norm_array.copy()
        thresh_upper_raw_array = thresh_upper_norm_array.copy()
        thresh_lower_raw_array = thresh_upper_norm_array.copy()
        thresh_median_raw_array = thresh_upper_norm_array.copy()
        for id, id_dem_unique2 in enumerate(id_dem_unique):
            # id_bin = idx of locations on map matching elevation band
            # thresh_upper_norm[id] grabs the upper normalized threshold for
            # the particular elevation band for instance
            id_bin = map_id_dem == id_dem_unique2
            try:
                thresh_upper_norm_array[id_bin] = thresh_upper_norm[id]
                thresh_upper_raw_array[id_bin] = thresh_upper_raw[id]
                thresh_lower_norm_array[id_bin] = thresh_lower_norm[id]
                thresh_lower_raw_array[id_bin] = thresh_lower_raw[id]
                thresh_median_raw_array[id_bin] = median_raw[id]
            except IndexError as e:
                self.log.debug(e)
        # save as attribute for plotting purposes
        self.median_elevation = thresh_median_raw_array

        # 4) Find flags and set attributes.
        #    Use thresh_<upper or lower>_<norm or raw>_array to find outliers
        #    in map space and save flags as attributes

        # Dictionary to translate values from UserConfig
        keys_local = {'loss' : {'operator' : 'less',
                                'flag' : 'flag_elevation_loss',
                                'mat_diff_norm' : thresh_lower_norm_array,
                                'mat_diff' : thresh_lower_raw_array},
                    'gain' : {'operator' : 'greater',
                                'flag' : 'flag_elevation_gain',
                                'mat_diff_norm' : thresh_upper_norm_array,
                                'mat_diff' : thresh_upper_raw_array}}

        basin_change = ['loss', 'gain']
        shape_temp = getattr(np, 'shape')(getattr(self, 'mat_diff'))
        diff_mats = [['mat_diff', 'mat_diff_norm'], ['mat_diff', 'mat_diff_norm']]
        # Loop to set flag_elevation_LOSS and then flag_elevation_GAIN attributes
        for id, basin_change in enumerate(basin_change):
            # initiate ones array to be used for conditional and/or overlapping
            # of diff and diff_norm outliers
            temp_out_init = np.ones(shape = shape_temp, dtype = bool)
            # loop through raw depth (mat_diff) and normalized (mat_diff_norm)
            # temp_out_init = outliers raw AND outliers normalized
            for diff_mat_name in diff_mats[id]:
                # yields diff_mat or diff_mat_norm array
                diff_mat = getattr(self, diff_mat_name)
                 # yields thresh_..._array
                elevation_thresh_array = keys_local[basin_change][diff_mat_name]
                # finds pixels exceeding elevation band thresholds
                temp_out = getattr(np, keys_local[basin_change]['operator'])(diff_mat, elevation_thresh_array) & temp_out_init
                temp_out_init = temp_out.copy()
            # remove flags below 'snowline'
            temp_out_init[~self.mask_nan_snow_present] = False

            # MOVING WINDOW:
            # If UserConfig specified moving window.  This will potentially
            # remove spatially isolated outliers from flag
            # Finds pixels idenfied as outliers (temp_out_init) which have a
            # minimum number of neighbor outliers within moving window
            # Note: the '& temp_out_init' ensures that ONLY pixels originally
            # classified as outliers are kept
            flag_name = keys_local[basin_change]['flag']
            if apply_moving_window:
                pct = self.mov_wind2(temp_out_init, moving_window_size)
                temp_out_init = (pct > neighbor_threshold) & temp_out_init

            # if flag_elevation_loss, filter out flags below snowline
            if flag_name == 'flag_elevation_loss':
                temp_out_init[self.dem_clip < self.snowline_elev] = False
            # set flag_elevation_loss and gain arrays as attributes

            # if True, only flag pixels where veg > 5m is present
            if elev_flag_only_veg:
                # Open veg layer from topo.nc and identify pixels with veg present (veg_height > 5)
                with rio.open(self.file_path_veg_te) as src:
                    topo_te = src
                    veg_height_clip = topo_te.read()  #not sure why this pulls the dem when there are logs of
                    veg_height_clip = veg_height_clip[0]
                    veg_present = veg_height_clip > 5

                temp_out_init = temp_out_init & veg_present
            setattr(self, flag_name, temp_out_init.copy())

        # 5) Create dataframe to save as CSV
        # Save dataframe of elevation band satistics on thresholds
        # Simply preparing the column names:
        column_names = ['elevation', '{}% change (m)', '{}% change (norm)',
                        '{}% change (m)', '{}% change (norm)', '50% change (m)',
                         'elevation_count']
        column_names_temp = []
        ct = 0
        for names in column_names:
            # if percentile (non-hardcoded), then format col name
            if '{}' in names:
                names = names.format(str(outlier_percentiles[ct]))
                ct += 1
            column_names_temp.append(names)

        # arrange data in numpy 2D array for saving to dataframe
        temp = np.stack((elevation_edges[id_dem_unique], thresh_upper_raw.ravel(),
                        thresh_upper_norm.ravel(), thresh_lower_raw.ravel(),
                        thresh_lower_norm.ravel(), median_raw.ravel(),
                        elevation_count.ravel()), axis = -1)
        temp = np.around(temp, 2)
        df = pd.DataFrame(temp, columns = column_names_temp)
        df.to_csv(path_or_buf = self.file_path_out_csv, index=False)

        tock = time.clock()
        log_msg1 = 'flag_elevation = {} seconds'.format(round(tock - tick, 2))
        self.log.debug(log_msg1)
    def stats_report(self, flag_attribute_names):
        """
        Quick utility to print out table of stats to shell and save to log.
        The meat of utility is reporting to user the potential change in snow
        depth if flagged pixels are updated.  Potential change is estimated
        based on self.thresh_median which is the median raw snow depth per
        elevation band.

        Args:
            flag_attribute_names:      flag names i.e. flag_elevation_gain
        """

        map_id_dem, id_dem_unique, elevation_edges = \
            get_elevation_bins(self.dem_clip, self.mask_nan_snow_present, \
                                self.elevation_band_resolution)

        mat_diff = self.mat_diff.copy()
        delta, cell_count, pct_coverage = [], [], []
        for flag_name in flag_attribute_names:
            row = []
            flag = getattr(self, flag_name)
            mask_temp = self.mask_nan_snow_present & ~flag
            mat_diff_clip = mat_diff[mask_temp]
            map_id_dem_clip = map_id_dem[mask_temp]

            median_raw = np.zeros(id_dem_unique.shape, dtype = np.int16)

            # save bin statistics per elevation bin to a numpy 1D Array i.e. list
            for id, id_dem_unique2 in enumerate(id_dem_unique):
                median_raw[id] = np.percentile(
                            mat_diff_clip[map_id_dem_clip == id_dem_unique2],
                            self.elevation_band_resolution)

            # Place threshold values onto map in appropriate elevation bin
            # Used to find elevation based outliers
            thresh_median_array = np.zeros(mat_diff.shape, dtype=np.float16)
            for id, id_dem_unique2 in enumerate(id_dem_unique):
                id_bin = map_id_dem == id_dem_unique2
                try:
                    thresh_median_array[id_bin] = median_raw[id]
                except IndexError as e:
                    self.log.debug(e)

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
        df = pd.DataFrame({'flag' : flag_attribute_names, 'cell count' : cell_count, '*percent coverage' : pct_coverage, '**\u0394' : delta})
        df = df[['flag', 'cell count', '*percent coverage', '**\u0394']]
        table = tabulate(df, headers='keys', colalign = ["left", "left", "right", "left", "right"], tablefmt = "github")
        table_footer = '\n\n*percent coverage = percent of flagged cells relative to total cells with snow present' \
                        '\n**\u0394 = the % change in basin total snow depth if flagged cells' \
                        '\n\twere instead replaced with the median depth at its elevation band\n'
        log_msg1 = '\n{0}{1}'.format(table, table_footer)
        self.log.info(log_msg1)
