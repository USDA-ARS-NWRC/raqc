from raqc import multi_array
from inicheck.tools import get_user_config, check_config
from inicheck.config import MasterConfig
from inicheck.output import generate_config, print_config_report
# from snowav.utils.MidpointNormalize import MidpointNormalize
import rasterio as rios
import sys, os
from subprocess import check_output
import argparse
import copy

def main():

    parser = argparse.ArgumentParser(description = "running raqc through the command line")
    parser.add_argument('--user-config', type=str, help = 'user configuration file')
    args = parser.parse_args()
    if os.path.exists(args.user_config):
        configFile = args.user_config

    mcfgFile = MasterConfig(modules = 'raqc')
    ucfg = get_user_config(configFile, mcfg = mcfgFile)

    #ensure no errors
    warnings, errors = check_config(ucfg)
    if errors != [] or warnings != []:
        print_config_report(warnings, errors)

    #checking_later allows not to crash with errors.
    cfg = copy.deepcopy(ucfg.cfg)

    raqc_obj = multi_array.Flags(cfg['paths']['file_path_in_date1'],
                cfg['paths']['file_path_in_date2'], cfg['paths']['file_path_topo'],
                cfg['paths']['file_path_out'],
                cfg['paths']['basin'], cfg['paths']['file_name_modifier'],
                cfg['thresholding']['elevation_band_resolution'])

    # if files passed are already clipped to each other, then no need to repeat
    if not raqc_obj.already_clipped:
        remove_files = cfg['mandatory_options']['remove_clipped_files']
        raqc_obj.clip_extent_overlap(remove_files)

    raqc_obj.make_diff_mat()

    raqc_obj.mask_basic()

    # Not fun late addition to RAQC.  Determine is basin is losing or gaining
    # snow overall.  If losing then don't include losing flag and vice versa
    # for gaining.  They are too noisy in those scenarios.

    # If snownc path is
    # provided, use that.  Otherwise user has specified gaining or losing
    # based off of their own information or time of year

    gaining_determination_method = cfg['mandatory_options']['method_determine_gaining']
    gaining_determination_method = gaining_determination_method.lower()
    fp_snownc = cfg['mandatory_options']['gaining_file_path_snownc']

    # pass gaining_determination_method into function.  If 
    raqc_obj.determine_basin_change(fp_snownc, 'thickness', gaining_determination_method)

    # add histogram flag if desired for analysis
    flags = ['basin', 'elevation', 'zero_and_nan']
    if cfg['histogram_outliers']['include_hist_flag']:
        flags.append('histogram')

    # Translate UserConfig flag names into verbose, attribute names
    # i.e. 'basin' becomes 'flag_basin' in this list and self.flag_basin
    # attribute in raqc_object (raqc_obj)
    flag_attribute_names = raqc_obj.format_flag_names(flags, True)
    print(flag_attribute_names)

    # code block only executed if UserConfig specifies histogram flag
    if cfg['histogram_outliers']['include_hist_flag']:
        histogram_mats = cfg['histogram_outliers']['histogram_mats']
        action = cfg['histogram_outliers']['action']
        operator = cfg['histogram_outliers']['operator']
        value = cfg['histogram_outliers']['value']
        raqc_obj.mask_advanced(histogram_mats, action, operator, value)
        # if user specified histogram outliers in user config
        num_bins = cfg['histogram_outliers']['num_bins']
        threshold_histogram_space = cfg['histogram_outliers']['threshold_histogram_space']
        moving_window_name = cfg['histogram_outliers']['moving_window_name']
        moving_window_size = cfg['histogram_outliers']['moving_window_size']
        want_plot = cfg['histogram_outliers']['plot']
        raqc_obj.make_histogram(histogram_mats, num_bins, threshold_histogram_space, moving_window_size)
        raqc_obj.plot_hist(want_plot)

    # if user wants to check for blocks via 2D moving window
    apply_moving_window = cfg['block_behavior']['apply_moving_window']
    block_window_size = cfg['block_behavior']['moving_window_size']
    block_window_threshold = cfg['block_behavior']['neighbor_threshold']

    # minimum snow depth for determining snowline elevation
    snowline_threshold = cfg['thresholding']['snowline_threshold']


    raqc_obj.flag_basin(apply_moving_window, block_window_size,
                            block_window_threshold, snowline_threshold,
                            raqc_obj.gaining)


    outlier_percentiles = cfg['thresholding']['outlier_percentiles']
    elev_flag_only_veg = cfg['thresholding']['elev_flag_only_veg']
    raqc_obj.flag_elevation(apply_moving_window,
                            block_window_size, block_window_threshold,
                            snowline_threshold, outlier_percentiles,
                            elev_flag_only_veg)

    file_out = cfg['paths']['file_path_out']
    include_arrays = cfg['mandatory_options']['include_arrays']

    # Output statistics table to log
    # first remove unnecessary flags
    # remove flag_zero_and_nan because statistics in table are based on pixels
    # where snow was present
    # Note replace_zero_nan is needed to futureproof changes to flag options
    # by developer.  currently try except unneccessary as flag_zero_nan
    # is not an optional flag
    try:
        flag_attribute_names.remove('flag_zero_and_nan')
        replace_zero_nan = True
    except ValueError:
        replace_zero_nan = False
        pass

    # zero_and_nan is not helpful for comparing snownc to tif file
    is_date1_nc = 'snownc' in raqc_obj.file_path_date2_te
    if is_date1_nc:
        replace_zero_nan = False

    # remove noisy flags based on total basin gain or loss determination

    if raqc_obj.gaining:
        flag_attribute_names.remove('flag_basin_gain')
    else:
        flag_attribute_names.remove('flag_basin_loss')

    raqc_obj.stats_report(flag_attribute_names)

    if cfg['thresholding']['want_thresholds_plot']:
        raqc_obj.thresholds_plot()

    # Almost done! Save flags and arrays to Tif
    # First add back the zero_and_nan flag
    if replace_zero_nan:
        flag_attribute_names.append('flag_zero_and_nan')
    else:
        try:
            flag_attribute_names.remove('flag_zero_and_nan')
        except ValueError:
            pass

    resampling_percentiles = cfg['thresholding']['resampling_percentiles']
    raqc_obj.save_tiff(flag_attribute_names, include_arrays, \
                        resampling_percentiles)

    #backup config file
    config_backup_location = raqc_obj.file_path_out_backup_config
    cfg['thresholding']['elevation_band_resolution'] = \
                                    raqc_obj.elevation_band_resolution
    ucfg.cfg = cfg
    generate_config(ucfg, config_backup_location)

if __name__=='__main__':
    main()
