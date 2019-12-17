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
    ucfg = get_user_config(configFile, mcfg = mcfgFile, checking_later = False)

    #ensure no errors
    warnings, errors = check_config(ucfg)
    if errors != [] or warnings != []:
        print_config_report(warnings, errors)

    #checking_later allows not to crash with errors.
    cfg = copy.deepcopy(ucfg.cfg)

    raqc_obj = multi_array.Flags(cfg['paths']['file_path_in_date1'],
                cfg['paths']['file_path_in_date2'], cfg['paths']['file_path_topo'],
                cfg['paths']['file_path_out'], cfg['paths']['file_path_snownc'],
                cfg['paths']['basin'], cfg['paths']['file_name_modifier'],
                cfg['thresholding']['elevation_band_resolution'])

    # if files passed are already clipped to each other, then no need to repeat
    if not raqc_obj.already_clipped:
        remove_files = cfg['options']['remove_clipped_files']
        raqc_obj.clip_extent_overlap(remove_files)

    raqc_obj.make_diff_mat()

    raqc_obj.mask_basic()

    raqc_obj.determine_basin_change('thickness')

    # add histogram flag if desired for analysis
    flags = ['basin', 'elevation', 'zero_and_nan']
    if cfg['histogram_outliers']['include_hist_flag']:
        flags.append('histogram')

    # Translate UserConfig flag names into verbose, attribute names
    # i.e. 'basin' becomes 'flag_basin' in this list and self.flag_basin
    # attribute in raqc_object (raqc_obj)
    flag_attribute_names = raqc_obj.format_flag_names(flags, True)

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
                            block_window_threshold, snowline_threshold)


    outlier_percentiles = cfg['thresholding']['outlier_percentiles']
    elev_flag_only_veg = cfg['thresholding']['elev_flag_only_veg']
    raqc_obj.flag_elevation(apply_moving_window,
                            block_window_size, block_window_threshold,
                            snowline_threshold, outlier_percentiles,
                            elev_flag_only_veg)

    file_out = cfg['paths']['file_path_out']
    include_arrays = cfg['options']['include_arrays']

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

    # remove noisy flags based on total basin gain or loss determination
    if raqc_obj.gaining:
        flag_attribute_names.remove('flag_basin_gain')
    else:
        flag_attribute_names.remove('flag_basin_loss')

    raqc_obj.stats_report(flag_attribute_names)

    # Almost done! Save flags and arrays to Tif
    # First add back the zero_and_nan flag
    if replace_zero_nan:
        flag_attribute_names.append('flag_zero_and_nan')

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
