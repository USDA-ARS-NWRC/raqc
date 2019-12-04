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

    # 1) ENSURE CHRONOLOGICAL ORDER
    # Ensure that dataset 1 and dataset2 are in chronological order
    file_path_dataset1 = multi_array.Flags(cfg['paths']['file_path_in_date1']
    file_path_dataset2 = multi_array.Flags(cfg['paths']['file_path_in_date2']
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

    raqc_obj = multi_array.Flags(cfg['paths']['file_path_in_date1'],
                cfg['paths']['file_path_in_date2'], cfg['paths']['file_path_topo'],
                cfg['paths']['file_path_out'], cfg['paths']['basin'],
                cfg['paths']['file_name_modifier'],
                cfg['thresholding']['elevation_band_resolution'],
                cfg['paths']['file_path_snownc'])

    # if files passed are already clipped to each other, then no need to repeat
    if not raqc_obj.already_clipped:
        remove_files = cfg['options']['remove_clipped_files']
        raqc_obj.clip_extent_overlap(remove_files)

    raqc_obj.make_diff_mat()

    # Add check_config
    histogram_mats = cfg['histogram_outliers']['histogram_mats']
    action = cfg['histogram_outliers']['action']
    operator = cfg['histogram_outliers']['operator']
    value = cfg['histogram_outliers']['value']
    raqc_obj.mask_basic()
    raqc_obj.mask_advanced(histogram_mats, action, operator, value)

    # if user specified histogram outliers in user config
    num_bins = cfg['histogram_outliers']['num_bins']
    threshold_histogram_space = cfg['histogram_outliers']['threshold_histogram_space']
    moving_window_name = cfg['histogram_outliers']['moving_window_name']
    moving_window_size = cfg['histogram_outliers']['moving_window_size']
    raqc_obj.make_histogram(histogram_mats, num_bins, threshold_histogram_space, moving_window_size)

    # if user wants to check for blocks
    apply_moving_window = cfg['block_behavior']['apply_moving_window']
    block_window_size = cfg['block_behavior']['moving_window_size']
    block_window_threshold = cfg['block_behavior']['neighbor_threshold']
    snowline_threshold = cfg['thresholding']['snowline_threshold']

    raqc_obj.flag_basin_blocks(apply_moving_window, block_window_size,
                            block_window_threshold, snowline_threshold)

    outlier_percentiles = cfg['thresholding']['outlier_percentiles']

    raqc_obj.flag_elevation_blocks(apply_moving_window,
                            block_window_size, block_window_threshold,
                            snowline_threshold, outlier_percentiles)

    flags = ['histogram', 'basin_gain', 'basin_loss', 'elevation_gain',
            'elevation_loss', 'zero_and_nan']
    raqc_obj.stats_report(flags)
    want_plot = cfg['options']['plot']
    # raqc_obj.plot_this(want_plot)
    raqc_obj.plot_hist(want_plot)

    file_out = cfg['paths']['file_path_out']
    include_arrays = cfg['options']['include_arrays']
    include_masks = cfg['options']['include_masks']

    raqc_obj.save_tiff(file_out, flags, include_arrays, include_masks)

    #backup config file
    config_backup_location = raqc_obj.file_path_out_backup_config
    ucfg.cfg = cfg
    generate_config(ucfg, config_backup_location)

if __name__=='__main__':
    main()
