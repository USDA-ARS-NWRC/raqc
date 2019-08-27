from raqc import multi_array
from inicheck.tools import get_user_config
from inicheck.tools import MasterConfig
# from snowav.utils.MidpointNormalize import MidpointNormalize
import rasterio as rios
import sys, os
from subprocess import check_output
import argparse


def main():

    parser = argparse.ArgumentParser(description = "running raqc through the command line")
    parser.add_argument('--user-config', type=str, help = 'user configuration file')
    args = parser.parse_args()
    if os.path.exists(args.user_config):
        configFile = args.user_config

    mcfgFile = MasterConfig(modules = 'raqc')
    ucfg = get_user_config(configFile, mcfg = mcfgFile)

    #checking_later allows not to crash with errors.
    cfg = ucfg.cfg

    # this initiates raqc object with file paths
    raqc_obj = multi_array.Flags(cfg['files']['file_path_in_date1'], cfg['files']['file_path_in_date2'],
                cfg['files']['file_path_topo'], cfg['files']['file_path_out'], cfg['files']['file_name_modifier'])
    # if files passed are already clipped to each other, then no need to repeat
    if not raqc_obj.already_clipped:
        raqc_obj.clip_extent_overlap()

    raqc_obj.make_diff_mat()
    name = cfg['difference_arrays']['name']
    action = cfg['difference_arrays']['action']
    operator = cfg['difference_arrays']['operator']
    val = cfg['difference_arrays']['val']
    raqc_obj.mask_advanced(name, action, operator, val)


    # Gather all flags specified in config
    flags = cfg['flags']['flags']
    # if user specified histogram outliers in user config
    if 'hist' in flags:
        histogram_mats = cfg['histogram_outliers']['histogram_mats']
        num_bins = cfg['histogram_outliers']['num_bins']
        # raqc_obj.hist2d_with_bins_mapped(histogram_mats, num_bins)
        threshold_histogram_space = cfg['histogram_outliers']['threshold_histogram_space']
        moving_window_name = cfg['histogram_outliers']['moving_window_name']
        moving_window_size = cfg['histogram_outliers']['moving_window_size']
        raqc_obj.make_hist(histogram_mats, num_bins, threshold_histogram_space, moving_window_size)
        # raqc_obj.outliers_hist(threshold_histogram_space, moving_window_name, moving_window_size)  # INICHECK
    # if user wants to check for blocks
    for flag in ['basin_block', 'elevation_block']:
        if flag in flags:
            block_window_size = cfg['block_behavior']['moving_window_size']
            block_window_threshold = cfg['block_behavior']['neighbor_threshold']
            snowline_thresh = cfg['block_behavior']['snowline_thresh']
            elevation_band_resolution = cfg['block_behavior']['elevation_band_resolution']
            outlier_percentiles = cfg['block_behavior']['outlier_percentiles']
            if flag == 'basin_block':
                raqc_obj.flag_blocks(block_window_size, block_window_threshold, snowline_thresh, elevation_band_resolution)
            elif flag == 'elevation_block':
                raqc_obj.hypsometry(block_window_size, block_window_threshold, snowline_thresh, elevation_band_resolution, outlier_percentiles)

    raqc_obj.combine_flags(flags)  # makes a combined flags map (which is not output), but also collects list of flag names for later
    file_out = cfg['files']['file_path_out']
    want_plot = cfg['plot']['interactive_plot']
    if want_plot == True:
        raqc_obj.plot_this()
    raqc_obj.save_tiff(file_out)


if __name__=='__main__':
    main()
