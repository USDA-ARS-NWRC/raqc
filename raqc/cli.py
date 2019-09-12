from raqc import multi_array
from inicheck.tools import get_user_config, check_config
from inicheck.config import MasterConfig
from inicheck.output import generate_config, print_config_report
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
    ucfg = get_user_config(configFile, mcfg = mcfgFile, checking_later = False)

    #ensure no errors
    warnings, errors = check_config(ucfg)
    if errors != [] or warnings != []:
        print_config_report(warnings,errors)

    #checking_later allows not to crash with errors.
    cfg = ucfg.cfg

    # ENSURE elevation and basin were in flags[flags] if trees was
    # Gather all flags specified in config
    flags = cfg['flags']['flags']
    tree_loss = cfg['flags']['tree_loss']
    tree_gain = cfg['flags']['tree_gain']

    # determine which flags were requested
    basin_present = 'basin_block' in flags
    elevation_present = 'elevation_block' in flags
    both_present = False
    while True:
        if basin_present & elevation_present:
            both_present = True
            break
        else:
            if any([wrd in (tree_loss + tree_gain) for wrd in ['or', 'and']]): #both requested in flags
                both_required = True
                missing = 'add to flags ' + basin_present * 'elevation_block' + elevation_present * 'basin_block'
                break
            else:
                dict_logic = {'basin' : 1, 'elevation' : -1}
                dict_determine_required = {2 : 'basin_required', -2 : 'elevation_required'}
                keys = dict_logic[tree_loss] + dict_logic[tree_gain]
                if keys == 0:  # both requested in flags
                    missing = 'add to flags ' + basin_present * 'elevation_block' + elevation_present * 'basin_block'
                    break
                else:
                    requirement_present = dict_determine_required[keys]
                    missing = '{} is required'.format(requirement_present)
                    break
    print(missing)

    # initiate raqc object with file paths
    raqc_obj = multi_array.Flags(cfg['paths']['file_path_in_date1'], cfg['paths']['file_path_in_date2'],
                cfg['paths']['file_path_topo'], cfg['paths']['file_path_out'], cfg['paths']['basin'], cfg['paths']['file_name_modifier'],
                cfg['block_behavior']['elevation_band_resolution'])

    # if files passed are already clipped to each other, then no need to repeat
    if not raqc_obj.already_clipped:
        remove_files = cfg['options']['remove_clipped_files']
        raqc_obj.clip_extent_overlap(remove_files)

    raqc_obj.make_diff_mat()

    # Add check_config
    name = cfg['difference_arrays']['name']
    action = cfg['difference_arrays']['action']
    operator = cfg['difference_arrays']['operator']
    val = cfg['difference_arrays']['val']
    raqc_obj.mask_advanced(name, action, operator, val)

    # if user specified histogram outliers in user config
    if 'histogram' in flags:
        histogram_mats = cfg['histogram_outliers']['histogram_mats']
        num_bins = cfg['histogram_outliers']['num_bins']
        threshold_histogram_space = cfg['histogram_outliers']['threshold_histogram_space']
        moving_window_name = cfg['histogram_outliers']['moving_window_name']
        moving_window_size = cfg['histogram_outliers']['moving_window_size']
        raqc_obj.make_hist(histogram_mats, num_bins, threshold_histogram_space, moving_window_size)

    # if user wants to check for blocks
    for flag in ['basin_block', 'elevation_block']:
        if flag in flags:
            block_window_size = cfg['block_behavior']['moving_window_size']
            block_window_threshold = cfg['block_behavior']['neighbor_threshold']
            snowline_threshold = cfg['block_behavior']['snowline_threshold']
            outlier_percentiles = cfg['block_behavior']['outlier_percentiles']
            if flag == 'basin_block':
                apply_moving_window = cfg['flags']['apply_moving_window_basin']
                raqc_obj.basin_blocks(apply_moving_window, block_window_size, block_window_threshold, snowline_threshold)
            elif flag == 'elevation_block':
                apply_moving_window = cfg['flags']['apply_moving_window_elevation']
                elevation_thresholding = [cfg['flags']['elevation_loss'], cfg['flags']['elevation_gain']]
                raqc_obj.hypsometry(apply_moving_window, block_window_size, block_window_threshold, snowline_threshold,
                                        outlier_percentiles, elevation_thresholding)

    if 'tree' in cfg['flags']['flags']:
        logic = [cfg['flags']['tree_loss'], cfg['flags']['tree_gain']]
        raqc_obj.tree(logic)

    want_plot = cfg['options']['interactive_plot']
    if want_plot == True:
        raqc_obj.plot_this()

    file_out = cfg['paths']['file_path_out']
    include_arrays = cfg['options']['include_arrays']
    include_masks = cfg['options']['include_masks']

    raqc_obj.save_tiff(file_out, flags, include_arrays, include_masks)

    #backup config file
    config_backup_location = raqc_obj.file_path_out_backup_config
    generate_config(ucfg, config_backup_location)

if __name__=='__main__':
    main()
