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
        print_config_report(warnings,errors)

    #checking_later allows not to crash with errors.
    cfg = copy.deepcopy(ucfg.cfg)
    # cfg['flags']['flags'].extend(['test'])
    # print(cfg['flags']['flags'])
    # print(ucfg.cfg['flags']['flags'])

    # ENSURE elevation and basin were in flags[flags] if trees was
    # Gather all flags specified in config

    # TONS of logic to populate some flags if necessary
    # Tree flag requires basin_block and/or elevation_block.  Check that values are present in UserConfig
    if 'tree' not in cfg['flags']['flags']:
        tree = False
    else:
        tree = True
        tree_loss = cfg['flags']['tree_loss']
        tree_gain = cfg['flags']['tree_gain']

        # determine which flags were requested
        basin_in_flag = 'basin_block' in cfg['flags']['flags']
        elevation_in_flag = 'elevation_block' in cfg['flags']['flags']
        # initiate both_present
        both_present = False
        missing_flag = []
        while True:
            if basin_in_flag & elevation_in_flag:
                both_present = True
                break
            else:
            # [flag][flag] contains either basin, elevation or neither
                if any([wrd in (tree_loss + tree_gain) for wrd in ['or', 'and']]):
                    # both elevation and basin blocks needed but not in
                    # in UserConfig [flags][flags]
                    if basin_in_flag + elevation_in_flag == 0:
                        # need both elevation and basin flags
                        missing_flag.extend(('elevation_block', 'basin_block'))
                    else:
                        # adds missing flag string to list
                        [missing_flag.append(item) for item in \
                        [basin_in_flag * 'elevation_block', \
                         elevation_in_flag * 'basin_block'] if item != '']
                else:
                 # if no 'and' 'or' but basin, elevation or neither flag requested
                    dict_logic = {'basin' : 1, 'elevation' : -1}
                    dict_determine_required = {2 : 'basin_block', \
                                                -2 : 'elevation_block'}
                    keys = dict_logic[tree_loss] + dict_logic[tree_gain]
                    if keys == 0:
                        # both elevation and basin requested for trees,
                        # but one or both not in flags
                        [missing_flag.append(item) for item in \
                        [basin_in_flag * 'elevation_block', \
                         elevation_in_flag * 'basin_block'] if item != '']
                        break
                    else:
                        if keys == basin_in_flag * 2:
                            # tree loss and/or tree gain have needed flags (basin)
                            pass
                        elif keys == elevation_in_flag * -2:
                            # tree loss and/or tree gain have needed flags (elevation)
                            pass
                        else:
                            # tree loss and/or tree gain are missig basin or elevation
                            required = dict_determine_required[keys]
                            missing_flag.append(required)
            if not both_present:
                if len(missing_flag) == 2:
                    missing_temp = missing_flag.copy()
                    missing_temp.insert(1, 'and')
                    missing_concat = ' '.join(missing_temp)
                    missing_concat = 'the {} flags'.format(missing_concat)
                else:
                    missing_concat = 'the {} flag'.format(missing_flag[0])
                print(("\nThe tree flag was requested in the UserConfig."
                        "\nThis REQUIRES both elevation_block and basin_block flags."
                        "\nYou are MISSING: {0}"
                        "\nWould you like to add the flag(s)?"
                        "\ntype 'yes' or 'no'"
                        "\nIf 'no', system will exit").format(missing_concat))
                response = input()
                if response.lower() == 'yes':
                    cfg['flags']['flags'].extend(missing_flag)
                    pass
                elif response.lower() == 'no':
                    sys.exit("exiting program")
                else:
                    print('please answer "yes" or "no"')
                break

    raqc_obj = multi_array.Flags(cfg['paths']['file_path_in_date1'],
                cfg['paths']['file_path_in_date2'], cfg['paths']['file_path_topo'],
                cfg['paths']['file_path_out'], cfg['paths']['basin'],
                cfg['paths']['file_name_modifier'],
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
    value = cfg['difference_arrays']['value']
    raqc_obj.mask_basic()
    raqc_obj.mask_advanced(name, action, operator, value)

    # if user specified histogram outliers in user config
    if 'histogram' in cfg['flags']['flags']:
        histogram_mats = cfg['histogram_outliers']['histogram_mats']
        num_bins = cfg['histogram_outliers']['num_bins']
        threshold_histogram_space = cfg['histogram_outliers']['threshold_histogram_space']
        moving_window_name = cfg['histogram_outliers']['moving_window_name']
        moving_window_size = cfg['histogram_outliers']['moving_window_size']
        raqc_obj.make_histogram(histogram_mats, num_bins, threshold_histogram_space, moving_window_size)

    # if user wants to check for blocks
    for flag in ['basin_block', 'elevation_block']:
        if flag in cfg['flags']['flags']:
            block_window_size = cfg['block_behavior']['moving_window_size']
            block_window_threshold = cfg['block_behavior']['neighbor_threshold']
            snowline_threshold = cfg['block_behavior']['snowline_threshold']
            outlier_percentiles = cfg['block_behavior']['outlier_percentiles']
            if flag == 'basin_block':
                apply_moving_window = cfg['flags']['apply_moving_window_basin']
                raqc_obj.flag_basin_blocks(apply_moving_window, block_window_size, block_window_threshold, snowline_threshold)
            elif flag == 'elevation_block':
                apply_moving_window = cfg['flags']['apply_moving_window_elevation']
                elevation_thresholding = [cfg['flags']['elevation_loss'], cfg['flags']['elevation_gain']]
                raqc_obj.flag_elevation_blocks(apply_moving_window, block_window_size, block_window_threshold, snowline_threshold,
                                        outlier_percentiles, elevation_thresholding)


    if tree:
        logic = [tree_loss, tree_gain]
        raqc_obj.flag_tree_blocks(logic)

    raqc_obj.effect_flags()
    want_plot = cfg['options']['plot']
    raqc_obj.plot_this(want_plot)

    file_out = cfg['paths']['file_path_out']
    include_arrays = cfg['options']['include_arrays']
    include_masks = cfg['options']['include_masks']

    raqc_obj.save_tiff(file_out, cfg['flags']['flags'], include_arrays, include_masks)

    #backup config file
    config_backup_location = raqc_obj.file_path_out_backup_config
    ucfg.cfg = cfg
    generate_config(ucfg, config_backup_location)

if __name__=='__main__':
    main()
