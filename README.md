====
raqc
====



.. image:: https://img.shields.io/travis/zuhlmann/raqc.svg
        :target: https://travis-ci.org/zuhlmann/raqc




Purpose
-------
Inspect 3D time-series data for unrealistic spatial patterns and statistical outliers. Enables quick quality assessment of modelled and remotely-sensed 3D data products used in time-sensitive workflows such as operational snow models.

Usage
-----
Currently takes two time-sequential geoTIFFs (.tif) and outputs a map which flags suspect and potentially bad pixel locations. More flags increases likelihood that pixels are problematic. Certain flag combinations can be used to diagnose the type of error in data acquisition, processing or modeling responsible for the suspect data.


* Free software: GNU General Public License v3
* Documentation: https://raqc.readthedocs.io.


Basic Tutorial:
--------
RAQC was designed to **determine the quality** of snow depth images from lidar point clouds, and to further identify pixels and chunks of pixels that were **processed incorrectly.***  We ran into snow depth images where nans were represented by -9999 and 0 interchangeably.  This was problematic as 0 means 0m snow depth in much of the image where data WAS in fact collected.  Additionally, vegetation such as forests can lead to major errors in ranging measurements from lidar.  We attempted to flag suspect pixels and quantify the image as a whole as either "good" or "bad"

To Run:
- RAQC utilizes a configuration file managed by inicheck (https://github.com/USDA-ARS-NWRC/inicheck).  User must set all parameters here and run throught the command line.

Here is a sample user configuration file (UserConfig) <i>Note: some options MAY have changed</i>

- [difference_arrays]
**required** to visualize the 2D histogram **if** [options][interactive_plot] = y
Limited functionality currently.  Clips attributes from [name] based on operator [action], [operator] and [value].  Default is snow depth less than 1700cm and normalized difference < 20 or 2,000%.

- [flags]
*this section enables user to select which flags to include in analysis, wheter to apply moving windows when applicable and how to define the construction of each flag.*
-[flags] choose flags.  if basin_block or elevation block is selected, flags of 'loss' and 'gain will be created.  For example, [flags][basin_blocks] will yield ```flag_basin_loss``` and ```flag_basin_gain```.

'''
################################################################################
# Match, Clip and Filter Dates
################################################################################
[difference_arrays]
name:                      date1, difference_normalized
action:                     compare, compare
operator:                    less_than, greater_than, less_than, greater_than
value:                          1700, -1, 20, -1.1

################################################################################
# which outlier tests
################################################################################
[flags]
flags:                          histogram, basin_block, elevation_block, tree
apply_moving_window_basin:      True
elevation_loss:                 difference
elevation_gain:                 difference, difference_normalized
apply_moving_window_elevation:  True
tree_loss:                      and
tree_gain:                      or

################################################################################
# Histogram space parameters
################################################################################
[histogram_outliers]
histogram_mats:                date1, difference_normalized
num_bins:                      60,  200
threshold_histogram_space:     0.45, 1
moving_window_name:            bins
moving_window_size:             3

################################################################################
[block_behavior]
moving_window_size:               5
neighbor_threshold:               0.39
snowline_threshold:               40
elevation_band_resolution:        50
outlier_percentiles:              95, 95, 5, 5

################################################################################
[options]
include_arrays:                     difference, difference_normalized, date1, date2
include_masks:                      None
interactive_plot:                    n
remove_clipped_files:               False

#[section]
#item:    #value



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
