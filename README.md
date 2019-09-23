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
### [difference_arrays]  
<i>required to visualize the 2D histogram when ```[options][interactive_plot] = y```</i>  
- Clips array from ```[name]``` based on items ```[action]```, ```[operator]``` and ```[value]```.  
- Default is snow depth less than 1700cm and normalized difference < 20 or 2,000%.
- <b>ex)</b>  
    

### [flags]
<i>this section enables user to select which flags to include in analysis, wheter to apply moving windows when applicable and how to define the construction of each flag.</i>
- ```[flags]``` choose flags.  if ```basin_block``` or ```elevation block``` is selected, flags of 'loss' and 'gain will be created for basin or elevation respectively.  
    **For example:** ```[flags][basin_blocks]``` will yield ```flag_basin_loss``` and ```flag_basin_gain```.
-  ex)
```[difference_arrays]
name:                      date1, difference_normalized
action:                     compare, compare
operator:                    less_than, greater_than, less_than, greater_than
value:                          1700, -1, 20, -1.1
```
Below config options will date1 depth < 1700 cm, normalized change < 20 or 2,000% and nans.  Saved to --->  ``self.overlap_conditional```
