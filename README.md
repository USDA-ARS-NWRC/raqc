raqc

.. image:: https://img.shields.io/travis/zuhlmann/raqc.svg
        :target: https://travis-ci.org/zuhlmann/raqc

Purpose
-------
Inspect 3D time-series data for unrealistic spatial patterns and statistical outliers. Enables quick quality assessment of modelled and remotely-sensed 3D data products used in time-sensitive workflows such as operational snow models.

Usage
-----
Currently takes two time-sequential geoTIFFs (.tif) and outputs a multi-banded boolean image which flags suspect and potentially bad pixel locations diagnosing different issues, such as too much positive change, negative change or spatially clustered change. More flags increase the likelihood that pixels are problematic. Certain flag combinations can be used to diagnose the type of error in data acquisition, processing or modeling responsible for the suspect data.

RAQC was designed to **determine the quality** of snow depth images from lidar point clouds, and to further identify pixels and chunks of pixels that were **processed incorrectly.**  Processing errors and workflow flaws which produced these issues with our lidar-derived raster images resulted primarilly from: **1)** nans were represented by -9999 and 0 interchangeably.  This was problematic as 0 means 0m snow depth in much of the image where data WAS NOT collected.  Additionally, vegetation such as forests can lead to major errors in ranging measurements from lidar, wherein the digital surface model erroneously classifed vegetation regurns (branches, trees, etc) as ground returns.  We attempted to flag suspect pixels and quantify the image as a whole as either "good" or "bad"

* Free software: GNU General Public License v3

To Run:
--------
<i>RAQC utilizes a configuration file managed by inicheck (https://github.com/USDA-ARS-NWRC/inicheck).  User must set all parameters here and run throught the command line.</i>

Here is a breakdown of the configuration file(UserConfig) sections and short examples...  
&nbsp;&nbsp;<i>Note: some options MAY have changed</i>  
### [difference_arrays]  
<i>Helps to visualize the 2D histogram when ```[options][interactive_plot] = y```</i>  
- Clips array from ```[name]``` based on items ```[action]```, ```[operator]``` and ```[value]```.  
- The default is ```date1``` depth < 1700cm and ```normalized difference``` < 20 or 2,000%.
-  **ex)** Below config options will create mask date1 depths > 1700 cm, normalized change > 20 or 2,000% and nans from 2D histogram and outliers:  

```[difference_arrays]
name:                      date1, difference_normalized
action:                    compare, compare
operator:                  less_than, greater_than, less_than, greater_than
value:                      1700, -1, 20, -1.1
``` 

### [flags]
<i>this section enables user to select which flags to include in analysis, whether to apply moving windows when applicable and how to define the construction of each flag.</i>
- The ```[flags]``` section chooses flags to compute.  If ```basin_block```, ```elevation block``` or ```tree``` is selected, 'loss' and 'gain' flags will be created for each of ```basin```, ```elevation``` or ```tree``` respectively.  
&nbsp;&nbsp;**ex)** ```[flags][basin_blocks]``` will yield ```flag_basin_loss``` and ```flag_basin_gain```.
- ```[elevation_loss]``` and ```[elevation_gain]``` sections specifify whether to use ```difference``` and/or ```difference_normalized```.  If both selected, then logic is for **and** 
    - i.e. <i>find outliers</i> where **both** ```difference``` & ```difference_normalized``` exceed elevation band thresholds.
- ```tree``` flag is composed of ```elevation_block``` and/or ```basin_block``` flags, but **only flagged if** <i>vegetation is also present</i> in pixel (<i>currently defined as vegetation_height > 5m from topo.nc</i>).
- ```[tree_loss]``` and ```[tree_gain]``` are required items if ```[flags][tree]``` is specified.  The ```tree``` flag can combine or use ```elevation``` and ```basin``` flags individually or as compound conditions i.e. ```and``` or ```or```.  Options are ```elevation```, ```basin```, ```or``` or ```and```.  

### [histogram_outliers]
<i>sets parameters for 2D histogram-space outliers</i>  

**ex)**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```date1``` 60 bins with snow depth range 0 to 1700cm --> bin widths of ~ 28cm.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```difference_normalized``` 200 bins with range of -1 to 20 --> bin width of ~.10 or 10% change increments.
- ```[threshold_histogram_space]:  0.45, 1```: target cells with < ```0.45``` (45%) of cells within moving window AND >= ```1``` bin count **will be flagged**
```
[histogram_outliers]
histogram_mats:                date1, difference_normalized         #x, y axis respectively
num_bins:                      60,  200                             #x, y axis respectively
threshold_histogram_space:     0.45, 1
moving_window_name:            bins
moving_window_size:             3
```
![image](https://raw.githubusercontent.com/USDA-ARS-NWRC/raqc/master/docs/placeholder_histogram.png)

### [block_behavior]
<i> sets paramaters for ```elevation_blocks``` and ```basin_blocks```</i>
- Items ```[moving_window_size]``` and ```[neighbor_threshold]``` same as in ```[histogram outliers]``` section
- ```[snowline_threshold]``` has a default of 40cm based off trial and error.  This paramater is described in docstrings in raqc/raqc source code.
- ```[elevation_band_resolution]``` should be fine but also broad enough to get an adequate sample size (pixels) per elevation band.  ```[Elevation band resolution]``` sets increment sizes used to determine thresholds in```elevation_block``` flags.  When RAQC calculates ```elevation_blocks``` flags, outliers are defined based on normalized and raw snow depth difference at each elevation band.
- ```[outlier_percentiles]``` = [thresh_upper_norm, thresh_upper_raw, thresh_lower_norm, thresh_lower_raw] in %.  These are the thresholding percentiles used to determine ```elevation_block_loss``` (thresh_lower...) and ```elevation_block_gain``` (thresh_upper...) flags.

**ex) Thresholding values sample from USCASJ20170402_to_20170605 difference DataFrame **

| **Elevation (m)** | **Thresh: 95% change (cm)** | **Thresh 95% change (norm)** | **Thresh: 5% change (cm)** | **5% change (norm)** |  **bin count** |
| --- | --- | --- | --- | --- | --- |
| 2800 | 103 | 1.99 | -304 | -1 | 23500 |
| 2850 | 98 | 1.62 | -289 | -1 | 32400 |
| 2900 | 115 | 2.80 | -274 | -1 | 10500 |
| 2950 | 112 | 3.32 | -246 | -1 | 29600 |

<i>95% and 5% refer to upper and lower thresholds respectively</i>

```[flags]
flags:                          histogram, basin_block, elevation_block, tree
apply_moving_window_basin:      True
elevation_loss:                 difference
elevation_gain:                 difference, difference_normalized
apply_moving_window_elevation:  True
tree_loss:                      and
tree_gain:                      or
```
Using above table and UserConfig:  
&nbsp;&nbsp; **ex1)** Within the 2800m elevation bin, pixels with (```difference``` > 103cm) & (```difference_normalized``` > 1.99), the ```elevation_gain``` flag will be **True**, indicating an **Outlier Flag**

&nbsp;&nbsp; **ex2)** Within the 2800m elevation bin, pixels with (```difference``` < -304cm), the ```elevation_loss``` flag will be **True**, indicating an **Outlier Flag**

### [options]
<i>options for extra options in RAQC</i>
- ```[include_arrays]```:  Option to save clipped snow depth files and difference matrices to another geotiff
- ```[include_masks]```: Option to add masks used in flag calculations to flag geotiff  
- ```[interactive_plot]```: Will temporarilly pause during RAQC execution to display 2D histogram  
- ```[remove_clipped_files]```: Delete clipped files ('...clipped_to...') created in clip_extent_overlap() 




