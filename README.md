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
<i>Helps to visualize the 2D histogram when ```[options][interactive_plot] = y```</i>  
- Clips array from ```[name]``` based on items ```[action]```, ```[operator]``` and ```[value]```.  
- the default is ```date1``` depth < 1700cm and ```normalized difference``` < 20 or 2,000%.
-  ex) Below config options will create mask with date1 depth < 1700 cm, normalized change < 20 or 2,000% and **NO** nans.  

```[difference_arrays]
name:                      date1, difference_normalized
action:                     compare, compare
operator:                    less_than, greater_than, less_than, greater_than
value:                          1700, -1, 20, -1.1
``` 

### [flags]
<i>this section enables user to select which flags to include in analysis, whether to apply moving windows when applicable and how to define the construction of each flag.</i>
- ```[flags]``` chooses flags to compute.  If ```basin_block```, ```elevation block``` or ```tree``` is selected, 'loss' and 'gain' flags will be created for each of ```basin```, ```elevation``` or ```tree``` respectively.  
    **ex)** ```[flags][basin_blocks]``` will yield ```flag_basin_loss``` and ```flag_basin_gain```.
- ```[elevation_loss]``` and ```[elevation_gain]``` sections specifify whether to use ```difference``` and/or ```difference_normalized```.  If both selected, then logic is for **and** i.e. <i>find outliers</i> where **both** ```difference``` & ```difference_normalized``` exceed elevation band thresholds.
- ```tree``` flag is derived from ```elevation_block``` and/or ```basin_block``` flags, but only flagged **if** vegetation is also present in pixel (currently defined as vegetation_height > 5m from topo.nc).
- ```[tree_loss]``` and ```[tree_gain]``` are required items if ```[flags][tree]``` is specified.  The ```tree``` flag can combine or use ```elevation``` and ```basin``` flags individually or as compound conditions i.e. ```and``` or ```or```.  Options are ```elevation```, ```basin```, ```or``` or ```and```.  

### [histogram_outliers]
<i>sets parameters for 2D histogram space outliers</i>  
**ex)** 
- ```[num_bins]```  [x-axis bin count, y-axis bin count]: 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```date1``` 60 bins with a snow depth range of 0 to 1700cm and bin widths of ~ 28cm.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```difference_normalized``` 200 bins with range of -1 to 20 and bin width of ~.10 or 10% change increments.
- ```[threshold_histogram_space]:  0.45, 1```: target cells with < ```0.45``` or 45% of cells within moving window of >= ```1``` bin count **will be flagged**
```
[histogram_outliers]
histogram_mats:                date1, difference_normalized
num_bins:                      60,  200
threshold_histogram_space:     0.45, 1
moving_window_name:            bins
moving_window_size:             3
```
### [block_behavior]
<i> sets paramaters for ```elevation_blocks``` and ```basin_blocks```</i>
- Items ```[moving_window_size]``` and ```[neighbor_threshold]``` same as in ```[histogram outliers]``` section
- ```[snowline_threshold]``` has a default of 40cm based off trial and error.  This paramater is described in docstrings in raqc/raqc
- ```[elevation_band_resolution]``` should be fine but also broad enough to get an adequate sample size (pixels) per elevation band.  ```[Elevation band resolution]``` sets increment sizes used to determine thresholds in```elevation_block``` flags.  When RAQC calculates ```elevation_blocks``` flags, outliers are defined based on normalized and raw snow depth difference at each elevation band.
- ```[outlier_percentiles]``` = [thresh_upper_norm, thresh_upper_raw, thresh_lower_norm, thresh_lower_raw] in %.  These are the thresholding percentiles used to determine ```elevation_block_loss``` (thresh_lower...) and ```elevation_block_gain``` (thresh_upper...) flags.
- **ex)**  
| **Elevation** | **95% change (m)** | **95% change ((date1 - date2)  date1)** | **5% change (m)** | **5% change ((date1 - date2)  date1)** |  **bin count** |
| --- | --- | --- | --- | --- | --- |
| 2800 | 103 | 1.99 | -304 | -1 | 23500 |
| 2850 | 98 | 1.62 | -289 | -1 | 32400 |
| 2900 | 115 | 2.80 | -274 | -1 | 10500 |
| 2950 | 112 | 3.32 | -246 | -1 | 29600 |



