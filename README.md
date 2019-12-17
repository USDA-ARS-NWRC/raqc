raqc

.. image:: https://img.shields.io/travis/zuhlmann/raqc.svg
        :target: https://travis-ci.org/zuhlmann/raqc

Purpose
-------
Inspect 3D time-series data for unrealistic spatial patterns and statistical outliers. Enables quick quality assessment of modelled and remotely-sensed 3D data products used in time-sensitive workflows such as operational snow models.

Usage
-----
Currently takes two time-sequential geoTIFFs (.tif) and outputs a multi-banded boolean image which flags suspect and potentially bad pixel locations diagnosing different issues, such as too much positive change, negative change or spatially clustered change. More flags increase the likelihood that pixels are problematic. Certain flag combinations can be used to diagnose the type of error in data acquisition, processing or modeling responsible for the suspect data.

RAQC was designed to **determine the quality** of snow depth images derived from lidar point clouds, and to further identify pixels and chunks of pixels that were **processed incorrectly.**  Processing errors and workflow flaws which produced these issues with our lidar-derived raster images resulted primarilly from: **1)** nans being represented by -9999 and 0 interchangeably between dates.  This was problematic as 0 means 0m snow depth in much of the image where data WAS NOT collected.  Additionally, vegetation such as forests can lead to major errors in ranging measurements from lidar, wherein the digital surface model erroneously classifed vegetation returns (branches, trees, etc) as ground returns.

We attempted to flag suspect pixels in multiple categories and quantify the entire image as either "good" or "bad".  The user may have some discretion in setting thresholds and choosing flags which appear applicable for the particular pair of images.  This depends on the total relative number of pixels being flagged.

Additional functionality was added for the first flight of the year.  In this case, modelled snow depth from AWSM runs could be used.

* Free software: GNU General Public License v3

To Run:
--------
<i>RAQC utilizes a configuration file managed by inicheck (https://github.com/USDA-ARS-NWRC/inicheck).  User must set all parameters here and run throught the command line.</i>

Here is a breakdown of the configuration file (UserConfig) sections and short examples...
&nbsp;&nbsp;<i>Note: some options MAY have changed</i>

Flags:
--------
There are multiple categories of flags.  The new lidar image (tif) is compared to the previous lidar image for the snow year.  If the lidar image is the first of the year then the previous day's modelled snow depth from ```snow.nc "thickness"``` can be used as a baseline.  From the change between these two images, change statistics are evaluated for potential outliers.  There are essentially two main statistical categories: ```elevation``` and ```basin```.  The names are vestigal to previous definitions, but in short ```basin``` is any pixel that went from 0 snow to some snow, or vice versa while ```elevation``` flags pixels that with extreme change (loss or gain) relative to their elevation band.  Furthermore, ```elevation``` pixels must have <b>both</b> an extreme snow depth change, and an extreme elative snow depth change.

ex) [make a table with nice example with elevation band]

<i>In first iteration of RAQC there was a ```flags``` section which allowed user to CHOOSE which flags to include.  However, this was abandoned. Now all flags are included, with the exception of ```histogram``` which takes a long time to run and has not proven useful.  The section ```histogram_outliers``` will detail options</i>

Useful information for reading the code.  This section describes how flags were made

-'loss' and 'gain' flags will be created for each of ```basin``` and ```elevation```.
- For runs with two lidar flights, ```basin_gain``` or ```basin_loss``` will be removed from ```flags``` depending on the calculated change in basin snow depth.  In most cases, ```basin_loss``` will be removed.  In most years the first flight is around peak SWE and the second flight is deeper into melt season, therefore a lot of the low and mid elevation has completed melted.  Therefore ```basin_loss``` will flag expected melt pixels erroneously.

- ```[elevation_loss]``` and ```[elevation_gain]``` result from ```difference``` **and** ```difference_normalized```.
    - i.e. <i>find outliers</i> where **both** ```difference``` & ```difference_normalized``` exceed elevation band thresholds.
- ```elevation``` flags are further constrained to pixels with <b>vegetation</b>. Vegetation is defined as vegetation_height > 5m from (topo.nc).

### [histogram_outliers]
<i>sets parameters for 2D histogram-space outliers</i>

**ex)**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```date1``` 60 bins with snow depth range 0 to 1700cm --> bin widths of ~ 28cm.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```difference_normalized``` 200 bins with range of -1 to 20 --> bin width of ~.10 or 10% change increments.
- ```[threshold_histogram_space]:  0.45, 1```: target cells with < ```0.45``` (45%) of cells within moving window AND >= ```1``` bin count **will be flagged**
```
[histogram_outliers]
histogram_mats:                date1, difference_normalized         #x, y axis respectively
action:                    compare, compare
operator:                  less_than, greater_than, less_than, greater_than
value:                      1700, -1, 20, -1.1
num_bins:                      60,  200                             #x, y axis respectively
threshold_histogram_space:     0.45, 1
moving_window_name:            bins
moving_window_size:             3
plot:                         save
```
<b>the first four options are mostly for visualization</b>
<i> ```[histogram_outliers][plot]``` = ```save``` or ```show`` creates a plot of 2D histogram</i>
- Clips array from ```[name]``` based on items ```[action]```, ```[operator]``` and ```[value]```.
- The default is ```date1``` depth < 1700cm and ```normalized difference``` < 20 or 2,000%.
-  **ex)** Above config options will create mask date1 depths > 1700 cm, normalized change > 20 or 2,000% and nans from 2D histogram plot and flags:


![image](https://raw.githubusercontent.com/USDA-ARS-NWRC/raqc/devel/docs/placeholder_histogram.png)

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

Utilities and Standalone Scripts:
--------
### ncrcat_CL.py
<i>This script digs into the AWSM directory i.e ```...ops/runs``` for subdirectories i.e. ```run20190206``` to open up and concatenate or "smash" as many ```snow.nc``` files under subdirectories housed under single directory into one ```snow.nc``` file as user specifies.  This is a <b>Command Line</b> script utilizing ```argparse```.

<b>ex1)</b>  Concatenate ALL ```snow.nc``` files within ```--dir-in``` into a single ```snow.nc``` file:

   ```>>> python ncrcat_CL.py --dir-in '/mnt/snow/albedo/sanjoaquin/ops/wy2019/ops/runs' --base-file 'snow.nc' --file-out '~/projects/data/snow_agglom3.nc'``

- For the above example:
  - ```dir-in```: the directory with the ```run<YYYYMMDD>``` subdirectories containing ```snow.nc``` files. from AWSM runs. <i>
  Note the subdirectories MUST be in the ```run<YYYYMMDD>``` format i.e. ```run20190401``` for script to parse properly.
  - ```--base-file```:  .nc filename in subdirectory to grab i.e. ```snow.nc```.  Script currently hardcoded to use ```thickness``` band from ```snow.nc``` from AWSM model run.
  Note: <i>could easily alter script to use ```em.nc``` file or other bands of ```snow.nc``` file by simply changing the ```ncrcat``` ```-v``` (variable) option in the command line call at bottom of script -  Instructions to do this are commented in script</b>
  - ```--file-out``` The file path (including filename) to output file.

<b>ex2)</b>  Specify dates to concatenate.  More specifically, specify <b>start date</b>, <b>end date</b> and <b>increment</b>.

```>>>python ncrcat_CL.py --dir-in '/mnt/snow/albedo/sanjoaquin/ops/wy2019/ops/runs' --base-file 'snow.nc' --file-out '~/projects/data/snow_agglom3.nc' --start-date '20181110' --end-date '20190121' --increment 7```

- For the above example:
&emsp;<i>Note: options/flags described in previous example remain the same</i>

  - ```--start-date``` and ```--end-date```: passed as a string of the format ```<YYYYMMDD>``` to bound dates to concatenate.
  - ```--increment```: an integer which is days to between each consecutive date in concatenated files i.e. ```range(0, end_date - start_date, increment)```
  <i>Note:</i> ```end_date - start_date``` = number of days.

<b>ex3)</b> Use em.nc file
&emsp;<i>Note</i>: in below example, you MUST change ```-v 'thickness'``` to ```v '<band name>'``` in script as described in comments for it to work.

```>>>python ncrcat_CL.py --dir-in '/mnt/snow/albedo/sanjoaquin/ops/wy2019/ops/runs' --base-file 'em.nc' --file-out '~/projects/data/em_agglom3.nc' --start-date '20181110' --end-date '20190121' --increment 7```

- For the above example:
  - Same as <b>ex2)</b> except it will concatenate the ```em.nc``` bands specified by ```-v``` flag
