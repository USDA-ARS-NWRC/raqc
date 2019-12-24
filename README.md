raqc

.. image:: https://img.shields.io/travis/zuhlmann/raqc.svg
        :target: https://travis-ci.org/zuhlmann/raqc

Purpose
-------
Inspect 3D time-series data for unrealistic spatial patterns and statistical outliers. Enables quick quality assessment of modelled and remotely-sensed 3D data products used in time-sensitive workflows such as operational snow models.

Usage
-----
Takes two time-sequential raster files and outputs a multi-banded boolean image which flags suspect and potentially bad pixel locations diagnosing different issues, such as too much positive change, negative change or spatially clustered change. More flags increase the likelihood that pixels are problematic. Certain flag combinations can be used to diagnose the type of error in data acquisition, processing or modeling responsible for the suspect data.

RAQC was designed to **determine the quality** of snow depth images derived from lidar point clouds, and to further identify pixels and chunks of pixels that were **processed incorrectly.**  Processing errors and workflow flaws which produced these issues with our lidar-derived raster images resulted primarilly from: **1)** nans being represented by -9999 and 0 interchangeably between dates.  This was problematic as 0 means 0m snow depth in much of the image where data WAS NOT collected.  Additionally, vegetation such as forests can lead to major errors in ranging measurements from lidar, wherein the digital surface model erroneously classifed vegetation returns (branches, trees, etc) as ground returns.

We attempted to flag suspect pixels in multiple categories and quantify the entire image as either "good" or "bad".  The user may have some discretion in setting thresholds and choosing flags which appear applicable for the particular pair of images.  This depends on the total relative number of pixels being flagged.

Additional functionality was added for the first flight of the year when there is no earlier lidar TIF to compare.  In this case, a modelled snow depth output from an AWSM run (netCDF file) could be used as the "first flight", or date1.

* Free software: GNU General Public License v3


Concepts: What are Flags
------
Flags are the indicators of problem pixels in a lidar update.  There are multiple categories of flags.  The new lidar image (tif) is compared to the previous lidar image for the snow year.  If the lidar image is the first of the year then the previous day's modelled snow depth from ```snow.nc "thickness"``` can be used as a baseline.  From the change between these two images, change statistics are evaluated for potential outliers.  There are essentially two main statistical categories:
- ```elevation``` and ```basin```.  The names are vestigal to previous definitions, but in short ```basin``` is any pixel that went from 0 snow depth to some snow depth, or vice versa while ```elevation``` flags pixels with extreme change (loss or gain) relative to their elevation band.  Each will create either or both a ```loss``` and or ```gain``` flag. The ```elevation``` flags are more precise than ```basin```. They must have <b>both</b> an extreme snow depth change, and an extreme relative snow depth change.
  - For instance:
    - April 1 depth at pixel = 15m and June 1 depth at pixel = 12m would have 15-12 = <b>3m</b> change, but only 3/15 = <b>20%</b> relative change.
    - However another pixel on April 1 = 6m and on June 1 = 3m would have 6-3 = <b>3m</b> change, but 3/6 = <b>50%</b> relative change

- Additionally there is a <b>zero_to_nan</b> flag and a <b>histogram</b> flag.  The former flag highlights pixels that had a zero in one date and nan in another.  The ```histogram``` flag was not found to be useful, but the image is maintained in RAQC in case somebody wants to develop it further.

In first iteration of RAQC there was a ```flags``` section which allowed user to CHOOSE which flags to include.  However, this was abandoned. Now all flags are included, with the exception of ```histogram``` which takes a long time to run and has not proven useful.  The section ```histogram_outliers``` will detail options

Additionally, only one of ```basin_gain``` and ```basin_loss``` are included in most circumstances.  If the basin is generally gaining snow then only ```basin_loss``` will be be output.  Similiarly for ```basin_loss``` as these can be noisy.

- For runs with two lidar flights, ```basin_gain``` or ```basin_loss``` will be removed from ```flags``` depending on the calculated change in basin snow depth.  In most cases, ```basin_loss``` will be removed.  In most years the first flight is around peak SWE and the second flight is deeper into melt season, therefore a lot of the low and mid elevation has completed melted.  Therefore ```basin_loss``` will flag expected melt pixels erroneously.

- ```[elevation_loss]``` and ```[elevation_gain]``` result from ```difference``` **and** ```difference_normalized```.
    - i.e. <i>find outliers</i> where **both** ```difference``` & ```difference_normalized``` exceed elevation band thresholds.
- ```elevation``` flags can be further constrained to pixels with <b>vegetation</b>. Vegetation is defined as vegetation_height > 5m from (topo.nc).

To Run:
--------
<i>RAQC utilizes a configuration file managed by inicheck (https://github.com/USDA-ARS-NWRC/inicheck).  User must set all parameters here and</i> <b>run throught the command line like this:</b>



```racq --user-config -path/to/userconfig.ini```

<i>this will run RAQC using your user config.  That's how it runs...OH WAIT! What about the UserConfig???</i>



Config File Options (the guts...)
-------------

Here is a breakdown of the configuration file (UserConfig) sections and short examples...
&nbsp;&nbsp;

### [paths]
```
file_path_in_date1:   /home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/USCASJ20190614_clipped_to_20190704_SUPERsnow_depth_50p0m_agg_merged.tif
; file_path_in_date2:  /home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/USCASJ20190704_clipped_to_20190614_SUPERsnow_depth_50p0m_agg.tif
file_path_in_date1:   /mnt/snow/albedo/sanjoaquin/ops/wy2019/ops/runs/run20190613/snow.nc
file_path_in_date2:   /home/zachuhlmann/projects/data/SanJoaquin/USCASJ20190614_SUPERsnow_depth_50p0m_agg.tif
; file_path_in_date2:   /home/zachuhlmann/projects/data/SanJoaquin/USCASJ20190704_SUPERsnow_depth_50p0m_agg_merged.tif
file_path_topo:     /home/zachuhlmann/projects/data/SanJoaquin/SanJoaquin_topo.nc
file_path_out:       /home/zachuhlmann/projects/data/
basin:                SanJoaquin
file_name_modifier:     devel
```
- `file_path_in_date1`: Path to date1 snow file. Can be lidar tif or snow.nc.
- `file_path_in_date2`: Path to date2 snow file.  Must be lidar tif.
- `file_path_topo`: Path to topo.nc file
- `file_path_out`:  Just the path, not the filename.  Branched from this will be subdirectory structure identical to model runs (mostly) i.e. file_path_out/20190614_20190704/
- `file_name_modifier`: This simply adds a specifier to the name of most output files from RAQC runs.  Base file name is essentially a concatenation of the basin abbreviation and dates 1 and 2.  The `file_name_modifier` allows user to create unique base name for different runs while experimenting with options.
  - For example, BASIN_ABBR_Date1_Date2_no_mov_wind and BASIN_ABBR_Date1_Date2_veg could be created by setting `file_name_modifier` to `no_mov_wind` and `veg` respectively.
- <b>Note</b>: `file_path_date1` and `file_path_date2` can be explicitly set as the clipped files if RAQC run previously with same files. However RAQC will also detect their presence automatically during RAQC run, and alert user that they will be used.
### [histogram_outliers]
<i>sets parameters for 2D histogram-space outliers</i>

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
- the first four options are mostly for visualization
<i> ```[histogram_outliers][plot]``` = ```save``` or ```show`` creates a plot of 2D histogram</i>
  - show` Will temporarilly pause during RAQC execution to display 2D histogram
- Clips array from ```[name]``` based on items ```[action]```, ```[operator]``` and ```[value]```.
- The default is ```date1``` depth < 1700cm and ```normalized difference``` < 20 or 2,000%.
-
-  **ex)** <i> Explanation of `[histogram_outliers]` config options above.  Refer to image below for visualization</i>
  - ```value```: Remove date1 depths > 1700 cm and normalized change > 20 or 2,000% from 2D hist
  - ```num_bins```: set x axis to ```date1``` depth at 60 bins with snow depth range 0 to 1700cm --> bin widths of ~ 28cm. Set y-axis to ``difference_normalized``` into 200 bins with range of -1 to 20 --> bin width of ~.10 or 10% change increments.
  - ```[threshold_histogram_space]```:  Target cells with < 0.45 (45%) of cells within moving window AND >= 1 bin count **will be flagged** (yellow cells in right panel)

![image](https://raw.githubusercontent.com/USDA-ARS-NWRC/raqc/devel/docs/placeholder_histogram.png)


### [block_behavior]
<i> sets moving window paramaters for ```elevation_blocks``` and ```basin_blocks```. The moving window helps to identify blocks of pixels and remove scattered pixels that are flagged</i>
```
[block_behavior]
apply_moving_window:              False
moving_window_size:               5
neighbor_threshold:               0.39
```

- ```[apply_moving_window]``` -- set to ```True``` to pass moving window over ```elevation``` and ```basin``` flags.
- ```[moving_window_size]``` is in pixels and ```[neighbor_threshold]``` is the proportion of pixels within moving window below which pixels are removed as flags.
- ex)  ```moving_window_size``` = 5 and ```neighbor_threshold``` = 0.39 translates to a 5 x 5 pixel moving window passed over ```elevation``` and ```basin``` flags.  At each target pixel, moving window is centered.  If less than 39% of pixels within window are flagged, then target pixel flag is removed.

### [thresholding]
<i>Mostly thresholds and parameters that determine thresholds.</i>
```
snowline_threshold:               40
elevation_band_resolution:        20
outlier_percentiles:              95, 95, 5, 5
want_thresholds_plot:             True
resampling_percentiles:            0.1, 0.2, 0.3
elev_flag_only_veg:                False
```

- ```[snowline_threshold]``` has a default of 40cm based off trial and error.  This paramater is described in docstrings in raqc/raqc source code.  It is the threshold to determine snowline from the date with the lowest snow depth.  It is the lowest elevation band with a mean snow depth greater than the <snowline_threshold>.
  - ex) ```snowline_threshold``` = 40m finds the lowest elevation band with a mean snow depth greater than 40m
- ```[elevation_band_resolution]``` is in meters.  This is the binning resolution to partition DEM into elevation bins.  For example 50m would result in bins at 2050 - 2100m, 2100 - 2150m and so on. It should be fine but also broad enough to get an adequate sample size (pixels) per elevation band.  When RAQC calculates ```elevation``` flags, outliers are defined based on normalized and raw snow depth difference outliers at each elevation band.  Binning too fine or too course may affect the precision of these thresholds.  Zach used a resolution of 50m but could be anything down to roughly 20m (limited by UINT8 data type used to conserve memory)
- ```[outlier_percentiles]``` = [thresh_upper_norm, thresh_upper_raw, thresh_lower_norm, thresh_lower_raw] in %.  These are the thresholding percentiles used to determine ```elevation_loss``` (thresh_lower...) and ```elevation_gain``` (thresh_upper...) flags.
- ```[want_thresholds_plot]``` Set to ```True``` to save a graphical representation of thresholds used to set ```elevation``` flags.  Will be saved in output file directory.
- ```[resampling_perentiles]``` <i>only applied if input files were less than 50m</i>.  This is a list of any size with thresholds to set flags when downsampling from 3m to 50m for example.  Each value from list will result in a band in the ```resampled_thresh_flags```.
  - ex) ```0.1, 0.2, 0.3``` will output a band for each the 10%, 20% and 30% thresholds.  For a simple example, if a <b>1m</b> input file was resampled to <b>5m</b> pixels, each resulting pixel in the 5m image would need to be composed of 25 (5x5) original 1m pixels somehow.  In this case, the 0.1 threshold indicates that the resampled 0.1 band will only register flags in pixels where greater than 0.1 * 25 = 2.5 1m pixels were present in the resampled 5m pixels.  In this case, there must be at least 3 pixels out of the original 25 composing the new image that were flagged.  A band for each of the 0.1, 0.2 and 0.3 thresholds will be present in the ```flags_resampled_thresh``` output tif. In this case, a 3 banded tif is created. These values can be custom set in config file to whatever thresholds.
- ```[elev_flag_only_veg]``` is ```True``` or ```False```.  Set to ```True``` to restrict ```elevation``` flags to ONLY pixels where veg > 5m is present.  Topo.nc veg height layer is used to determine this.  In other words, if this is set to ```True``` then the ```elevation_flags``` can ONLY occur where veg height > 5m.

**ex) Thresholding values sample from USCASJ20170402_to_20170605 difference DataFrame **

| **Elevation (m)** | **Thresh: 95% change (cm)** | **Thresh 95% change (norm)** | **Thresh: 5% change (cm)** | **5% change (norm)** |  **bin count** |
| --- | --- | --- | --- | --- | --- |
| 2800 | 103 | 1.99 | -304 | -1 | 23500 |
| 2850 | 98 | 1.62 | -289 | -1 | 32400 |
| 2900 | 115 | 2.80 | -274 | -1 | 10500 |
| 2950 | 112 | 3.32 | -246 | -1 | 29600 |

<i>95% and 5% refer to upper and lower thresholds respectively</i>


Using above table and UserConfig:
&nbsp;&nbsp; **ex1)** Within the 2800m elevation bin, pixels with (```difference``` > 103cm) & (```difference_normalized``` > 1.99), the ```elevation_gain``` flag will be **True**, indicating an **Outlier Flag**

&nbsp;&nbsp; **ex2)** Within the 2800m elevation bin, pixels with (```difference``` < -304cm), the ```elevation_loss``` flag will be **True**, indicating an **Outlier Flag**

### [mandatory_options]
<i>Most are mandatory...</i>
- ```[method_determine_gaining]```: slightly confusing parameter.  This determines whether both ```basin_gain``` and ```basin_loss``` will be included in ```flag``` array.  This is determined by calculation in RAQC method ```determine_basin_change``` which uses a snow.nc file from the date of first flight and another one day prior to date of second flight (just in case Direct Insertion already happened).  If the basin lost snow overall between these dates, then ```basin_loss``` flag is removed, and likewise if the basin gained snow the ```basin_gained``` flag is removed.  As mentioned earlier, ```basin_loss``` can be noisy if basin is losing.  For instance if date1 is April 1 and date2 is June1 then the lower and mid elevations will have melted a lot.  We would not want ```basin_loss``` as it would potentially include lots of that loss.  ```method_determine_gaining``` has five options:
  1. <b>snownc</b>: select this and pass a snownc file path in [`gaining_file_path_snow_nc`].  This takes snow nc files from the dates mentioned in the first bullet and finds whether basin gained or lost snow overall.
  2. <b>manual_loss</b> or <b>manual_gain</b>.  User can simply use their experience to proclaim loss or gain and RAQC will remove flags accordingly.
  3. <b>read_meta</b>: if RAQC already run on these files and files still remain, then gaining determination already calculated and saved to ...metadata.txt file.  This option will use that previous classification and avoid more calculations.
  4. <b>neither</b>: If user wants to include BOTH ```Basin_gain``` and ```basin_loss```.  This option will include both flags in output.
- `gaining_file_path_snownc`: if ```snownc``` selected above, then this is filepath to run/runYYYYMMDD/snow.nc
- ```[include_arrays]```:  Option to save clipped snow depth files and difference matrices to a separate geotiff file.  RECOMMENDED! for visualizing.  Include all bands you want in output. Output will be multi-banded tiff
- ```[remove_clipped_files]```: Delete clipped files ('...clipped_to...') created in clip_extent_overlap()
-

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
