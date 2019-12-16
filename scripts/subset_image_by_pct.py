import basin_setup.basin_setup as bs
import copy
import math
from subprocess import run
'''
small script to clip larger tifs to smaller sizes.  Using smaller files enables
quicker raqc runs.  Use clipped files on larger basins with the 3m files to
reduce time and memory usage while tinkering with config file values.
Preview the effect of config options on flags before running raqc on 3m files.
Input files will NOT be overwritten. NOTE: the parse_extent method from
basin_setup currently does not have the cellsize_return option available for
tifs, but there is a pull request on github to incorporate code I wrote to do so.
Also, spatial resolutions of 50m and 3m are hardcoded into this script. Change
3m to whatever your input is, and 50m to something else if you need. Just replace
all '50' with <new val> and likewise for '3' in the code.

Args (hardcoded):
    month_day, fp:  these create the file paths for snow date 1 and 2
    fp_clipped: output filepaths for clipped files
    subset_pct: percentage (technically proportion) to shrink down clipped image
'''


month_day = '0422'
month_day2 = '0601'
fp = '/home/zachuhlmann/projects/data/SanJoaquin/USCASJ2018{}_SUPERsnow_depth.tif'.format(month_day)
fp2 = '/home/zachuhlmann/projects/data/SanJoaquin/USCASJ2018{}_SUPERsnow_depth.tif'.format(month_day2)
fp_clipped = '/home/zachuhlmann/projects/data/SanJoaquin/USCASJ2018{}_SUPERsnow_depth_subsetZ_del.tif'.format(month_day)
fp_clipped2 = '/home/zachuhlmann/projects/data/SanJoaquin/USCASJ2018{}_SUPERsnow_depth_subsetZ_del.tif'.format(month_day2)
# IMPORTANT!! the relative size of subsetted image.  Will be roughly clipped
# from center of min bounding extent box
subset_pct = 0.10

#extents
ext = bs.parse_extent(fp, cellsize_return = True)
ext2 = bs.parse_extent(fp2, cellsize_return = True)
xmin, ymin, xmax, ymax = ext[0], ext[1], ext[2], ext[3]
xmin2, ymin2, xmax2, ymax2 = ext2[0], ext2[1], ext2[2], ext2[3]
# find minimum bounding extent of both images
xmin_overlap = max(xmin, xmin2)
ymin_overlap = max(ymin, ymin2)
xmax_overlap = min(xmax, xmax2)
ymax_overlap = min(ymax, ymax2)

# x length of min bounding extent
x_meters = xmax - xmin
# y length of min bounding extent
y_meters = ymax - ymin
box_options = []

# subset image to a box
# Assume image is square
# clip to a box whose length of sides is a proportion of the x_dimension size
subset_box_length = math.ceil(x_meters * subset_pct)
rem = subset_box_length % 50
# get subset box dim to multiple of spatial rez
subset_box_length = subset_box_length - rem
# ensure that subset box length is divisible by input and output spatial rez
# start from original subset_box length and grow by multiples of smaller rez
# (3m) until divisible by both large (50m) and small (3) rez
for i in range(0,1000,50):
    div_by_3 = ((subset_box_length + i) % 3) == 0
    div_by_50 = ((subset_box_length + i) % 50) == 0
    if not (div_by_3 and div_by_50):
        pass
    else:
        subset_box_length = subset_box_length + i
        break

# Example: if subset_pct = 0.4, then user wants the clipped image to be 40%
# the size of original.  Therefore it will begin 30% in from xmin and end at
# 70% of xmin or 30% from xmax.  The Y dimension will be different since
# y_meters probably != x_meters.  In example, it will begin at 30% from ymin,
# but it's ending will be 0.3 * y_meters + subset_box_length.  Therefore
# box will be centered on x axis but not necessarilly on y-axis

off_pct = (1 - subset_pct) / 2
x_off1 = x_meters * off_pct
rem_x = x_off1 % 3
# get first x offset
# round to whole number divisible by smaller rez (3)
# x_off1 meters from xmin to clip in
x_off1 = x_off1 - rem_x
# UTM easting to clip xmin
x_off1 = x_off1 + xmin
# UTM easting to clip xmax
x_off2 = x_off1 + subset_box_length

# same as for x
y_off1 = y_meters * off_pct
rem_y = y_off1 % 3
# round to whole number divisible by smaller rez (3)
y_off1 = y_off1 - rem_y
#UTM northing to clip ymin
y_off1 = y_off1 + ymin
#utm northing to clip ymax
y_off2 = y_off1 + subset_box_length
off_idx = [x_off1, y_off1, x_off2, y_off2]

# convert float list to string list
te_str = []
[te_str.append(int(te)) for te in off_idx]

# run gdal to clip to target extents(te) for both files
file_path = [fp, fp2]
file_path_clipped = [fp_clipped, fp_clipped2]
for i in range(2):
    fp = file_path[i]
    fp_clipped = file_path_clipped[i]
    run_str = 'gdalwarp -te {} {} {} {} {} {} -overwrite'.format(*te_str, fp, fp_clipped)
    print('\n', '-'*25, 'Your command line call:', '-'*25, '\n', run_str, '\n')
    run(run_str, shell = True)
