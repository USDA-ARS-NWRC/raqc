import os
import logging
# solution to almost eliminating Rastrio logging to log file
logging.getLogger('rasterio').setLevel(logging.WARNING)
import rasterio as rio
import time
import numpy as np
import math
import json
import affine
import datetime
import matplotlib.pyplot as plt
from . import plotables as pltz
from snowav.utils.MidpointNormalize import MidpointNormalize
from memory_profiler import profile

def prep_coords(file_path_dataset1, file_path_dataset2, file_path_topo, band):
    """
    Puts coordinates of all files into exact same extent and spatial resolution

    Args:
        file_path_dataset1:     file path to dataset of first date
        file_path_dataset1:     file path to dataset of second date
        file_path_topo:         file path to topo.nc file
        band:                   string with band name to use in .nc file
    Returns
        topo_rez_same:      boolean indicating if the topo.nc file has same spatial
                            resolution as the datasets
        extents_same:       boolean indicating if the extents of datasets and topo
                            are the same
        min_extents:        list with minimum overlapping extents of the 3 datasets
    """

    with rio.open(file_path_dataset1) as src:
        d1 = src
        meta = d1.profile
    tock = time.clock()

    with rio.open(file_path_dataset2) as src:
        d2 = src
        meta2 = d2.profile

    #formats string to open topo.nc file with rio.open
    topo_file_open_rio = 'netcdf:{0}:{1}'.format(file_path_topo, band)
    with rio.open(topo_file_open_rio) as src:
        d3 = src
        meta3 = d3.profile

    #grab basics
    # Note: This is the X Direction resolution.  We are assuming that X and Y
    # resolutions, when rounded are the same
    rez = meta['transform'][0]  # spatial resolution
    rez2 = meta2['transform'][0]
    rez3 = meta3['transform'][0]

    # Seemingly wierd but necessary and simple way to check if bounds
    # of both dates AND topo are the same.
    # Rio...disjoint_bounds ONLY ACCEPTS two args, hence the if statements
    # to hack effectly add 3rd argument.
    # result of rio...disjoint_bounds is True if different, False is same
    extents_same = ~ rio.coords.disjoint_bounds(d2.bounds, d1.bounds)
    if extents_same:
        extents_same2 = rio.coords.disjoint_bounds(d1.bounds, d3.bounds)
        extents_same = extents_same2 # True or False
    else:
        extents_same = extents_same  # False

    # check that date1 and date2 spatial resolutions are the same.
    if round(rez) == round(rez2):  #hack way to check all three rez the same
        pass
    else:
        sys.exit("check that spatial resolution of your two repeat array \
        files are the same must fix and try again")
    if round(rez) == round(rez3):
        topo_rez_same = True
    else:
        topo_rez_same = False

    # .nc have different standards for spatial metadata than geotiff,
    # At least the metadata pulled from rasterio
    d3_left = bounds_to_pix_coords(d3.bounds.left, rez3, 'left')
    d3_bottom = bounds_to_pix_coords(d3.bounds.bottom, rez3, 'bottom')
    d3_right = bounds_to_pix_coords(d3.bounds.right, rez3, 'right')
    d3_top = bounds_to_pix_coords(d3.bounds.top, rez3, 'top')


    # grab bounds of common/overlapping extent of date1, date2 and topo.nc
    # and prepare function call for gdal to clip to extent and align
    left_max_bound = max(d1.bounds.left, d2.bounds.left, d3_left)
    bottom_max_bound = max(d1.bounds.bottom, d2.bounds.bottom, d3_bottom)
    right_min_bound =  min(d1.bounds.right, d2.bounds.right, d3_right)
    top_min_bound = min(d1.bounds.top, d2.bounds.top, d3_top)

    # ensure nothing after decimal - nice whole number, admittedly a float
    left_max_bound = evenly_divisible_extents(left_max_bound, rez2)
    bottom_max_bound = evenly_divisible_extents(bottom_max_bound, rez2)
    right_min_bound = evenly_divisible_extents(right_min_bound, rez2)
    top_min_bound = evenly_divisible_extents(top_min_bound, rez2)

    min_extents = [left_max_bound, bottom_max_bound, right_min_bound, \
                            top_min_bound]
    rez = [rez, rez2, rez3]
    return topo_rez_same, extents_same, min_extents, rez

def evenly_divisible_extents(coord, rez):
    """
    This utility rectifies coordinates in geographically referenced datasets for
    simpler use when identical extents and spatial resolutions are required.  In
    RAQC multiple geotiffs and/or netCDF files are aligned with each other.  This
    utility rounds dataset extents of each file to coordinates evenly divisible
    by the spatial resolution.  The spatial resolution will also be rounded to
    nearest integer.

    Args:
        coord: coordinate (int or float)
        rez:    spatial resolution i.e. 50m
    Returns:
        coord_updated:  coordinate rounded to nearest multiple of rounded resolution.
                        ex) evenly_divisible_extents(2026, 49.99) returns 2050.
                            If coord = 2024 instead, 2000 returned
     See** needs tweaks to work reliable with resolutions (rez) < 2
    """

    # ** limitation mentioned in docstring

    coord_round = round(coord) % round(rez)
    if coord_round != 0:
        if coord_round > 0.5 * rez:
            coord_updated = coord + (round(rez) - (coord % round(rez)))
        else:
            coord_updated = coord - (coord % round(rez))
    else:
        coord_updated = coord
    return(coord_updated)

def bounds_to_pix_coords(bound, rez, location):
    """
    Rectifies possible quarks using rasterio, namely that <dataset>.bounds
    outputs exterior bounding box of dataset, NOT the pixel locations.
    May need more investigation, but works when matching netCDF to geotiff
    coordinates using Rasterio

    Args:
        bound:      bounding box extents from rasterio dataset of .nc file
        rez:        spatial resolution
        location:   left, right, top, or bottom bounding coordinates

    Returns:
        bound_new:  correct pixel coordinates for bounds
    """

    # dictionary converts key into string of 'out' or 'in' for shifting bounds
    # in or out
    dict_offset_coords = {'left':'out', 'right':'in', 'bottom':'out', 'top':'in'}
    location = dict_offset_coords[location]
    if location == 'in':
        bound_new = bound - rez / 2
    if location == 'out':
        bound_new = bound + rez / 2
    return(bound_new)

def rasterio_netCDF(file_path):
    """
    A quick standalone demonstration on what rasterio vs. netCDF4 provides in
    terms of bounds. Used in development and saved just in case...
    """
    from netCDF4 import Dataset
    from tabulate import tabulate

    ncfile = Dataset(file_path, 'r')
    lats = ncfile.variables['y']
    lons = ncfile.variables['x']
    netCDF_row = ['netCDF4', lons[0], lons[-1], lats[-1], lats[0]]

    topo_file_open_rio = 'netcdf:{0}:{1}'.format(file_path, 'dem')
    with rio.open(topo_file_open_rio) as src:
        d1 = src
        meta = d1.profile

    rio_row = ['rasterio', d1.bounds.left, d1.bounds.right, d1.bounds.bottom, d1.bounds.top]
    list_list = [netCDF_row, rio_row]
    col_names = ['', 'left', 'right', 'bottom', 'top']
    print('bounds from both')
    print('\n',tabulate(list_list, headers = col_names, floatfmt = ".1f"), '\n')
    print(type(d1.bounds))

def get_elevation_bins(dem, dem_mask, elevation_band_resolution):
    """
    returns edges for dem elevation binning, an array with bin id numbers
    indicating index of elevation bin, and finally a list of all the unique
    indices from the array

    Args:
        dem:                        dem array
        dem_mask:                   mask clipping dem for binning
        elevation_band_resolution:  resolution of bin increments i.e. 50m
    Returns:
        map_id_dem:         dem array with bin ids in place of elevation.
                            shape = dem_mask.shape
        elevation_edges:    bin edges for elevation binning.  Think like a
                            histogram with bin edges, but with elevation
                            bounds i.e. 2150 - 2200m has edges 2150 and 2200
                            SHAPE = elevation range of masked DEM
                                   rounded to be divisible by
                                    (elevation_band_resolution) /
                                    elevation_band_resolution

        id_dem_unique:      pretty unnecessary output as can be ascertained
                            from elevation_edges.  These are indices of
                            elevation bins from 0 to n -1 ; where n = # of
                            elevation bins

    """
    # use overlap_nan mask for snowline because we want to get average
    # snow per elevation band INCLUDING zero snow depth

    dem_clipped = dem[dem_mask]
    min_elev, max_elev = np.min(dem_clipped), np. max(dem_clipped)

    # Sets dem bins edges to be evenly divisible by the elevation band rez
    edge_min = min_elev % elevation_band_resolution
    edge_min = min_elev - edge_min
    edge_max = max_elev % elevation_band_resolution
    edge_max = max_elev + (elevation_band_resolution - edge_max)
    # creates elevation bin edges using min, max and elevation band resolution
    elevation_edges = np.arange(edge_min, edge_max + 0.1, \
                                    elevation_band_resolution)
    # Create bins for elevation bands i.e. from 1 to N where
    #   N = (edge_max - edge_min) / elevation_band_resolution
    #   For example --->
    #   if edge_min, max and resolution = 2000m, 3000m and 50m respectively
    #   then bin 1 encompasses cells from 2000m to 2050m
    #   and bin 20 cells from 2950 to 3000m
    id_dem = np.digitize(dem_clipped, elevation_edges) -1
    id_dem = np.ndarray.astype(id_dem, np.uint8)
    # get list (<numpy array>) of bin numbers (id_dem_unique)
    id_dem_unique = np.unique(id_dem)
    # initiate map of ids with nans max(id_dem) + 1
    map_id_dem = np.full(dem_mask.shape, id_dem_unique[-1] + 1, dtype=np.uint8)
    # places bin ids into map space (map_id_dem)
    map_id_dem[dem_mask] = id_dem

    return map_id_dem, id_dem_unique, elevation_edges

def check_DEM_resolution(dem_clip, elevation_band_resolution):
    """
    Brief method ensuring that DEM resolution from UserConfig can be partitioned
    indem datatype - i.e. that the elevation_band_resolution (i.e. 50m) yields
    <= 255 elevation bands based on the elevation range of topo file.

    Args:
        dem_clip:                   just a dem that may or may not be clipped
        elevation_band_resolution:  resolution of bin increments i.e. 50m
    Returns
    """
    min_elev, max_elev = np.min(dem_clip), np.max(dem_clip)
    num_elev_bins = math.ceil((max_elev - min_elev) / elevation_band_resolution)
    min_elev_band_rez = math.ceil((max_elev - min_elev) / 254)
    if num_elev_bins > 254:
        print('In the interest of saving memory, please lower (make more coarse)' \
        '\nyour elevation band resolution' \
        '\nthe number of elevation bins must not exceed 254 ' \
        '\ni.e. (max elevation - min elevation) / elevation band resolution must not exceed 254)' \
        '\nEnter a new resolution ---> (Must be no finer than {0})'.format(min_elev_band_rez))

        while True:
            elevation_band_resolution = input()
            try:
                elevation_band_resolution = float(elevation_band_resolution)
            except ValueError:
                print('must enter a float or integer')
            else:
                if elevation_band_resolution > min_elev_band_rez:
                    print('your new elevation_band_resolution will be: {}'
                            .format(elevation_band_resolution))
                    break
                else:
                    print('Value still too fine. Enter a new resolution ---> '
                        'must be no finer than{0})'.format(min_elev_band_rez))
    return(elevation_band_resolution)
def get16bit(array):
    """
    Converts array into numpy 16 bit integer

    Args:
        array:  array in meters to convert into 16bits centimeters
    """
    id_nans = array == -9999
    array_cm = np.round(array,2) * 100
    array_cm = np.ndarray.astype(array_cm, np.int16)
    array_cm[id_nans] = -9999
    return(array_cm)

def apply_dict(original_list, dictionary, nested_key):
    """
    Basically translates Config file names into attribute names if necessary
    Args:
        origingal_list:     list to be mapped using dictionary
        dictionary:         nested dictionary (nuff said)
        nested_key          key (string) of outter nest
    Returns:
        mapped_list:        list mapped to new values
    """
    mapped_list = []
    for val in original_list:
        try:
            mapped_list.append(dictionary[nested_key][val])
        except KeyError:
            mapped_list.append(val)
    return(mapped_list)

def update_meta_from_json(file_path_json):
    """
    manually add add 'transform' key to metadata.
    unable to retain with json.dump as it is not "serializable
    basically pass epsg number <int> to affine.Affine and replace
    value in metadata with output

    args:
        file_path_json:     in this case, with metadata
    returns:
        meta_orig:          updated metadata
    """
    with open(file_path_json) as json_file:
        meta_orig = json.load(json_file)
    crs_object = rio.crs.CRS.from_epsg(meta_orig['crs'])
    transform_object = affine.Affine(*meta_orig['transform'][:6])
    meta_orig.update({'crs' : crs_object, 'transform' : transform_object})
    return(meta_orig)

def create_clipped_file_names(file_path_out_base, file_path_dataset1,
                                file_path_dataset2):
    """
    Create file names for clipped snow depth files.
    Note: these are just the file names, files are not yet created

    Args:
        file_path_out_base:     base path from which to join clipped files
        file_path_dataset1:     file path original snow depth date1
        file_path_dataset2:     file path original snow depth date2
    Returns:
        file_path_date1_te:     file path of date1 clipped file
        file_path_date2_te:     file path of date2 clipped file
    """
    # Create file paths and names
    file_name_date1_te_temp = os.path.splitext(os.path.expanduser \
                                (file_path_dataset1).split('/')[-1])[0]
    #find index of date start in file name i.e. find idx of '2' in 'USCATE2019...'
    id_date_start = file_name_date1_te_temp.index('2')
    # grab file name bits from both dates to join into descriptive name
    file_name_date1_te_first = os.path.splitext \
                                (file_name_date1_te_temp)[0][:id_date_start + 8]
    file_name_date1_te_second = os.path.splitext \
                                (file_name_date1_te_temp)[0][id_date_start:]
    file_name_date2_te_temp = os.path.splitext(os.path.expanduser \
                                (file_path_dataset2).split('/')[-1])[0]
    file_name_date2_te_first = os.path.splitext \
                                (file_name_date2_te_temp)[0][:id_date_start + 8]
    file_name_date2_te_second = os.path.splitext \
                                (file_name_date2_te_temp)[0][id_date_start:]
    # ULTIMATELY what is used as file paths
    file_path_date1_te = os.path.join(file_path_out_base, \
                                        file_name_date1_te_first + '_clipped_to_' + \
                                        file_name_date2_te_second + '.tif')
    file_path_date2_te = os.path.join(file_path_out_base, \
                                        file_name_date2_te_first + '_clipped_to_' +  \
                                        file_name_date1_te_second + '.tif')

    return(file_path_date1_te, file_path_date2_te)

def format_date(date_string):
    """
    short, cheap function to convert string to datetime
    """
    if isinstance(date_string, int):
        date_string = str(date_string)

    year = int(date_string[:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    datetime_obj = datetime.date(year, month, day)
    return datetime_obj

def snowline(dem, basin_mask, elevation_band_resolution, depth1, depth2, \
                snowline_threshold):
    """
    Finds the snowline based on the snowline_threshold. The lowest elevation
    band with a mean snow depth >= the snowline threshold is set as the snowline.
    This value is later used to discard outliers below snowline.

    Args:
        snowline_threshold:     Minimum average snow depth within elevation band,
                                which defines snowline_elev.  In Centimeters
    Returns:
        snowline_elev:      Lowest elevation where snowline_threshold is
                            first encountered
    """

    map_id_dem, id_dem_unique, elevation_edges = \
        get_elevation_bins(dem, basin_mask, \
                            elevation_band_resolution)

    # initiate lists (<numpy arrays>) used to determine snowline
    snowline_mean = np.full(id_dem_unique.shape, -9999, dtype = 'float')
    # use the matrix with the deepest mean basin snow depth to determine
    # snowline thresh,  assuming deeper avg basin snow = lower snowline
    if np.mean(depth2[basin_mask]) > np.mean(depth1[basin_mask]):
        #this uses date1 to find snowline
        mat_temp = depth1
        mat = mat_temp[basin_mask]
    else:
        #this uses date2 to find snowline
        mat_temp = depth2
        mat = mat_temp[basin_mask]
    # Calculate mean for pixels with no nan (nans create errors)
    # in each of the elevation bins
    map_id_dem2_masked = map_id_dem[basin_mask]
    for id, id_dem_unique2 in enumerate(id_dem_unique):
        snowline_mean[id] = getattr(mat[map_id_dem2_masked == id_dem_unique2], 'mean')()
    # Find SNOWLINE:  first elevation where snowline occurs
    id_min = np.min(np.where(snowline_mean > snowline_threshold))
    snowline_elev = elevation_edges[id_min]  #elevation of estimated snowline

    # # Open veg layer from topo.nc and identify pixels with veg present (veg_height > 5)
    # with rio.open(self.file_path_veg_te) as src:
    #     topo_te = src
    #     veg_height_clip = topo_te.read()  #not sure why this pulls the dem when there are logs of
    #     self.veg_height_clip = veg_height_clip[0]
    # self.veg_presence = self.veg_height_clip > 5

    return snowline_elev

# dem
def determine_basin_change(file_path_snownc1, file_path_snownc2, file_path_topo,
                            file_path_base, band):
    """
    First pass (nod to Mark for the term "First Pass" for example "I have a P I,
    here is my first pass at retaining my dignity - I'll shoot 17 consecutive
    shots 2.5 ft from basket until I win") at determining if basin lost or
    gained snow.  This short protocol takes two snow.nc files
    Args:
        snownc1:    array from model run of date 1
        snownc2:    array from model run one day prior to date 2.  This assures
                    the lidar update is not included
        file_path_topo:     grab mask layer from this file
        file_path_base:     just for object naming purposes
        band:               Could theoretically select a band other than 'thickness'
    Returns:
        gaining:       True if basin gained snow.  False if basin lost snow.
                        this will be used later to shed a noisy flag
        basin_total_change:  total change within basin mask (from topo.nc) and
                            where snow was present on at least one date
        basin_avg_change:    same as total change, but average
    """
    import sys

    snownc1_file_open_rio = 'netcdf:{0}:{1}'.format(file_path_snownc1, band)
    snownc2_file_open_rio = 'netcdf:{0}:{1}'.format(file_path_snownc2, band)
    topo_file_open_rio = 'netcdf:{0}:{1}'.format(file_path_topo, 'mask')

    date1 = file_path_snownc1.split('/')[-1][3:11]
    date2 = file_path_snownc2.split('/')[-1][3:11]
    file_path_out = os.path.join(file_path_base, \
                'RAQC_modelled_basin_diff_{}_to_{}.png'.format(date1, date2))

    with rio.open(snownc1_file_open_rio) as src:
        snownc1_obj = src
        snownc1 = snownc1_obj.read()

    with rio.open(snownc2_file_open_rio) as src:
        snownc2_obj = src
        snownc2 = snownc2_obj.read()

    with rio.open(topo_file_open_rio) as src:
        mask_obj = src
        mask = mask_obj.read()

    # Get statistics on snow depth change
    # snownc1 and snownc2 are now the arrays of specified band (thickness)
    snownc1 = snownc1[0]
    snownc2 = snownc2[0]
    # basin mask array
    mask = mask[0]
    mask = np.ndarray.astype(mask, np.bool)
    # only interested in pixels with snow on at least one day
    both_zeros = (np.absolute(snownc1) ==0) & (np.absolute(snownc2) == 0)
    zeros_and_mask = mask & ~both_zeros
    # snow property change (thickness)
    diff = snownc2 - snownc1
    # only within mask where snow present
    diff_clipped_to_mask = diff[zeros_and_mask]

    basin_total_change = np.sum(diff_clipped_to_mask)
    basin_area = diff_clipped_to_mask.shape[0]
    basin_avg_change = round(basin_total_change / basin_area, 2)

    # determine if basin-wide snow depth is gaining or losing
    if basin_total_change > 0:
        gaining = True
    else:
        gaining = False

    cbar_string = '\u0394 thickness (m)'
    suptitle_string = '\u0394 snow thickness (m): snow.nc run{}_to_{}'. \
                        format(date1, date2)

    # # plot and save
    # basic_plot(diff, zeros_and_mask, cbar_string, suptitle_string, file_path_out)

    return gaining, basin_total_change, basin_avg_change

def return_snow_files(file_path_snownc, year1, year2):
    # Snow.nc file paths
    # Grab modelled snow depth one day prior to both lidar updates
    # Returns boolean True if gaining snow, and the net change
    snownc_date2 = format_date(year2)
    snownc_date2 -= datetime.timedelta(days = 1)
    snownc_date2 = snownc_date2.strftime("%Y%m%d")

    # snow.nc file paths to determine if basin is gaining or losing snow depth overall
    file_path_snownc1 = os.path.join(file_path_snownc, \
                                        'run' + year1, 'snow.nc')
    file_path_snownc2 = os.path.join(file_path_snownc, \
                                        'run' + snownc_date2, 'snow.nc')

    # temp_snow1 = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/run20190613_snow.nc'
    # temp_snow2 = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/run20190703_snow.nc'
    #
    # file_path_snownc1, file_path_snownc2 = temp_snow1, temp_snow2

    return file_path_snownc1, file_path_snownc2

def basic_plot(array, mask, cbar_string, suptitle_string, file_path_out):
    '''
    Attempt to generalize a plot function for abstraction.
    Args:
        array:      original array to plot
        mask:       mask if only considering masked area of array. For example,
                    if nans or value clipping

    '''

    # if clipping map too much, change these perecentiles
    # for instance from 1 to 0.1 and 99 to 99.9
    array_clipped_to_mask = array[mask]

    minC = np.nanpercentile(array_clipped_to_mask, 1)
    maxC = np.nanpercentile(array_clipped_to_mask, 99)

    # Now plot change
    # nans where array mask
    diff_map = np.full(array.shape, np.nan)
    diff_map[mask] = array_clipped_to_mask
    # limits used in colorbar
    cb_range_lims = [minC, maxC]

    pltz_obj = pltz.Plotables()
    pltz_obj.marks_colors()
    pltz_obj.cb_readable(cb_range_lims, 'L', 5)

    fig, axes = plt.subplots(nrows = 1, ncols = 1)
    h = axes.imshow(diff_map, cmap = pltz_obj.cmap_marks, norm=MidpointNormalize(midpoint = 0))
    axes.axis('off')
    cbar = fig.colorbar(h, ax=axes, fraction = 0.04, pad = 0.04, \
            orientation = 'vertical', extend = 'both', ticks = pltz_obj.cb_range)
    cbar.set_label(cbar_string, rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize = 8)
    h.set_clim(minC, maxC)
    fig.suptitle(suptitle_string)
    plt.savefig(file_path_out, dpi = 180)

# @profile
def debug_fctn():
    print('lets just see what the memory is')
