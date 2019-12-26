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
import basin_setup.basin_setup as bs

def prep_coords(file_path_dataset1, file_path_dataset2, file_path_topo):
    """
    Puts coordinates of all files into exact same extent and spatial resolution.
    Also passes extents to evenly_divisible_extents method which rounds extents
    to being evenly divisible by spatial rez i.e. if spatial rez = 50 then extents
    must be multiples of 50, so 300025m would not be acceptable and would be
    rounded to 300050m.

    Args:
        file_path_dataset1:     file path to dataset of first date (tif or nc)
        file_path_dataset2:     file path to dataset of second date
        file_path_topo:         file path to topo.nc file
    Returns
        topo_rez_same:      boolean indicating if the topo.nc file has same spatial
                            resolution as the datasets
        extents_same:       boolean indicating if the extents of datasets and topo
                            are the same
        bounds_even:        list with minimum overlapping extents of the 3 datasets
        rez:            list of the three spatial resolutions of the 3 datasets
    """

    d1_coords = bs.parse_extent(file_path_dataset1, cellsize_return = True)
    d2_coords = bs.parse_extent(file_path_dataset2, cellsize_return = True)
    d3_coords = bs.parse_extent(file_path_topo, cellsize_return = True)

    #grab basics
    # Note: This is the X Direction resolution.  We are assuming that X and Y
    # resolutions, when rounded are the same
    d1_extents = d1_coords[:4]
    d2_extents = d2_coords[:4]
    d3_extents = d3_coords[:4]
    # spatial resolution
    rez1 = d1_coords[4]
    rez2 = d2_coords[4]
    rez3 = d3_coords[4]

    # determine if extents of topo and datasets are same (they're probably not)
    extents_same = True
    for ext1, ext2, ext3 in zip(d1_extents, d2_extents, d3_extents):
        extents_same = extents_same * (ext1 == ext2 == ext3)

    # check that date1 and date2 spatial resolutions are the same.
    if round(rez1) == round(rez2):  #hack way to check all date1 and date2 rez the same
        pass
    else:
        sys.exit("The spatial resolutions of date1 and date2 snow files need to match \
        Date1 is {0} and Date 2 is{1}. \
        The resolutions must be the same \
        If First Flight and using snow.nc file, use: \
         'gdalwarp -tr <rez lidar tiff> <rez lidar tiff> -r <path/to/input> <path/to/output> \
         to get matching spatial rez THEN pass as dataset1 and rerun RAQC".format(rez1, rez2))

    topo_rez_same = round(rez1) == round(rez3)

    # grab bounds of common/overlapping extent of date1, date2 and topo.nc
    # and prepare function call for gdal to clip to extent and align
    bounds = []
    bounds.append(max(d1_extents[0], d2_extents[0], d3_extents[0]))
    bounds.append(max(d1_extents[1], d2_extents[1], d3_extents[1]))
    bounds.append(min(d1_extents[2], d2_extents[2], d3_extents[2]))
    bounds.append(min(d1_extents[3], d2_extents[3], d3_extents[3]))

    # ensure nothing after decimal - nice whole number, admittedly a float
    bounds_even = []
    for bound in bounds:
        bounds_even.append(evenly_divisible_extents(bound, rez2))

    rez = [rez1, rez2, rez3]
    return topo_rez_same, extents_same, bounds_even, rez

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
    Brief method ensuring that DEM resolution from UserConfig is not so fine that
    range of DEM is partitioned into more bands than UINT8 datatype can classify
    - i.e. that the elevation_band_resolution (50m, 20m, etc.) yields
    <= 255 elevation bands based on the elevation range of topo file.

    Args:
        dem_clip:                   just a dem that may or may not be clipped
        elevation_band_resolution:  resolution of bin increments i.e. 50m
    Returns
        elevation_band_resolution:  returns elevation band resolution which
                                    may have changed by user to comply with
                                    maximum resolution of UINT8 - 254 elev bands
    """
    dem_clip[dem_clip==-9999] = np.nan
    min_elev, max_elev = np.nanmin(dem_clip), np.nanmax(dem_clip)
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
    The 'te' refers to gdal flag 'te' or 'target extent'

    Args:
        file_path_out_base:     base path from which to join clipped files
        file_path_dataset1:     file path original snow depth date1
        file_path_dataset2:     file path original snow depth date2
    Returns:
        file_path_date1_te:     file path of date1 clipped file
        file_path_date2_te:     file path of date2 clipped file
    """
    is_date1_nc = os.path.splitext(file_path_dataset1)[-1] == '.nc'

    if not is_date1_nc:
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

    else:
        date1 = os.path.splitext(file_path_dataset1)[0].split('/')[-2][3:]
        # the file name without format extension (.nc)
        file_name_date2_te_temp = os.path.splitext(os.path.expanduser \
                                    (file_path_dataset2).split('/')[-1])[0]
        #find index of date start in file name i.e. find idx of '2' in 'USCATE2019...'
        id_date_start = file_name_date2_te_temp.index('2')

        basin_abbr = file_name_date2_te_temp[:id_date_start]
        date2 = file_name_date2_te_temp[id_date_start:id_date_start+8]

        file_name_date2_te_second = file_name_date2_te_temp[id_date_start+8:]
        # Now let's combine into names
        file_path_date1_te = os.path.join(file_path_out_base,
                                '{0}{1}_clipped_to_{2}{3}.tif'
                                .format(basin_abbr, date1, date2,
                                        file_name_date2_te_second))
        file_path_date2_te = os.path.join(file_path_out_base,
                                '{0}{1}_clipped_to_run{2}_snownc.tif'
                                .format(basin_abbr, date2, date1))


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

def return_snow_files(file_path_snownc, year1, year2):
    '''
    short utility to create file paths for snow nc file
    '''
    # Snow.nc file paths
    # Grab modelled snow depth one day prior to both lidar updates
    snownc_date2 = format_date(year2)
    snownc_date2 -= datetime.timedelta(days = 1)
    snownc_date2 = snownc_date2.strftime("%Y%m%d")

    # snow.nc file paths to determine if basin is gaining or losing snow depth overall
    file_path_snownc1 = os.path.join(file_path_snownc, \
                                        'run' + year1, 'snow.nc')
    file_path_snownc2 = os.path.join(file_path_snownc, \
                                        'run' + snownc_date2, 'snow.nc')

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
