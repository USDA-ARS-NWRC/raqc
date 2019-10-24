import rasterio as rio
import time
import numpy as np

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

    return topo_rez_same, extents_same, min_extents

def evenly_divisible_extents(coord, rez):
    """
    This utility rectifies coordinates in geographically referenced datasets for
    simpler use when identical extents and spatial resolutions are required.  In
    RAQC multiple geotiffs and/or netCDF files are aligned with each other.  This
    utility rounds dataset extents of each file to coordinates evenly divisible
    by the spatial resolution.  The spatial resolution will also be rounded to
    nearest integer.

    Arguments
    coord: coordinate (int or float)
    rez:    spatial resolution i.e. 50m
    Returns
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
    print('coord, ', coord)
    print('coord_updated, ', coord_updated)
    return(coord_updated)

def bounds_to_pix_coords(bound, rez, location):
    """
    Rectifies possible quarks using rasterio, namely that <dataset>.bounds
    outputs exterior bounding box of dataset, NOT the pixel locations.
    May need more investigation, but works when matching netCDF to geotiff
    coordinates using Rasterio

    Arguments
    bound:      bounding box extents from rasterio dataset of .nc file
    rez:        spatial resolution
    location:   left, right, top, or bottom bounding coordinates

    Returns:
    bound_new:  correct pixel coordinates for bounds
    """

    print('bound in', bound)
    # dictionary converts key into string of 'out' or 'in' for shifting bounds
    # in or out
    dict_offset_coords = {'left':'out', 'right':'in', 'bottom':'out', 'top':'in'}
    location = dict_offset_coords[location]
    if location == 'in':
        bound_new = bound - rez / 2
    if location == 'out':
        bound_new = bound + rez / 2
    print('bound returned ', bound_new)
    return(bound_new)

def rasterio_netCDF(file_path):
    """
    A quick standalone demonstration on what rasterio vs. netCDF4 provides in
    terms of bounds.
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
    print('bin edges ', elevation_edges)
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
