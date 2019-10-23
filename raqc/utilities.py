import rasterio as rio
import time

def prep_coords(file_path_dataset1, file_path_dataset2, file_path_topo, band):

    tick = time.clock()
    with rio.open(file_path_dataset1) as src:
        d1 = src
        meta = d1.profile
    tock = time.clock()
    print(tock - tick)

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
    d3_left = get_nc_bounds(d3.bounds.left, rez3, 'up')
    d3_bottom = get_nc_bounds(d3.bounds.bottom, rez3, 'up')
    d3_right = get_nc_bounds(d3.bounds.right, rez3, 'down')
    d3_top = get_nc_bounds(d3.bounds.top, rez3, 'down')


    # grab bounds of common/overlapping extent of date1, date2 and topo.nc
    # and prepare function call for gdal to clip to extent and align
    left_max_bound = max(d1.bounds.left, d2.bounds.left, d3_left)
    bottom_max_bound = max(d1.bounds.bottom, d2.bounds.bottom, d3_bottom)
    right_min_bound =  min(d1.bounds.right, d2.bounds.right, d3_right)
    top_min_bound = min(d1.bounds.top, d2.bounds.top, d3_top)

    # ensure nothing after decimal - nice whole number, admittedly a float
    left_max_bound = increment_extents(left_max_bound, rez2, 'up')
    bottom_max_bound = increment_extents(bottom_max_bound, rez2, 'up')
    right_min_bound = increment_extents(right_min_bound, rez2, 'down')
    top_min_bound = increment_extents(top_min_bound, rez2, 'down')

    min_extents = [left_max_bound, bottom_max_bound, right_min_bound, \
                            top_min_bound]

    return topo_rez_same, extents_same, min_extents

def increment_extents(coord, rez, up_down):
    """
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

def get_nc_bounds(extent, rez, down_up):
    print('extent in', extent)
    if down_up == 'down':
        extent = extent - rez / 2
    if down_up == 'up':
        extent = extent + rez / 2
    print('extent returned ', extent)
    return(extent)
