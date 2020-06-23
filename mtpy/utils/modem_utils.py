import math
import os

import numpy as np
import geopandas as gpd
import gdal
import osr

import mtpy.modeling.modem
from mtpy.utils.mtpylog import MtPyLog

_logger = MtPyLog.get_mtpy_logger(__name__)


def shapefile_to_geoseries(path, name_field='id'):
    """Loads a shapefile as a GeoDataFrame and converts it to a 
    GeoSeries. This is used for defining ModEM model zones by polygon.

    Args:
        path (str or bytes): Full path to the shapefile to load.
        name_field (str): Name of the field containing the name of
            each poylgon. Set as the 'index' field on the GeoSeries.

    Returns:
        GeoSeries: Shapefile shapes as a GeoSeries with index set as
            'name_field' field.
    """
    df = gpd.read_file(path)
    return gpd.GeoSeries(list(df.geometry), df[name_field], crs=df.crs)


def rotate_transform(gt, angle, pivot_east, pivot_north):
    """Rotates a geotransform.

    Args:
        gt (tuple of float): A 6-tuple GDAL style geotransfrom
            (upperleft X, pixel width, row rotation,
             upperleft Y, column rotation, pixel height)
        angle (float): Angle in degrees to rotate by.
        pivot_east, pivot_north (float): The pivot point of rotation

    Returns:
        tuple of float: A rotated geotransform.
    """
    ox, pw, rrot, oy, crot, ph = gt
    rot = math.radians(angle)
    gt[0] = pivot_east + (ox - pivot_east) * math.cos(rot) \
        + (oy - pivot_north) * math.sin(rot)
    gt[1] = pw * math.cos(rot)
    gt[2] = pw * -math.sin(rot)
    gt[3] = pivot_north - (ox - pivot_east) * math.sin(rot) \
        + (oy - pivot_north) * math.cos(rot)

    gt[4] = ph * math.sin(rot)
    gt[5] = ph * math.cos(rot)
    return gt


def array2geotiff_writer(filename, origin, pixel_width, pixel_height, data,
                         angle=None, epsg_code=4283, center=None, rotate_origin=False,
                         ndv=np.nan):
    """Writes a 2D array as a single band geotiff raster and ASCII grid.

    Args:
        filename (str): Name of tiff/asc grid. If rotated, this will be
            appended to the filename as 'name_rot{degrees}.tif'.
        origin (tuple of int): Upper-left origin of image as (X, Y).
        pixel_wdith, pixel_height (float): Pixel resolutions in X and Y
            dimensions respectively.
        data (np.ndarray): The array to be written as an image.
        angle (float): Angle in degrees to rotate by. If None or 0 is
            given, no rotation will be performed.
        epsg_code (int): The 4-digit EPSG code of the data CRS.
        center (tuple): A tuple containing image center point as
            (easting, northing).
        rotate_origin (bool): If True, image will be rotated about the
            upper-left origin. If False, will be rotated about center.
            The `center` param must be provided if rotating about
            center.

    Returns:
        str, str: Filename of the geotiff ([0]) and ASCII grid ([1]).
    """
    gt = [origin[0], pixel_width, 0, origin[1], 0, pixel_height]

    # Apply rotation by tweaking geotransform. The data remains the
    # same but will appear roated in a viewer e.g. ArcGIS.
    if angle:
        if rotate_origin:
            gt = rotate_transform(gt, angle, origin[0], origin[1])
        else:
            gt = rotate_transform(gt, angle, center.east, center.north)
        filename = '{}_rot{}.tif'\
                   .format(os.path.splitext(filename)[0], angle)

    rows, cols = data.shape
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
    out_raster.SetGeoTransform(gt)
    out_band = out_raster.GetRasterBand(1)
    out_band.SetNoDataValue(ndv)
    out_band.WriteArray(data)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    out_raster.SetProjection(srs.ExportToWkt())
    out_band.FlushCache()

    # output to ascii format
    ascii_filename = "{}.asc".format(os.path.splitext(filename)[0])
    driver2 = gdal.GetDriverByName('AAIGrid')
    driver2.CreateCopy(ascii_filename, out_raster)

    return filename, ascii_filename


def get_gdal_origin(centers_east, east_cell_size, mesh_center_east,
                    centers_north, north_cell_size, mesh_center_north,
                    south_positive=False):
    """Works out the upper left X, Y points of a grid.

    Args:
        centers_east, centers_north (np.ndarray): Arrays of cell
            centers along respective axes.
        cell_size_east, cell_size_north (float): Cell sizes in
            respective directions.
        mesh_center_east, mesh_center_north (float): Center point
            of the survey area in some CRS system.
        south_positive: True if a '+south' CRS, False otherwise.

    Return:
        float, float: The upper left coordinate of the image in
            relation to the survey center point. Used as GDAL origin.
    """
    north_corner = centers_north[0] if south_positive else centers_north[-1]
    return (centers_east[0] + mesh_center_east - east_cell_size / 2,
            north_corner + mesh_center_north + north_cell_size / 2)


def get_centers(arr):
    """Get the centers from an array of cell boundaries.

    Args:
        arr (np.ndarray): An array of cell boundaries.

    Returns:
        np.ndarray: An array of cell centers.
    """
    return np.mean([arr[:-1], arr[1:]], axis=0)


def strip_padding(arr, pad, keep_start=False):
    """Strip padding cells from grid data. Padding cells occur at the
    the start and end of north and east grid axes, and at the end of
    grid depth axis.

    Note we handle the special case of `pad=0` by returning the data
    untouched (a `slice(None)` will do nothing when used as an index
    slice).

    Args:
        arr (np.ndarray): Axis of grid data (east, north or depth).
        pad (int): Number of padding cells being stripped.
        keep_start (bool): If True, padding is only stripped from the
            end of the data. Intended for use with depth axis.

    Return:
        np.ndarray: A copy of `arr` with padding cells removed.
    """
    if keep_start:
        pad = slice(None) if pad == 0 else slice(None, -pad)
    else:
        pad = slice(None) if pad == 0 else slice(pad, -pad)

    return arr[pad]


def strip_resgrid(res_model, y_pad, x_pad, z_pad):
    """Similar to `strip_padding` but handles the case of stripping
    padding from a 3D resistivity model.

    """
    y_pad = slice(None) if y_pad == 0 else slice(y_pad, -y_pad)
    x_pad = slice(None) if x_pad == 0 else slice(x_pad, -x_pad)
    z_pad = slice(None) if z_pad == 0 else slice(None, -z_pad)
    return res_model[y_pad, x_pad, z_pad]


def list_depths(model, zpad=None):
    """
    Return a list of available depth slices in the model.

    Args:
        model_file (str): Path to ModEM .rho file.
        zpad (int, optional): Number of padding slices to remove from
            bottom of model. If None, model pad_z value is used.

    Returns:
        list of float: A list of available depth slices.
    """
    # Try to insantiate from model file
    if not isinstance(model, mtpy.modeling.modem.Model):
        model = mtpy.modeling.modem.Model()
        model.read_model_file(model_fn=model)

    cz = get_centers(model.grid_z)
    zpad = model.pad_z if zpad is None else zpad
    return cz[:-zpad]


def get_depth_indices(centers_z, depths):
    """Finds the indices for the elements closest to those specified
    in a given list of depths.

    Args:
        centers_z (np.ndarray): Grid centers along the Z axis (i.e.
            available depths in our model).
        depths (list of int), int, None: A list of depths,
            or single depth, to retrieve indices for. Providing 'None'
            will return all indices.

    Returns:
        set: A set of indices closest to provided depths.
    """
    def _nearest(array, value):
        """Get index for nearest element to value in an array.
        """
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array)
                or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx

    if depths:
        if isinstance(depths, list) or isinstance(depths, tuple):
            res = {_nearest(centers_z, d) for d in depths}
        else:
            res = {_nearest(centers_z, depths)}
        _logger.info("Slices closest to requested depths: {}".format([centers_z[di] for di in res]))
        return res
    else:
        _logger.info("No depths provided, getting all slices...")
        return set(range(len(centers_z)))


