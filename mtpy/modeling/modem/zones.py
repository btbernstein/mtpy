"""
This module contains functions for dividing a ModEM model into "zones" -
clusters of cells that have similar resistivities.

It supports heuristics that attempt to identify zones automatically, or
user input for defining zones manually.

CreationDate: 28-04-2020 10:51:34 AEST
Developer: brenainn.moushall@ga.gov.au
"""
import sys
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import ndimage

from mtpy.modeling.modem import Model, Data
from mtpy.utils.mtpylog import MtPyLog
from mtpy.utils import gis_tools, EPSG_DICT
from mtpy.utils.modem_utils import (strip_resgrid, get_centers, strip_padding, get_depth_indices,
                                    get_centers, get_gdal_origin, array2geotiff_writer)


_logger = MtPyLog.get_mtpy_logger(__name__)


def _load_model(model_file):
    model = Model()
    model.read_model_file(model_fn=model_file)
    return model


def _load_data(data_file):
    data = Data()
    data.read_data_file(data_fn=data_file)
    return data


class Zone(object):
    def __init__(self, model_name, membership, identifier, resistivity, depths, mask, top, right,
                 bottom, left, area, epsg, stations, min_depth, max_depth, method, method_range,
                 cell_size):
        self.model_name = model_name
        self.membership = membership
        self.identifier = identifier
        self.res = resistivity
        self.depths = depths
        self.mask = mask
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left
        self.area = area
        self.epsg = epsg
        self.stations = stations
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.method = method
        self.method_range = method_range
        self.cell_size = cell_size

    def write_csv(self):
        pass

    def plot(self, res_scaling=None, min_depth=None, max_depth=None, num_y_ticks=5,
             figsize=(5, 10), savepath=None):
        """Plots the resistivity of a zone against model depth.

        zone_res: np.ndarray
        """
        min_depth = min(self.depths) if min_depth is None else min_depth
        max_depth = max(self.depths) if max_depth is None else max_depth
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylim(max_depth, min_depth)
        step = max_depth // num_y_ticks
        ticks = list(reversed(np.arange(0, max_depth, step)))
        del ticks[-1]
        ticks.insert(0, max_depth)
        ticks.append(min_depth)
        ax.set_yticks(ticks)
        ax.set_ylabel("Depth, m")
        zone_name = ' '.join((self.model_name, self.membership, str(self.identifier)))
        ax.set_title(zone_name)
        zone_mean_res = np.mean(self.res, axis=0)
        zone_min_res = np.min(self.res, axis=0)
        zone_max_res = np.max(self.res, axis=0)
        if res_scaling == 'log':
            zone_mean_res = np.log10(zone_mean_res)
            zone_min_res = np.log10(zone_min_res)
            zone_max_res = np.log10(zone_max_res)
            ax.set_xlabel("Resistivity, " + r"$\Omega$" + "m (log10)")
        else:
            ax.set_xlabel("Resistivity, " + r"$\Omega$" + "m (no scaling)")
        ax.plot(zone_mean_res, self.depths, color='k', lw=1.5)
        ax.plot(zone_min_res, self.depths, color='k', alpha=0.1, ls='--')
        ax.plot(zone_max_res, self.depths, color='k', alpha=0.1, ls='--')
        ax.fill_betweenx(self.depths, zone_min_res, zone_max_res, alpha=0.1, color='k')
        ax.xaxis.set_tick_params(which='minor', bottom=True)
        savepath = os.path.join(savepath, zone_name.replace(' ', '_').replace(':', ''))
        fig.tight_layout()
        fig.savefig(f"{savepath}.png")
        plt.close(fig)


def _meanshift_cluster(res):
    """Uses MeanShift clustering to label cells.

    Parameters
    ----------
    res: np.ndarray
        The resistivity grid of the model with depths limited to
        a specific range.

    Returns
    -------
        A 2D array of shape (res.shape[0], res.shape[1]) where groups of
        distinct values correspond to a cluster.
    """
    # Need to transpose the data into shape (X*Y, 1) for inputting
    # to mean shift clusterer.
    X = res.flatten().reshape(-1, 1)
    # See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    bw = estimate_bandwidth(X, quantile=0.3, random_state=1)
    msc = MeanShift(bw)
    labels = msc.fit_predict(X)
    labels = np.squeeze(labels.reshape(res.shape))
    return labels, {l: f'cluster {l}' for l in np.unique(labels)}


def _magnitude(res, mag_range):
    """Labels cells according to the magnitude of their resistivity.

    Parameters
    ----------
    res: np.ndarray
        The resistivity grid of the model with depths limited to a
        specific range.
    mag_range: int
        See description in `find_zones` docstring.
            
    Returns
    -------
        A 2D array of shape (res.shape[0], res.shape[1]) where groups of
        distinct values correspond to a zone being within a certain
        magnitud range.
    """
    log_res = np.floor(np.log10(res))
    membership_map = {}

    boundary_le = np.min(log_res)
    boundary_re = np.max(log_res)
    boundary = boundary_le
    boundaries = [boundary]
    while boundary < boundary_re:
        boundary = boundary + mag_range
        boundaries.append(boundary)

    result = np.digitize(log_res, boundaries)
    for x in np.unique(result):
        if x == 0:
            membership_map[x] = f'-inf to {10**boundaries[0]}'
        elif x > len(boundaries) - 1:
            membership_map[x] = f'{10**boundaries[-1]} to inf'
        else:
            membership_map[x] = f'{10**boundaries[x - 1]} to {10**boundaries[x]}'

    return result, membership_map


def _value(res, value_ranges):
    """Labels cells according to membership of a value range.

    Parameters
    ----------
    res: np.ndarray
        The resistivity grid of the model with depths limited to a
        specific range.
    value_ranges: int


    Returns
    -------
        A 2D array of shape (res.shape[0], res.shape[1]) where groups of
        distinct values correspond to a zone being within a certain
        value range.
    """
    result = np.digitize(res, value_ranges)
    membership_map = {}
    for x in np.unique(result):
        if x == 0:
            membership_map[x] = f'-inf to {value_ranges[0]}'
        elif x > len(value_ranges) - 1:
            membership_map[x] = f'{value_ranges[-1]} to inf'
        else:
            membership_map[x] = f'{value_ranges[x - 1]} to {value_ranges[x]}'
    return result, membership_map


def _label_zones(labels, l):
    """Takes a 2D-array of labels and applies unique labels to disinct
    zones that have given label l, i.e. it determines contiguous
    subzones within each membership class/label.

    Parameters
    ----------
    labels: np.ndarray
        A 2D array of labels.
    l: any
        The label/class membership to find and divide into subzones.
    """
    labelled, _ = ndimage.label(labels == l)
    return labelled


def find_zones(model_file, data_file=None, x_pad=None, y_pad=None, z_pad=None, depths=None,
               method='cluster', magnitude_range=None, value_ranges=None, contiguous=False,
               write_maps=True, map_outdir=None, write_csv=True):
    """Automatically divides a ModEM resistivity model into zones based
    on areas that have similar resistivity and other search parameters.

    Parameters
    ----------
    model_file: str or path
        Path to the ModEM .rho file.
    data_file: str or path, optional
        Path to the ModEM .dat file. Required if 'write_maps' is True.
        Is needed for getting center point of the survey area so the
        zone maps can be georeferenced.
    x_pad, y_pad, z_pad: int, optional
        Number of padding cells along each axis of the model. These are
        stripped from the model before processing is performed. If
        not provided, then the default values will be taken from the
        model.
    depths: int or tuple or list, optional
        Used to limit the search range when selecting zones. Can be a
        single depth as an integer (in which case the mean won't be
        taken when selecting zones, the depth will be considered as-is)
        or a tuple/list of (min_depth, max_depth) (in which case the
        mean will be taken across this depth range). If None, then
        the mean will be taken along all depths in the model when
        selecting zones.
    method: str, optional
        The method for selecting zones. Valid values are 'cluter',
        'magnitude' and 'value'. Cluster will use a MeanShift model to
        automatically cluster the model into zones. Magnitude will
        divide into zones based on the magnitude of mean resistivity.
        Value will divide into zones based on mean resistivity falling
        within a value range.
    magnitude_range: int, optional
        A single integer specifying a range of orders of magnitude to
        group zones into. E.g. 2 might result in bins of 0.1 - 1,
        100 - 1000. The amount of bins and their boundaries
        depends on the range of the data. Required if 'magnitude' method
        is selected.
    value_ranges: list or tuple, optional
        A monotonically increasing series of values that represent the
        edges of value ranges. [0] is the left-most edge and start of
        the bins and [-1] is the right-most edge and end of the bins.
        Values that fall outside of the bins will be classed as
        '-inf to [0]' and '[-1] to inf' respectively. Required if
        'value' method selected.
    contiguous: bool, optional
        If True, then for each bin/membership class as defined by the
        selection method, subzones will be created of contiguous cells
        that fall within this bin/class. If False, then each
        bin/membership class is considered a single zone, i.e. cells
        do not need to be connected.
        Visual example:
              a a a b b
              a b b c c
              a b c c a
        In the grid above, each letter represents a bin/class. If
        contiguous is True, then a is divided into subzones:
              1 1 1 0 0
              1 0 0 0 0
              1 0 0 0 2
        and the mean resistivity of a1 and a2 are taken and plotted
        separately. If contiguous is False, then a is considered a
        single, disconnected zone:
              1 1 1 0 0
              1 0 0 0 0
              1 0 0 0 1
        and the mean resistivity is taken across all a cells and
        plotted.
    write_maps: bool, optional
        If True, then a geotiff will be output for each membership
        class showing the zones within that class.
    map_outdir: str or path, optional
        Zones maps will be written to this directory. If not provided,
        they will be saved to the working directory.

    Returns
    -------
    dict
        A dictionary mapping [zone_class: [zone_res]]. If contiguous
        is False, then the list of zone arrays will have one member.
        Otherwise, there will be an array for each subzone. The array
        is a masked view of the resistivity of the zone.
    np.ndarray
        The model grid_z. This is returned for convenience of plotting
        zones against model depth.
    """
    if isinstance(depths, tuple) or isinstance(depths, list):
        if depths[1] <= depths[0]:
            raise ValueError("Provided depth range is invalid. Max depth ({}) must be less than "
                             "min depth ({}).".format(depths[1], depths[0]))
        else:
            min_depth = depths[0]
            max_depth = depths[1]
    elif not (depths is None or isinstance(depths, int) or isinstance(depths, float)):
        raise TypeError("Depths must be either a list/tuple containing two element depth range "
                        "(min, max), a single depth as an integer or float, or None to get all "
                        "depths in the model. Provided 'depths' was of type {}".format(type(depths)))
    else:
        min_depth = max_depth = 1

    if method == 'magnitude':
        if magnitude_range is None:
            raise TypeError("magnitude_range must be provided when using 'magnitude' method")
        elif not isinstance(magnitude_range, int):
            raise TypeError("magnitude_range needs to be a single interger defining a range of "
                            "magnitudes to use as zone bins, e.g. 1 = 0 : 1, 1 : 10, ...")
    elif method == 'value':
        if value_ranges is None:
            raise TypeError("value_ranges must be provided when using 'value_ranges' method")
        else:
            value_ranges = sorted(value_ranges)
            prev = -sys.maxsize - 1
            for vr in value_ranges:
                if prev >= vr:
                    raise ValueError("Value range must increase in ascending order (check that "
                                     "you haven't specified the same boundary twice).")
                prev = vr

    if write_maps or write_csv:
        if data_file is None:
            _logger.warning("Can't write maps or csv without data_file. Maps will not be written. Please "
                            "provide model .dat file as data_file to write maps or csv.")
            write_maps = False
            write_csv = False
        else:
            data = _load_data(data_file)

    model = _load_model(model_file)

    x_pad = model.pad_east if x_pad is None else x_pad
    y_pad = model.pad_north if y_pad is None else y_pad
    z_pad = model.pad_z if z_pad is None else z_pad

    # Strip padding cells from the res model and depth grid
    res = strip_resgrid(model.res_model, y_pad, x_pad, z_pad)
    grid_z = strip_padding(get_centers(model.grid_z), z_pad, keep_start=True)

    # Get indices for the provided depth/depth range
    start_ind = get_depth_indices(grid_z, min_depth).pop()
    end_ind = get_depth_indices(grid_z, max_depth).pop()
    inds = range(start_ind, end_ind + 1)

    if not inds:
        raise ValueError("No indices could be found in model depth grid for the depths provided.")
    if len(inds) == 1:
        # Dealing with a single depth - don't take mean
        res_sd = res[:, :, inds]  # res for selected depths
    else:
        # Dealing with multiple depths - take mean across depths
        res_sd = np.mean(res[:, :, inds], axis=2)  # res for selected depths

    if method == 'cluster':
        labels, mm = _meanshift_cluster(res_sd)
        method_range = None
    elif method == 'magnitude':
        labels, mm = _magnitude(res_sd, magnitude_range)
        method_range = magnitude_range
    elif method == 'value':
        labels, mm = _value(res_sd, value_ranges)
        method_range = value_ranges
    else:
        raise ValueError("Selected method not recognised.")

    # Construct zone objects - get some additional information
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    ce = get_centers(strip_padding(model.grid_east, x_pad))
    cn = get_centers(strip_padding(model.grid_north, y_pad))
    grid_e, grid_n = np.meshgrid(ce, cn)

    x_res = model.cell_size_east
    y_res = model.cell_size_north
    pixel_size = x_res * y_res

    center = data.center_point
    epsg_code = gis_tools.get_epsg(center.lat.item(), center.lon.item())
    wkt = EPSG_DICT[epsg_code]
    south_positive = '+south' in wkt

    zones = []
    for l in np.unique(labels):
        zone = np.squeeze(_label_zones(labels, l))
        if contiguous:
            subzones = np.unique(zone)
        else:
            subzones = [1]
        for x in subzones:
            print(x)
            if x == 0:
                continue
            if contiguous:
                zi = zone == x
            else:
                zi = zone != 0
            z_res = res[zi]
            top, right, bottom, left, area = get_zone_dimensions(zi, grid_e, grid_n, pixel_size,
                                                                 south_positive)
            stations = None
            # stations = get_stations_in_zone(zi, grid_e, grid_n, x_res, y_res)
            zones.append(Zone(model_name, mm[l], x, z_res, grid_z, zi,
                              top, right, bottom, left, area, epsg_code,
                              stations, min_depth, max_depth, method, method_range, pixel_size))
    return zones


def write_zones_csv(zones, mm, depths, model, x_pad, y_pad, data, contiguous):
    csv_data = defaultdict(list)
    ce = get_centers(strip_padding(model.grid_east, x_pad))
    cn = get_centers(strip_padding(model.grid_north, y_pad))
    grid_e, grid_n = np.meshgrid(ce, cn)

    x_res = model.cell_size_east
    y_res = model.cell_size_north
    pixel_size = x_res * y_res

    center = data.center_point
    epsg_code = gis_tools.get_epsg(center.lat.item(), center.lon.item())
    wkt = EPSG_DICT[epsg_code]
    south_positive = '+south' in wkt

    for l, z in zones.items():
        if contiguous:
            for x in np.unique(z):
                if x == 0:
                    continue
                zi = z == x
                top, right, bottom, left, area = \
                    get_zone_dimensions(zi, grid_e, grid_n, pixel_size, south_positive)
                get_stations_in_zone(zi, grid_e, grid_n, x_res, y_res, data.mt_dict)
                csv_data['class'].append(mm[l])
                csv_data['id'].append(x)
                csv_data['top'].append(top)
                csv_data['right'].append(right)
                csv_data['bottom'].append(bottom)
                csv_data['left'].append(left)
                csv_data['area'].append(area)
                if isinstance(depths, list) or isinstance(depths, tuple):
                    csv_data['min_depth'].append(min(depths))
                    csv_data['max_depth'].append(max(depths))
                else:
                    csv_data['min_depth'].append(depths)
                    csv_data['max_depth'].append(depths)
        else:
            zi = z != 0
            top, right, bottom, left, area = \
                get_zone_dimensions(zi, grid_e, grid_n, pixel_size, south_positive)
            csv_data['class'].append(l)
            csv_data['id'].append(x)
        df = pd.DataFrame(data=csv_data)
        df.to_csv('/tmp/test.csv')


def get_stations_in_zone(zi, grid_e, grid_n, x_res, y_res, station_dict):
    for station, mt_obj in station_dict.items():
        zge = grid_e[zi]
        zgn = grid_n[zi]
        print(mt_obj.east, mt_obj.north)
        print(mt_obj.lat, mt_obj.lon)
        if mt_obj.east in zge + x_res / 2 or mt_obj.east in zge - x_res / 2 \
                and mt_obj.north in zgn + y_res / 2 or mt_obj.north in zgn - y_res / 2:
            print(f"{station} in zone")


def get_zone_dimensions(zi, grid_e, grid_n, pixel_size, south_positive=False):
    top, right, bottom, left = \
        np.max(grid_n[zi]), np.max(grid_e[zi]), np.min(grid_n[zi]), np.min(grid_e[zi])
    if south_positive:
        tmp = bottom
        bottom = top
        top = tmp
    area = pixel_size * np.count_nonzero(zi)
    return top, right, bottom, left, area


def write_zone_map(zone, membership, model, x_pad, y_pad, data, contiguous, outdir):
    """
    Writes a raster displaying zones.

    Parameters
    ----------
    zone: np.ndarray
        A 2D array containing labels, with each group of unique label
        values representing a zone.
    membership: str
        A string defining the membership class/bin that the zone array
        belongs to.
    model: mtpy.modeling.modem.Model
        ModEM model object.
    x_pad, y_pad: int
        Padding cells along the model east and north dimensions.
        Stripped from the model before determining image origin.
    data: mtpy.modeling.modem.Data
        ModEM data object. Used to get survery center coordinate.
    contiguous: bool
        Whether or not contiguous zones are being considered. See 
        `find_zones` description for more detail.
    outdir: str or path, optional
        The directory to write the map to. If not provided, then maps
        are saved to the working directory.
    """
    ce = get_centers(strip_padding(model.grid_east, x_pad))
    cn = get_centers(strip_padding(model.grid_north, y_pad))
    x_res = model.cell_size_east
    y_res = model.cell_size_north
    center = data.center_point
    origin = get_gdal_origin(ce, x_res, center.east, cn, y_res, center.north)
    epsg_code = gis_tools.get_epsg(center.lat.item(), center.lon.item())
    if outdir is None:
        # save to working directory
        output_file = f"{membership.replace(' ', '_')}_zone_map.tif"
    else:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        output_file = os.path.join(outdir, f"{membership.replace(' ', '_')}_zone_map.tif")
    if not contiguous:
        zone = np.where(zone != 0, 1, 0)
    array2geotiff_writer(output_file, origin, x_res, -y_res, zone, epsg_code=epsg_code, ndv=0)


# Test code
if __name__ == '__main__':
    outdir = os.path.join('/', 'tmp', os.path.splitext(os.path.basename(sys.argv[1]))[0])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    zones = find_zones(sys.argv[1], depths=10, method='value', value_ranges=[10, 100, 10000],
                       contiguous=True, write_maps=True, data_file=sys.argv[2],
                       map_outdir='/tmp/maps')
    for z in zones:
        z.plot(res_scaling='log', min_depth=500, max_depth=10000, num_y_ticks=5,
               figsize=(5, 10), savepath='/tmp/test_plots')

