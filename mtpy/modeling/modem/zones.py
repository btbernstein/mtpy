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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import ndimage

from mtpy.modeling.modem import Model, Data
from mtpy.utils.mtpylog import MtPyLog
from mtpy.utils import gis_tools
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


def _meanshift_cluster(res):
    """
    Using MeanShift clustering to label cells
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
    """
    Labels cells according to the magnitude of their resistivity
    """
    result = np.log10(res) // mag_range
    membership_map = {}

    boundary_edge_power = np.unique(result)[0]
    for x in np.unique(result):
        membership_map[x] = f'{10**boundary_edge_power} to {10**(x+mag_range)}'
        boundary_edge_power = x + mag_range
    return result, membership_map


def _value(res, value_ranges):
    """
    Labels cells according to membership of a value range
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
    """
    Takes a 2D-array of labels and labels zones that have given
    label l.
    """
    labelled, _ = ndimage.label(labels == l)
    return labelled


def find_zones(model_file, data_file=None, x_pad=None, y_pad=None, z_pad=None, depths=None,
               method='cluster', magnitude_range=None, value_ranges=None, contiguous=False,
               write_maps=True, map_outdir=None):
    if isinstance(depths, tuple) or isinstance(depths, list):
        if depths[1] <= depths[0]:
            raise ValueError("Provided depth range is invalid. Max depth ({}) must be less than "
                             "min depth ({}).".format(depths[1], depths[0]))
    elif not (depths is None or isinstance(depths, int) or isinstance(depths, float)):
        raise TypeError("Depths must be either a list/tuple containing two element depth range "
                        "(min, max), a single depth as an integer or float, or None to get all "
                        "depths in the model. Provided 'depths' was of type {}".format(type(depths)))

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

    if write_maps:
        if data_file is None:
            _logger.warning("Can't write maps without data_file. Maps will not be written. Please "
                            "provide model .dat file as data_file to write maps.")
            write_maps = False
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
    if isinstance(depths, tuple) or isinstance(depths, list):
        start_ind = get_depth_indices(grid_z, [depths[0]]).pop()
        end_ind = get_depth_indices(grid_z, [depths[1]]).pop()
        inds = range(start_ind, end_ind + 1)
    else:
        inds = list(get_depth_indices(grid_z, depths))

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
    elif method == 'magnitude':
        labels, mm = _magnitude(res_sd, magnitude_range)
    elif method == 'value':
        labels, mm = _value(res_sd, value_ranges)

    zones = {}
    for l in np.unique(labels):
        zone = np.squeeze(_label_zones(labels, l))
        zones[l] = zone
        if write_maps:
            write_zone_map(zone, mm[l], model, x_pad, y_pad, data, contiguous, map_outdir)

    zone_res = defaultdict(list)
    if contiguous:
        # Treat each contiguous group as a discrete zone
        for l, z in zones.items():
            for x in np.unique(z):
                z_res = res[z == x]
                zone_res[mm[l]].append(z_res)
    else:
        for l, z in zones.items():
            z_res = res[z != 0]
            zone_res[mm[l]].append(z_res)

    return zone_res, grid_z


def write_zone_map(zone, membership, model, x_pad, y_pad, data, contiguous, outdir):
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


def plot_zone_res(zone_res, model_depths, zone_name, outdir, min_depth=None, max_depth=None,
                  res_scaling=None, num_y_ticks=5, figsize=(5, 10)):
    min_depth = min(model_depths) if min_depth is None else min_depth
    max_depth = max(model_depths) if max_depth is None else max_depth
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim(max_depth, min_depth)
    step = max_depth // num_y_ticks
    ticks = list(reversed(np.arange(0, max_depth, step)))
    ticks.insert(0, min_depth)
    ticks.append(max_depth)
    ax.set_yticks(ticks)
    ax.set_ylabel("Depth, m")
    ax.set_title(zone_name)
    zone_mean_res = np.mean(zone_res, axis=0)
    zone_min_res = np.min(zone_res, axis=0)
    zone_max_res = np.max(zone_res, axis=0)
    if res_scaling == 'log':
        zone_mean_res = np.log10(zone_mean_res)
        zone_min_res = np.log10(zone_min_res)
        zone_max_res = np.log10(zone_max_res)
        ax.set_xlabel("Resistivity, " + r"$\Omega$" + "m (log10)")
    else:
        ax.set_xlabel("Resistivity, " + r"$\Omega$" + "m (no scaling)")
    ax.plot(zone_mean_res, model_depths, color='k', lw=1.5)
    ax.plot(zone_min_res, model_depths, color='k', alpha=0.1, ls='--')
    ax.plot(zone_max_res, model_depths, color='k', alpha=0.1, ls='--')
    ax.fill_betweenx(model_depths, zone_min_res, zone_max_res, alpha=0.1, color='k')
    ax.xaxis.set_tick_params(which='minor', bottom=True)
    savepath = os.path.join(outdir, zone_name.replace(' ', '_').replace(':', ''))
    fig.savefig(f"{savepath}.png")
    plt.close(fig)


# Test code
if __name__ == '__main__':
    outdir = os.path.join('/', 'tmp', os.path.splitext(os.path.basename(sys.argv[1]))[0])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    zones, depths = find_zones(sys.argv[1], depths=10, method='magnitude', magnitude_range=2,
                               contiguous=True, write_maps=True, data_file=sys.argv[2],
                               map_outdir='/tmp/maps')
    for l, z in zones.items():
        for i, zz in enumerate(z):
            plot_zone_res(zz, depths, f'{l}: Zone {i+1}', outdir, min_depth=10, max_depth=10000,
                          res_scaling='log')

