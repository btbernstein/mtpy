"""
This module contains functions for dividing a ModEM model into "zones" -
clusters of cells that have similar resistivities.

It supports heuristics that attempt to identify zones automatically, or
user input for defining zones manually.

CreationDate: 28-04-2020 10:51:34 AEST
Developer: brenainn.moushall@ga.gov.au
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.ndimage import label

from mtpy.modeling.modem import Model
from mtpy.utils.mtpylog import MtPyLog
from mtpy.utils.modem_utils import (strip_resgrid, get_centers, strip_padding, get_depth_indices,
                                    get_centers)


_logger = MtPyLog.get_mtpy_logger(__name__)


def _load_model(model_file):
    model = Model()
    model.read_model_file(model_fn=model_file)
    return model


def find_zones(model_file, x_pad=None, y_pad=None, z_pad=None,
               depths=None):
    if isinstance(depths, tuple) or isinstance(depths, list):
        if depths[1] <= depths[0]:
            raise ValueError("Provided depth range is invalid. Max depth ({}) must be less than "
                             "min depth ({}).".format(depths[1], depths[0]))
    elif not (depths is None or isinstance(depths, int) or isinstance(depths, float)):
        raise TypeError("Depths must be either a list/tuple containing two element depth range "
                        "(min, max), a single depth as an integer or float, or None to get all "
                        "depths in the model. Provided 'depths' was of type {}".format(type(depths)))

    model = _load_model(model_file)

    # Get paddings if not provided
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

    # All depths, depth range, single depth, polygons containing cells
    # TODO: handle finding zones by averaging across multiple depths
    # TODO: region finding methods - by magnitudes, by user defined ranges
    # TODO: plot found zones
    if len(inds) == 1:
        # Dealing with a single depth - don't take mean
        # Log10 method - regions grouped by magnitude
        # Clustering method
        res_sd = res[:, :, inds]
        # Need to transpose the data into shape (X*Y, 1) for inputting
        # to mean shift clusterer.
        X = res_sd.flatten().reshape(-1, 1)
        # See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
        bw = estimate_bandwidth(X, quantile=0.3, random_state=1)
        msc = MeanShift(bw)
        labels = msc.fit_predict(X)
        labels = np.squeeze(labels.reshape(res_sd.shape))

    # Find contiguous groups of labels - these are the zones
    groups = _contiguous_element_grouper(labels)
    print(f"Unique regions found: {len(np.unique(groups))}")

    # Get X, Y indices for each region
    zones = []
    for zone_id in np.unique(groups):
        zones.append(groups == zone_id)

    # Get average resistivity for zones
    zone_res = []
    for z in zones:
        z_res = res[z]
        z_res_mean = np.mean(z_res, axis=0)
        zone_res.append(z_res_mean)
    return zone_res, grid_z


def _contiguous_element_grouper(labels):
    """
    Takes a 2D-array of labels and identifies contiguous groups of the
    same element.

    Credit to https://stackoverflow.com/questions/47523071/\
        grouping-adjacent-equal-elements-of-a-2d-numpy-array
    """
    flat_labels = labels.ravel()
    offset = 0
    result = np.zeros_like(labels)
    for l in flat_labels:
        labelled, num_features = label(labels == l)
        result += labelled + offset * (labelled > 0)
        offset += 1
    return result


def plot_zone(zone_mean_res, model_depths, zone_name, min_depth=None, max_depth=None,
              res_scaling=None):
    min_depth = min(model_depths) if min_depth is None else min_depth
    max_depth = max(model_depths) if max_depth is None else max_depth
    fig, ax = plt.subplots()
    ax.set_ylim(max_depth, min_depth)
    ax.set_ylabel("Depth (m)")
    ax.set_title(zone_name)
    if res_scaling == 'log':
        zone_mean_res = np.log10(zone_mean_res)
        ax.set_xlabel("Resistivity (log10)")
    else:
        ax.set_xlabel("Resistivity (no scaling)")
    ax.plot(zone_mean_res, model_depths)
    fig.savefig(f'/tmp/{zone_name}.png')
    fig.close()


if __name__ == '__main__':
    zr, depths = find_zones(sys.argv[1], depths=500)
    for zone_id, zone in enumerate(zr):
        plot_zone(zone, depths, f'Zone {zone_id}', 10, 10000, 'log')

