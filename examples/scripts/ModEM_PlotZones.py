# -*- coding: utf-8 -*-
"""
Provides several different methods for dividing a ModEM model into 
zones based on mean resistivity, and then plots the mean resistivity
for each zone against depth.

Creation: 30-04-2020 16:31:58 AEST
Developer: brenainn.moushall@ga.gov.au
"""
import os

# os.chdir(r'C:\mtpywin\mtpy')

import os.path as op

import numpy as np

from mtpy.modeling.modem.zones import find_zones, plot_zone_res


model_file = '/path/to/model_file.rho'
outdir = '/path/to/outdir/for/plots'
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Parameters and explanations

# depths: this is the range of depths in the model to be considered
# when looking for zones. The mean resistivity across these depth slices
# will be used to categorise zones based on the chosen selection method.
# Accepts a range as a 2 element tuple, a single value for a single
# depth slice, or None for all available depth slices.
depths = (100, 500)

# method: the method for categorising zones. There are three available

#   'magnitude': categorise zones based on the magnitude of their mean
#   resistivity. This requires the 'magnitude_range' parameter. The
#   magnitude range specifies a range of magnitudes to use as bins.

#   E.g., magnitude_range=2 could have bins 0.1: 10, 10: 1000, 1000: 100000
#   The number of bins will depend on the range of the data, but each
#   bin will represent an interval of magntiude_range orders of magnitude.

method = 'magnitude'
magntiude_range = 2

#   'value': categorise zones based on the range that their mean
#   resistivity falls into. This requires the 'value_ranges' parameter.
#   The value ranges are a list of integers specifying the left edge
#   of bin boundaries.

#   E.g. value_ranges=[0, 100, 1000] will create 4 bins of -inf: 0,
#   0: 100, 100: 1000, 1000: inf

# method = 'value'
# value_ranges = [1, 100, 1000]

#   'cluster': use scikit learn's MeanShift clustering algorithm to
#   automatically cluster zones based on average resistivities. This
#   is experimental. Also note it is much slower than the other methods.
#   This method doesn't require any additional parameters.

# method = 'cluster'

# contiguous: whether cells have to be contiguous to be treated as a
# zone. If True, then for each bin/membership class as defined by the
# selection method, subzones will be created of contiguous cells that
# fall within this bin/class.
# If False, then each bin/membership class is considered a single zone,
# i.e. cells do not need to be connected.

# Visual example:
#       a a a b b
#       a b b c c
#       a b c c a
# In the grid above, each letter represents a bin/class. If contiguous
# is True, then a is divided into subzones:
#       1 1 1 0 0
#       1 0 0 0 0
#       1 0 0 0 2
# and the mean resistivity of a1 and a2 are taken and plotted separately.
# If contiguous is False, then a is considered a single, disconnected zone:
#       1 1 1 0 0
#       1 0 0 0 0
#       1 0 0 0 1
# and the mean resistivity is taken across all a cells and plotted.

contiguous = True

# Number of padding cells for the ModEM model. These will be stripped
# before any processing is performed. If left as None, the padding cells
# will be read from the model file.

x_pad = None
y_pad = None
z_pad = None

# 'zones' is a dictionary of {class: [zones]}
zones, model_depths = find_zones(model_file, x_pad=x_pad, y_pad=y_pad, z_pad=z_pad,
                                 depths=depths,
                                 method=method, magnitude_range=magntiude_range,
                                 contiguous=contiguous)

for membership_class, zone in zones.items():
    # If contiguous is False, there will be only one '0' subzone comprising all cells in that class
    # min_depth, max_depth: depth range to plot
    # res_scaling: 'log' for log10, None for no scaling
    for subzone_index, subzone in enumerate(zone):
        plot_zone_res(subzone, model_depths, f'{membership_class}: zone {subzone_index}',
                      outdir, min_depth=10, max_depth=10000, res_scaling='log')
