# -*- coding: utf-8 -*-
"""
24-06-2020 12:50:28 AEST
brenainn.moushall@ga.gov.au

This is the same as ModEM_build_inputfiles.py but focuses on explaining
the parameters used to customise initial resisitvity and depth layer
growth for models. Each call to `Model` shows a different way of
setting resisitvity/depth.

For explanations of everything else going on, see ModEM_build_inputfiles.py.
"""
import os
# os.chdir(r'C:\mtpywin\mtpy') # change this path to the path where mtpy is installed
import os.path as op
from mtpy.modeling.modem import Model
from mtpy.modeling.modem import Data
from mtpy.modeling.modem import Covariance
from mtpy.core.edi import Edi
from mtpy.utils.calculator import get_period_list
from mtpy.utils.modem_utils import shapefile_to_geoseries

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

workdir = '/path/to/output'
edipath = '/path/to/edifiles'

start_period = 0.002
stop_period = 2000
periods_per_decade = 4
period_list = get_period_list(start_period, stop_period, periods_per_decade,
                              include_outside_range=True)
edi_list = [op.join(edipath, ff) for ff in os.listdir(edipath) if (ff.endswith('.edi'))]
if not op.exists(workdir):
    os.mkdir(workdir)

do = Data(edi_list=edi_list,
          inv_mode='1',
          save_path=workdir,
          period_list=period_list,
          period_buffer=2,
          error_type_z=np.array([['floor_percent', 'floor_egbert'],
                                 ['floor_egbert', 'percent']]),
          error_value_z=np.array([[20., 5.],
                                  [5., 20.]]),
          error_type_tipper='floor_abs',
          error_value_tipper=.03,
          model_epsg=28354
          )
do.write_data_file()

do.data_array['elev'] = 0.
do.write_data_file(fill=False)


# Setting initial resisitvity using a constant value:

# This will set every cell in the res model to the same value.

# Parameters
res_value = 100  # Sets every cell in the model to an initial res of 100 ohm/meter

mo = Model(station_locations=do.station_locations,
           cell_size_east=8000,
           cell_size_north=8000,
           pad_north=7,
           pad_east=7,
           pad_z=6,
           pad_stretch_v=1.6,
           pad_stretch_h=1.4,
           pad_num=3,
           pad_method='stretch',

           # RESISTIVITY ARGUMENTS
           res_initial_value=res_value,  # The value to be used when applying a constant initial res
           res_initial_method='constant',  # Informs that we want to set a constant value for res model

           n_layers=100,
           n_air_layers=10,
           z1_layer=10,
           z_target_depth=120000,
           z_mesh_method='new',
           z_layer_rounding=0
           )

mo.make_mesh()
mo.write_model_file()

# Setting initial resisitivty using a set of depth ranges and values:

# For each specified depth range, a res value is provided. This depth
# range in the model will be set to this res. Parts of the model not
# covered by the depth range will be set to a constant value.

# In this case, the range from 0m to 1000m will be set to 200 ohm/meter,
# and the range from 5000m to 10000m will be set to 500 ohm/meter. All
# cells outside of these depth ranges will be set to 100 ohm/meter.
custom_depth_ranges = [(0, 1000), (5000, 10000)]
custom_depth_values = [200, 500]
constant_res_value = 100

mo = Model(station_locations=do.station_locations,
           cell_size_east=8000,
           cell_size_north=8000,
           pad_north=7,
           pad_east=7,
           pad_z=6,
           pad_stretch_v=1.6,
           pad_stretch_h=1.4,
           pad_num=3,
           pad_method='stretch',

           # RESISTIVITY ARGUMENTS
           res_initial_value=constant_res_value,
           res_initial_method='by_range',  # Informs that we're setting values based on depth ranges
           res_initial_custom_depth_ranges=custom_depth_ranges,
           res_initial_custom_values=custom_depth_values,

           n_layers=100,
           n_air_layers=10,
           z1_layer=10,
           z_target_depth=120000,
           z_mesh_method='new',
           z_layer_rounding=0
           )

mo.make_mesh()
mo.write_model_file(save_path=workdir)

# Setting initial resistivity by defining zones using polygons:

# We can provide a series of 2d polygons (X, Y dimensions) that divide
# the model into zones.
# For each zone we can then set the resistivity for certain depth ranges
# as with the method above. Also as with the method above, any parts
# of the model that are unspecified will be set to a constant value.

# See examples/scripts/ModEM_PlotZones for more information on zones.

# 'polygons' must be a geopandas geoseries with the index being the
# name of each zone/area.

# A shapefile can used by providing it to the `shapefile_to_geoseries` function.
# The shapes will be reprojected to the CRS of the model during processing.
polygons = shapefile_to_geoseries('/home/bren/data_mtpy/all_aus.shp',
                                  name_field='name')

# Alternatively, the shapes can be specified here in the script.
poly_data = [Polygon([[556895, 7821291], [576972, 7806164], [553319, 7805064]]),
             Polygon([[627445, 7867099], [619011, 7839077], [611937, 7857033]])]
# Must provide the CRS of the polygons if providing them directly.
poly_epsg = 'epsg:32753'
poly_names = ['area1', 'area2']
polygons = gpd.GeoSeries(poly_data, poly_names, crs=poly_epsg)

# In this case, for polygon 'area1', the depth range 0m to 1000m will
# be set to 200 ohm/meter and the range 5000m to 10000m will be set to
# 300 ohm/meter. 'area2' 0m to 1000m will be set to 500 ohm/meter and
# 3000m to 15000m will be set to 1000 ohm/meter.
custom_depth_ranges = {'area1': [(0, 1000), (5000, 10000)],
                       'area2': [(0, 1000), (3000, 15000)]}
custom_depth_values = {'area1': [200, 300],
                       'area2': [500, 1000]}
# Areas not covered by the polygons + depth ranges will be set to 100
# ohm/meter.
constant_res_value = 100

mo = Model(station_locations=do.station_locations,
           cell_size_east=8000,
           cell_size_north=8000,
           pad_north=7,
           pad_east=7,
           pad_z=6,
           pad_stretch_v=1.6,
           pad_stretch_h=1.4,
           pad_num=3,
           pad_method='stretch',

           # RESISTIVITY ARGUMENTS
           res_initial_value=constant_res_value,
           res_initial_method='by_polygons',  # Setting value based on polygons
           res_initial_polygons=polygons,
           res_initial_custom_depth_ranges=custom_depth_ranges,
           res_initial_custom_values=custom_depth_values,
           # Need to also pass in the data object if the polygon method is used
           data_object=do,

           n_layers=100,
           n_air_layers=10,
           z1_layer=10,
           z_target_depth=120000,
           z_mesh_method='new',
           z_layer_rounding=0
           )

# Custom depth layer growth:

# This requires a fair bit of explanation.

# By default, depth layers are grown by providing a z1_layer thickness
# (depth of the first layer), the number of layers to create (N) and the
# target depth to create layers to. The layers are then generated by
# creating N log spaced layers up to the target depth.

# The custom option allows growing depth layers using a combination of
# factor increases, log spaced increases and constant increase.

# We need to provide a list of depth boundaries (rather than pairs of
# ranges). The first depth is always z1_thickness, and the last is
# z_target_depth. So if z1_layer = 10 and z_target_depth = 120000 then
# providing boundaries [100, 5000] will create a series of ranges
# from 10m to 100m, 100m to 5000m and 5000m to 120000m.

# The layers within each of these intervals are then generated by
# a series of function parameters.

# In this case, there are three depth intervals (as explained above).

# The first interval (10m to 100m) will contain 2 log spaced layers.
# (2, 'l') means create 2 log spaced layers.

# The second interval (100m to 500m) will create layers spaced at 100m.
# (100, 'c') means create layers at 100m constant spacing, starting at
# 100m and ending when 5000m (or greater) is reached.

# The third interval (5000m to 120000m) will contain layers grown at
# an increase of 1.5. First layer is 5000m (or wherever the previous
# growth function ended), second is (first layer * 1.5), third is
# (second layer * 1.5) and so on until 120000m or greater is reached.
# (1.5, 'f') means create layers with the spacing increasing by 1.5
# for each layer.

custom_depth_boundaries = [100, 5000]
custom_factors = [(2, 'l'), (100, 'c'), (1.5, 'f')]

mo = Model(station_locations=do.station_locations,
           cell_size_east=8000,
           cell_size_north=8000,
           pad_north=7,
           pad_east=7,
           pad_z=6,
           pad_stretch_v=1.6,
           pad_stretch_h=1.4,
           pad_num=3,
           pad_method='stretch',

           res_initial_value=res_value,
           res_initial_method='constant',

           # n_layers=100, # n_layers is ignored when using custom depth growth
           n_air_layers=10,
           z1_layer=10,  # First layer thickness and start of our depth intervals
           z_target_depth=120000,  # Target depth and end of our depth intervals
           z_mesh_method='by_range',  # Enables custom depth growth
           z_custom_factors=custom_factors,
           z_custom_depth_ranges=custom_depth_boundaries,
           z_layer_rounding=0
           )
