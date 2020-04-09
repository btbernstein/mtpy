# -*- coding: utf-8 -*-
"""
script to visualise rjmcmcmt output

"""
import os
import spike_detection as sd

resultsdir = '/home/bren/rjmcmcmt/examples/SouthernThomson/output'
outputdir = '/tmp/test_output'
if not os._exists(outputdir):
    os.mkdir(outputdir)

max_depth = 500

for path in os.listdir(resultsdir):
    station_name = path
    station_dir = os.path.join(resultsdir, station_name)
    depths, res = sd.load_data(station_dir)
    peaks = sd.peak_info(res, depths, max_depth)
    troughs = sd.peak_info(res, depths, max_depth, troughs=True)
    sd.plot(depths, res, peaks, troughs, savepath=os.path.join(outputdir, station_name + '.png'))
    sd.output_csv(peaks, savepath=os.path.join(outputdir, station_name + '_peaks.csv'))
    sd.output_csv(troughs, savepath=os.path.join(outputdir, station_name + '_troughs.csv'))

