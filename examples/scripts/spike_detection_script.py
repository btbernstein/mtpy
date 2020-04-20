# -*- coding: utf-8 -*-
"""
script to visualise rjmcmcmt output

"""
import os
import spike_detection as sd

from matplotlib.backends.backend_pdf import PdfPages

resultsdir = '/home/bren/rjmcmcmt/examples/SouthernThomson/output'
outputdir = '/tmp/test_output'
if not os.path.exists(outputdir):
    os.mkdir(outputdir)

# Parameters #

# Depth range to search for peaks/troughs
min_depth = 100
max_depth = 500

# The minimum prominence of the peak/trough to be included in results
# For an explanation of how prominence is calculated:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy-signal-peak-prominences
min_prominence = 100000

# End parameters #

station_data = []
figures = []
for path in os.listdir(resultsdir):
    station_name = path
    station_dir = os.path.join(resultsdir, station_name)
    depths, res = sd.load_data(station_dir)
    peaks = sd.peak_info(res, depths, min_prominence, min_depth, max_depth)
    troughs = sd.peak_info(res, depths, min_prominence, min_depth, max_depth, troughs=True)
    figures.append(sd.plot(station_name, depths, res, peaks, troughs, min_depth, max_depth, 
                   savepath=os.path.join(outputdir, station_name + '.png')))
    sd.output_csv(peaks, savepath=os.path.join(outputdir, station_name + '_peaks.csv'))
    sd.output_csv(troughs, savepath=os.path.join(outputdir, station_name + '_troughs.csv'))
    for p in peaks:
        print(f"Potential layer change in {station_name} at depth {p[0]}m with res {p[1]}")
    for t in troughs:
        print(f"Potential layer change in {station_name} at depth {t[0]}m with res {t[1]}")
    station_data.append((station_name, depths, res, peaks, troughs))

figures.append(sd.all_plot(station_data, min_depth, max_depth, 
                           savepath=os.path.join(outputdir, 'all_peaks.png')))

pdf = PdfPages(os.path.join(outputdir, 'all_plots.pdf'))
for fig in figures:
    pdf.savefig(fig)
pdf.close()

print(f"Complete! Outputs are located in '{outputdir}'")
