import sys
import os

from scipy.signal import find_peaks, peak_prominences, peak_widths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(station_dir):
    d = np.loadtxt(os.path.join(station_dir, 'interface_depth_hist.txt'))
    depths = np.power(10, d[:, 0])
    res = d[:, 1]
    return depths, res


def peak_info(res, depth, max_depth=500, troughs=False):
    """
    """
    depth_limit = depth <= max_depth
    depth = depth[depth_limit]
    res = res[depth_limit]
    if troughs:
        peaks = find_peaks(-res)[0]
    else:
        peaks = find_peaks(res)[0]
    peak_depths, peak_res = depth[peaks], res[peaks]
    if troughs:
        peak_prom = peak_prominences(-res, peaks)[0]
        peak_width = peak_widths(-res, peaks)[0]
    else:
        peak_prom = peak_prominences(res, peaks)[0]
        peak_width = peak_widths(res, peaks)[0]
    peak_info = tuple(reversed(sorted(zip(peak_depths, peak_res, peak_prom, peak_width), key=lambda x: x[2])))
    return peak_info


def plot(depth, res, peaks, troughs=None, max_depth=500, savepath='/tmp/peaks.png'):
    fig, ax = plt.subplots(figsize=(5, 10))
    xlim = [-10, np.max(res)]
    ylim = [max_depth, 0]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.plot(res, depth, 'b', lw=0.5)

    peak_depths, peak_res, _, _ = zip(*peaks)
    ax.scatter(peak_res, peak_depths, c='b')
    if troughs is not None:
        trough_depths, trough_res, _, _ = zip(*troughs)
        ax.scatter(trough_res, trough_depths, c='r')
    fig.savefig(savepath)


def output_csv(peaks, savepath='/tmp/peaks.csv'):
    depths, res, prom, width = zip(*peaks)
    df = pd.DataFrame({'depth': depths, 'resistivity': res, 'prominence': prom, 'width': width})
    df.to_csv(savepath)

if __name__ == '__main__':
    """
    argv[1]: Path to rjmcmcmt output station dir
    argv[2]: Max depth in metres (default 500)
    """
    file_path = sys.argv[1]
    max_depth = float(sys.argv[2])
    depths, res = load_data(file_path)
    peaks = peak_info(res, depths, max_depth)
    troughs = peak_info(res, depths, max_depth, troughs=True)
    output_csv(peaks)
    output_csv(troughs, savepath='/tmp/troughs.csv')
    plot(depths, res, peaks, troughs)
