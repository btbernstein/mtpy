import sys
import os

from scipy.signal import find_peaks, peak_prominences, peak_widths
from matplotlib.pyplot import cm
from matplotlib.axes._axes import _log as _mpl_logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_mpl_logger.setLevel('ERROR')

def load_data(station_dir):
    d = np.loadtxt(os.path.join(station_dir, 'interface_depth_hist.txt'))
    depths = np.power(10, d[:, 0])
    res = d[:, 1]
    return depths, res


def peak_info(res, depth, min_prominence=0, min_depth=0, max_depth=500, troughs=False):
    """
    """
    depth_limit = np.logical_and(min_depth <= depth, depth <= max_depth)
    depth = depth[depth_limit]
    res = res[depth_limit]
    if troughs:
        peaks = find_peaks(-res, prominence=min_prominence)[0]
    else:
        peaks = find_peaks(res, prominence=min_prominence)[0]

    if len(peaks) == 0:
        return (())

    peak_depths, peak_res = depth[peaks], res[peaks]

    if troughs:
        peak_prom = peak_prominences(-res, peaks)[0]
        peak_width = peak_widths(-res, peaks)[0]
    else:
        peak_prom = peak_prominences(res, peaks)[0]
        peak_width = peak_widths(res, peaks)[0]
    peak_info = tuple(reversed(sorted(zip(peak_depths, peak_res, peak_prom, peak_width), 
                               key=lambda x: x[2])))
    return peak_info


def plot(station_name, depth, res, peaks, troughs=None, min_depth=0, max_depth=500,
         savepath='/tmp/peaks.png'):
    fig, ax = plt.subplots(figsize=(5, 10))
    fig.suptitle(f"Potential Layer Changes ({station_name})")
    xlim = [-10, np.max(res)]
    ylim = [max_depth, min_depth]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Resistivity")
    plt.xticks(rotation=45)

    ax.plot(res, depth, c='b', lw=0.5)

    if all(len(p) == 0 for p in peaks):
        pass
    else:
        peak_depths, peak_res, _, _ = zip(*peaks)
    ax.scatter(peak_res, peak_depths, c='b', lw=0.5)
    if troughs is not None:
        if all(len(p) == 0 for p in troughs):
            pass
        else:
            trough_depths, trough_res, _, _ = zip(*troughs)
            ax.scatter(trough_res, trough_depths, c='b', lw=0.5)
    fig.savefig(savepath)
    return fig


def all_plot(station_data, min_depth=0, max_depth=500, savepath='/tmp/all_peaks.png'):
    fig, ax = plt.subplots(figsize=(5, 10))
    fig.suptitle("Potential Layer Changes (All Stations)")

    color_iterator = iter(cm.rainbow(np.linspace(0, 1, len(station_data))))
    highest_res = 0
    for sd in station_data:
        name, depth, res, peaks, troughs = sd
        highest_res = max(np.max(res), highest_res)

        c = next(color_iterator)
        ax.plot(res, depth, label=name, lw=0.5, c=c)
       
        if all(len(p) == 0 for p in peaks):
            pass
        else:
            peak_depths, peak_res, _, _ = zip(*peaks)
            ax.scatter(peak_res, peak_depths, c=c)

        if all(len(t) == 0 for t in troughs):
            pass
        else:
            trough_depths, trough_res, _, _ = zip(*troughs)
            ax.scatter(trough_res, trough_depths, c=c)

    ax.legend()
    ylim = [max_depth, min_depth]
    xlim = [-10, highest_res]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Resistivity")
    plt.xticks(rotation=45)

    fig.savefig(savepath)


def output_csv(peaks, savepath='/tmp/peaks.csv'):
    if all(len(p) == 0 for p in peaks):
        df = pd.DataFrame({'no_peaks_found': [0]})
    else:
        depths, res, prom, width = zip(*peaks)
        df = pd.DataFrame({'depth': depths, 'resistivity': res, 'prominence': prom, 'width': width})
    df.to_csv(savepath)
