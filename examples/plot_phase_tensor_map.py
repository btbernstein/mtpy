# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 07:29:58 2013

@author: Alison Kirkby

plots phase tensor ellipses as a map for a given frequency
"""
import glob
import os
import sys

import mtpy.imaging.plotptmaps as pptmaps


def main(edi_path, freq, save_path=None):
    """Plot Phase Tensor Map
    Args:
        edi_path: path to edi files
        save_path: None  not to save the figure to file; or /tmp/georgina
    Returns:
    """
    # gets edi file names as a list
    elst = glob.glob(os.path.join(edi_path, "*.edi"))

    # parameters describing ellipses, differ for different map scales: deg, m, km
    ellipse_dict = {'size': 0.1, 'colorby': 'phimin', 'range': (0, 90, 1), 'cmap': 'mt_bl2gr2rd'}

    # parameters describing the induction vector arrows
    arrow_dict = {'size': 0.05,
                  'lw': 0.005,
                  'head_width': 0.002,
                  'head_length': 0.002,
                  'threshold': 0.8,
                  'direction': 0}

    # parameters describing the arrow legend (should be self explanatory)
    arrow_legend_dict = {'position': 'upper right',
                         'fontpad': 0.0025,
                         'xborderpad': 0.07,
                         'yborderpad': 0.015}

    m = pptmaps.PlotPhaseTensorMaps(fn_list=elst,
                                    plot_freq=freq,
                                    # arrow_legend_dict=arrow_legend_dict,
                                    ftol=0.2,
                                    # xpad=0.02,  # change according to mapscale
                                    plot_tipper='yri',
                                    arrow_dict=arrow_dict,
                                    ellipse_dict=ellipse_dict,
                                    fig_size=(4, 4),
                                    mapscale='deg',  # deg or m, or km
                                    save_fn=save_path,
                                    fig_dpi=400)

    # m.redraw_plot()

    # if save_path is not None:
    #     plt.savefig(save_path, dpi=300)

    return


###################################################################################################
# How to Run:
# cd /path2/mtpy2
# export PYTHONPATH=/path2/mtpy2
# python examples/plot_phase_tensor_map.py ./examples/data/edi_files/georgina 10 /e/MTPY2_Outputs/
# python examples/plot_phase_tensor_map.py ./examples/data/edi_files 10 /e/MTPY2_Outputs/
# python examples/plot_phase_tensor_map.py tests/data/edifiles/ 10 /e/MTPY2_Outputs/
###################################################################################################
if __name__ == '__main__':

    # the MT edi dir
    edi_path = sys.argv[1]

    # the MT frequency
    # check the freq range in your input edi files: 10 for georgina tests/data/edifiles
    freq=0.0625
    if len(sys.argv)>2:
        freq=float(sys.argv[2])

    if len(sys.argv) > 3:
        save_file = sys.argv[3]
    else:
        save_file = None

    main(edi_path, freq, save_path=save_file)
