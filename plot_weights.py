#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import scipy.cluster.hierarchy as hier
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tsh; logger = tsh.create_logger(__name__)
from utils import read_weightsfile, read_truthfile
import class_colors

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot weights.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-w', '--weights', dest='weights', required=True, action='store', default=None, help='Weights file.')
    parser.add_argument('-t', '--truth', dest='truth', required=False, action='store', default=None, help='Truth file.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
        logger.info('Output directory %s', outdir)
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    config = tsh.read_config(opts, __file__)
    meta, ids, weights = read_weightsfile(opts.weights)

    lweights = weights #-np.log(weights)
    #lweights[np.isinf(lweights)] = lweights[~np.isinf(lweights)].min()
    clustering = hier.linkage(lweights, method='average')
    order = hier.leaves_list(clustering)
    ax = plt.gca()
    if opts.truth != None:
        divider = make_axes_locatable(ax)
        ax_target_y = divider.append_axes("right", size=1.0, pad=-0.3, sharey=ax)
    ax.imshow(lweights[np.ix_(order, order)])
    if opts.truth != None:
        meta, ids, target = read_truthfile(opts.truth)
        target -= 1
        target2 = np.zeros((30, len(target)), dtype=target.dtype)
        target2[:] = target[np.ix_(order)]
        target2 = target2.T
        ax_target_y.imshow(target2, cmap='classes')
        ax_target_y.set_axis_off()
    plt.savefig(os.path.join(outdir, os.path.splitext(os.path.basename(opts.weights))[0] + '-sorted.svg'))
    plt.close()

    plt.imshow(weights)
    plt.colorbar()
    plt.savefig(os.path.join(outdir, os.path.splitext(os.path.basename(opts.weights))[0] + '.svg'))
    plt.close()

