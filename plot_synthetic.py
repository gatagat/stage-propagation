#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

import tsh.obsolete as tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile
from colormaps import get_synthetic_colors

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.01
matplotlib.rcParams['figure.figsize'] = (2, 2)
matplotlib.rcParams['figure.subplot.left'] = 0.125
matplotlib.rcParams['figure.subplot.right'] = 0.9
matplotlib.rcParams['figure.subplot.bottom'] = 0.1
matplotlib.rcParams['figure.subplot.top'] = 0.9

extension = '.svg'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot two features from a file.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-x', dest='x_feature', required=True, action='store', default=None, help='X-axis feature.')
    parser.add_argument('-y', dest='y_feature', required=True, action='store', default=None, help='Y-axis feature.')
    parser.add_argument('--x-label', dest='x_label', required=False, action='store', default=None, help='X-axis label.')
    parser.add_argument('--y-label', dest='y_label', required=False, action='store', default=None, help='Y-axis label.')
    parser.add_argument('--no-y-ticks', dest='no_y_ticks', required=False, action='store_true', default=False, help='Do not put ticks on Y-axis.')
    parser.add_argument('--legend', dest='legend', required=False, action='store_true', default=False, help='Plot legend.')
    parser.add_argument('-t', '--truth', dest='truth', required=False, action='store', default=None, help='Truth file.')
    parser.add_argument('-p', '--pred', dest='pred', required=False, action='store', default=None, help='Prediction file.')
    parser.add_argument('-l', '--list', dest='list', required=False, action='store', default=None, help='List file.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    if opts.truth == None and (opts.pred == None or opts.list == None):
        raise ValueError('Either truth file (containing both features and labels) or predictions (containing labels) and list files (containing features) have to be specified.')
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
        logger.info('Output directory %s', outdir)
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    config = tsh.read_config(opts, __file__)
    if opts.truth != None:
        meta, data = read_listfile(opts.truth)
        target = data['class']
        basename = os.path.splitext(os.path.basename(opts.truth))[0]
        if opts.truth.endswith('.gz'):
            basename = os.path.splitext(basename)[0]
    else:
        meta, data = read_listfile(opts.list)
        pred_meta, pred = read_listfile(opts.pred)
        #inputname = os.path.splitext(os.path.basename(opts.list))[0]
        #if opts.list.endswith('.gz'):
        #    inputname = os.path.splitext(inputname)[0]
        #assert pred_meta['input_name'] == inputname
        assert (pred['id'] == data['id']).all()
        target = pred['pred']
        basename = os.path.splitext(os.path.basename(opts.pred))[0]
        if opts.pred.endswith('.gz'):
            basename = os.path.splitext(basename)[0]
    try:
        labels = meta[meta['truth'] + '_labels']
    except:
        labels = dict((n, 'Class %d' % n) for n in np.unique(target))
    all_classes = sorted(labels.keys())
    classes = np.unique(target)
    colors = get_synthetic_colors()
    shapes = np.array(['o', 's', '>'])[np.in1d(all_classes, classes)].tolist()
    colors = np.array(colors)[np.in1d(all_classes, classes)].tolist()
    for c, shape, color in zip(classes, shapes, colors):
        mask = target == c
        if mask.any():
            plt.scatter(data[mask][opts.x_feature], data[mask][opts.y_feature], c=color, marker=shape, edgecolors='none', alpha=0.4)
            plt.scatter(None, None, c=color, edgecolors='none', label=labels[c])
            plt.hold(True)
    if opts.x_label != None:
        plt.xlabel(opts.x_label)
    if opts.y_label != None:
        plt.ylabel(opts.y_label)
    plt.xticks([1,2,3], [1,2,3])
    if opts.no_y_ticks:
        plt.yticks([1,2,3], ['','',''])
    else:
        plt.yticks([1,2,3], [1,2,3])
    if opts.legend:
        plt.legend()
    plt.savefig(os.path.join(outdir, basename + '-%s-vs-%s' % (opts.x_feature, opts.y_feature) + extension))
    plt.close()

    for f in [opts.x_feature, opts.y_feature]:
        plt.hist([data[target == c][f] for c in classes], histtype='barstacked', color=colors, label=[labels[c] for c in classes])
        plt.xlabel('Count')
        plt.ylabel('Classes')
        if opts.legend:
            plt.legend()
        plt.savefig(os.path.join(outdir, basename + '-%s' % f + extension))
        plt.close()
