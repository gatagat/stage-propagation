#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile
from class_colors import colors

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot two features from a file.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-x', dest='x_feature', required=True, action='store', default=None, help='X-axis feature.')
    parser.add_argument('-y', dest='y_feature', required=True, action='store', default=None, help='Y-axis feature.')
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
    else:
        meta, data = read_listfile(opts.list)
        pred_meta, pred = read_listfile(opts.pred)
        assert pred_meta['input_name'] == os.path.splitext(os.path.basename(opts.list))[0]
        assert (pred['id'] == data['id']).all()
        target = pred['pred']
        basename = os.path.splitext(os.path.basename(opts.pred))[0]
    try:
        labels = meta[meta['truth'] + '_labels']
    except:
        labels = dict((n, 'Class %d' % n) for n in np.unique(target))
    all_classes = sorted(labels.keys())
    classes = np.unique(target)
    colors = np.array(colors)[np.in1d(all_classes, classes)]
    for c, color in zip(classes, colors):
        mask = target == c
        plt.scatter(data[mask][opts.x_feature], data[mask][opts.y_feature], c=color, edgecolors='none', alpha=0.4)
        plt.scatter(None, None, c=color, edgecolors='none', label=labels[c])
        plt.hold(True)
    plt.xlabel(opts.x_feature)
    plt.ylabel(opts.y_feature)
    plt.legend()
    plt.savefig(os.path.join(outdir, basename + '-%s-vs-%s.svg' % (opts.x_feature, opts.y_feature)))
    plt.close()

    for f in [opts.x_feature, opts.y_feature]:
        plt.hist([data[target == c][f] for c in classes], histtype='barstacked', color=colors, label=[labels[c] for c in classes])
        plt.xlabel('Count')
        plt.ylabel('Classes')
        plt.legend()
        plt.savefig(os.path.join(outdir, basename + '-%s.svg' % f))
        plt.close()
