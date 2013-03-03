#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot two features from a file.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-x', dest='x_feature', required=True, action='store', default=None, help='X-axis feature.')
    parser.add_argument('-y', dest='y_feature', required=True, action='store', default=None, help='Y-axis feature.')
    parser.add_argument('-l', '--list', dest='list', required=True, action='store', default=None, help='List file.')
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
    meta, data = read_listfile(opts.list)

    colors = [ '#bbb12d', '#1480fa', '#bd2309', '#faf214', '#2edfea', '#ea2ec4',
               '#14fa2f', '#ea2e40', '#cdcdcd', '#577a4d', '#2e46c0', '#f59422',
               '#219774', '#8086d9', '#000000' ]
    classes = np.unique(data['class'])
    for c, color in zip(classes, colors):
        mask = data['class'] == c
        plt.scatter(data[mask][opts.x_feature], data[mask][opts.y_feature], c=color, edgecolors='none', alpha=0.4)
        plt.scatter(None, None, c=color, edgecolors='none', label='Class %d' % c)
        plt.hold(True)
    plt.xlabel(opts.x_feature)
    plt.ylabel(opts.y_feature)
    plt.legend()
    plt.savefig(os.path.join(outdir, os.path.splitext(os.path.basename(opts.list))[0] + '-%s-vs-%s.svg' % (opts.x_feature, opts.y_feature)))
    plt.close()

    for f in [opts.x_feature, opts.y_feature]:
        plt.hist([data[data['class'] == c][f] for c in classes], histtype='barstacked', color=colors[:len(classes)], label=['Class %d' % c for c in classes])
        plt.xlabel('Count')
        plt.ylabel('Classes')
        plt.legend()
        plt.savefig(os.path.join(outdir, os.path.splitext(os.path.basename(opts.list))[0] + '-%s.svg' % f))
        plt.close()
