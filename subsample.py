#!/usr/bin/env python

import numpy as np
import os
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, write_listfile

def subsample(filename, sub):
    """
    sub is either a number of samples to select, or a fraction of samples to select
    """
    m, d = read_listfile(filename)
    if type(sub) == float:
        sub = int(np.round(len(d) * sub))
    return m, d[np.random.permutation(len(d))[:sub]]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes features for all the input data.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-n', dest='sub', required=True, action='store', default=None, help='Number (or fraction) of samples to select.')
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
    inputname = os.path.splitext(os.path.basename(opts.list))[0]
    if opts.list.endswith('.gz'):
        inputname = os.path.splitext(inputname)[0]
    config = tsh.read_config(opts, __file__)
    if opts.sub == None:
        sub = .1
    else:
        sub = float(opts.sub) if opts.sub.find('.') != -1 else int(opts.sub)
    m, d = subsample(opts.list, sub)
    write_listfile(os.path.join(outdir, inputname + '-part.csv'), d, sub=sub, **m)
