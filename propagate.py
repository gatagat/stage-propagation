#!/usr/bin/env python

import numpy as np
import os
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_argsfile, read_listfile, read_weightsfile, write_listfile, clean_args
from semisupervised import propagate_labels

method_table = {
        'harmonic': { 'function': lambda p, w, **kw: propagate_labels(p, w, method_name='harmonic', **kw) },
        'general': { 'function': lambda p, w, **kw: propagate_labels(p, w, method_name='general', **kw) }
        }

def propagate(method_name, method_args, predictions, weights, output_dir=None):
    args = method_args.copy()
    propagated = method_table[method_name]['function'](predictions, weights, output_dir=output_dir, **args)
    args['propagate_method'] = method_name
    return args, propagated


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Propagates labels using predictions and weights.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Method arguments file.')
    parser.add_argument('-w', '--weights', dest='weights', required=True, action='store', default=None, help='Weights file.')
    parser.add_argument('-p', '--predictions', dest='predictions', required=True, action='store', default=None, help='Predictions file.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    config = tsh.read_config(opts, __file__)
    args = {}
    meta, predictions = read_listfile(opts.predictions)
    args.update(meta)
    meta, sample_ids, weights = read_weightsfile(opts.weights)
    assert (predictions['id'] == np.array(sample_ids)).all()
    args.update(meta)
    if opts.args != None:
        args.update(read_argsfile(opts.args))
    args, prop = propagate(opts.method, args, predictions, weights, output_dir=outdir)
    clean_args(args)
    write_listfile(os.path.join(outdir, 'prop.csv'), prop, **args)
