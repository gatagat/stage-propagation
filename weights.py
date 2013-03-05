#!/usr/bin/env python

import numpy as np
import tempfile
import os

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, read_argsfile, clean_args, write_listfile
from weights_expression import get_distances as get_distances_expression


def get_pregenerated_feature_distances(data, weight_names=None, **kwargs):
    assert weight_names is not None
    kwargs['weight_names'] = weight_names
    return kwargs, tsh.pdist2(data[weight_names], distance=lambda a, b: (a.view(float) - b.view(float)) ** 2)


method_table = {
        'pregenerated': { 'function': get_pregenerated_feature_distances },
        'expression': { 'function': get_distances_expression }
        }


def compute_weight_matrix(method_name, method_args, data, output_dir=None):
    assert output_dir != None
    args = method_args.copy()
    distance_factor = args['distance_factor']
    args, D = method_table[method_name]['function'](data, output_dir=output_dir, **args)
    D = np.exp(-D * distance_factor)
    D[np.diag_indices_from(D)] = 0.
    cols = [str(i) for i in data['id']]
    W = np.core.records.fromarrays(
            [data['id']] + [D[:, i] for i in range(D.shape[1])],
            dtype=zip(['id'] + cols, [data.dtype['id']] + [np.float64] * len(cols)))
    return args, W


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes weights measuring similarity of the input data.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Arguments file.')
    parser.add_argument('-f', '--factor', dest='factor', required=False, action='store', type=float, default=-1, help='Distance factor.')
    parser.add_argument('-l', '--list', dest='list', required=True, action='store', default=None, help='List file.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    inputname = os.path.splitext(os.path.basename(opts.list))[0]
    config = tsh.read_config(opts, __file__)
    meta, data = read_listfile(opts.list)
    args = meta
    if opts.args != None:
        args.update(read_argsfile(opts.args))
    if opts.factor > 0.:
        args.update({'distance_factor': opts.factor})
    args, weights = compute_weight_matrix(opts.method, args, data, output_dir=outdir)
    clean_args(args)
    write_listfile(os.path.join(outdir, inputname + '-weights.csv'), weights, input_name=inputname, **args)
