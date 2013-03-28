#!/usr/bin/env python

import numpy as np
import tempfile
import os

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, read_argsfile, clean_args, write_listfile
from dissimilarities_expression import get_dissimilarities as get_dissimilarities_expression


def get_pregenerated_feature_distances(data, weight_names=None, **kwargs):
    assert weight_names is not None
    kwargs['weight_names'] = weight_names
    return kwargs, tsh.pdist2(data[weight_names], distance=lambda a, b: ((np.array(list(a)) - np.array(list(b))) ** 2).sum())


method_table = {
        'pregenerated': { 'function': get_pregenerated_feature_distances },
        'expression': { 'function': get_dissimilarities_expression }
        }


def compute_dissimilarity(method_name, method_args, data, input_name=None, output_dir=None):
    assert output_dir != None
    assert input_name != None
    args = method_args.copy()
    args, D = method_table[method_name]['function'](data, output_dir=output_dir, input_name=input_name, **args)
    cols = [str(i) for i in data['id']]
    dissim = np.core.records.fromarrays(
            [data['id']] + [D[:, i] for i in range(D.shape[1])],
            dtype=zip(['id'] + cols, [data.dtype['id']] + [np.float64] * len(cols)))
    return args, dissim


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes dissimilarity of the input data.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Arguments file.')
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
    args, dissim = compute_dissimilarity(opts.method, args, data, output_dir=outdir, input_name=inputname)
    clean_args(args)
    write_listfile(os.path.join(outdir, inputname + '-dissim.csv'), dissim, input_name=inputname, **args)
