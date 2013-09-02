#!/usr/bin/env python

import heapq
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


def compute_dissimilarity(method_name, method_args, data, n_jobs=None, input_name=None, output_dir=None):
    assert output_dir != None
    assert input_name != None
    args = method_args.copy()
    if 'cache_dir' in args:
        output_dir = args['cache_dir']
    return method_table[method_name]['function'](data, n_jobs=n_jobs, output_dir=output_dir, input_name=input_name, **args)

def threshold_nearest(w, knn=5, symmetric=None, **kwargs):
    keep = np.zeros(w.shape, dtype=bool)
    for j in range(w.shape[0]):
        keep[j, heapq.nsmallest(knn, range(w.shape[1]), key=lambda ind: w[j, ind])] = True
    if symmetric == 'union':
        keep = keep + keep.T
    elif symmetric == 'intersection':
        keep = keep * keep.T
    w[np.logical_not(keep)] = np.inf
    return w

threshold_method_table = {
        'knn': { 'function': threshold_nearest }
        }

def threshold_dissimilarity(method_name, method_args, w):
    args = method_args.copy()
    threshold_method_table[method_name]['function'](w, **args)
    return args, w

def prepare_weights_data(ids, id_dtype, w):
    cols = [str(i) for i in ids]
    dissim = np.core.records.fromarrays(
            [ids] + [w[:, i] for i in range(w.shape[1])],
            dtype=zip(['id'] + cols, [id_dtype] + [np.float64] * len(cols)))
    return dissim


def dissimilarities(methodname, listname, argsname=None, n_jobs=None, outdir=None):
    if outdir == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    else:
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    inputname = os.path.splitext(os.path.basename(listname))[0]
    meta, data = read_listfile(listname)
    args = meta
    if argsname != None:
        args.update(read_argsfile(argsname))
    args, w = compute_dissimilarity(methodname, args, data, n_jobs=n_jobs, output_dir=outdir, input_name=inputname)
    if 'threshold' in args and args['threshold'] != 'False':
        args, w = threshold_dissimilarity(args['threshold'], args, w)
    dissim = prepare_weights_data(data['id'], data.dtype['id'], w)
    clean_args(args)
    write_listfile(os.path.join(outdir, inputname + '-dissim.csv.gz'), dissim, input_name=inputname, **args)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes dissimilarity of the input data.')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Arguments file.')
    parser.add_argument('-l', '--list', dest='list', required=True, action='store', default=None, help='List file.')
    parser.add_argument('-j', '--jobs', dest='jobs', required=False, action='store', default=None, type=int, help='Number of parallel processes.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    dissimilarities(opts.method, opts.list, opts.args, opts.jobs, opts.output)
