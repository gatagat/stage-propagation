#!/usr/bin/env python

import heapq
import numpy as np
import tempfile
import os

import tsh.obsolete as tsh; logger = tsh.create_logger(__name__)
from utils import read_weightsfile, read_argsfile, clean_args, write_listfile

def threshold_nearest(w, k=5, symmetric=True, **kwargs):
    keep = np.zeros(w.shape, dtype=bool)
    for j in range(w.shape[0]):
        keep[j, heapq.nsmallest(k, range(w.shape[1]), key=lambda ind: w[j, ind])] = True
    w[np.logical_not(keep + keep.T)] = np.inf
    return w

method_table = {
        'knn': { 'function': threshold_nearest }
        }

def threshold_dissimilarity(method_name, method_args, w):
    args = method_args.copy()
    method_table[method_name]['function'](w, **args)
    return args, w

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Thresholds given dissimilarities.')
    #parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Arguments file.')
    parser.add_argument('-d', '--dissimilarities', dest='dissim', required=True, action='store', default=None, help='Dissimilarities file(s).')
    #parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    #if opts.output == None:
    #    outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    #else:
    #    outdir = opts.output
    #    if not os.path.exists(outdir):
    #        tsh.makedirs(outdir)
    #config = tsh.read_config(opts, __file__)
    meta, ids, w = read_weightsfile(opts.dissim)
    args = meta
    if opts.args != None:
        args.update(read_argsfile(opts.args))
    args, w = threshold_dissimilarity(opts.method, args, w)
    cols = [str(i) for i in ids]
    dissim = np.core.records.fromarrays(
            [ids] + [w[:, i] for i in range(w.shape[1])],
            dtype=zip(['id'] + cols, ['O'] + [np.float64] * len(cols)))
    clean_args(args)
    write_listfile(opts.dissim, dissim, **args)
