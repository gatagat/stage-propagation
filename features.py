#!/usr/bin/env python

import numpy as np
import os
import sys
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_argsfile, read_listfile, write_featurefile
from features_chaincode import get_chaincode_features, prepare_chaincode_features

def get_pregenerated_features(sample, features=None, **kwargs):
    assert features is not None
    return np.array([ sample[f] for f in features ], dtype=np.float64)


def prepare_pregenerated_features(data, features=None, **kwargs):
    return {}


method_table = {
        'pregenerated': { 'function': get_pregenerated_features, 'prepare': prepare_pregenerated_features },
        'chaincode': { 'function': get_chaincode_features, 'prepare': prepare_chaincode_features }
        }


def compute_features(method_name, method_args, data, output_dir=None):
    cache = {}
    args = method_args.copy()
    additional_args = method_table[method_name]['prepare'](data, output_dir=output_dir, **args)
    args.update(additional_args)
    compute_fn = method_table[method_name]['function']
    features = np.zeros((len(data), len(args['feature_names'])), dtype=np.float64)
    for i in range(len(data)):
        features[i, :] = compute_fn(data[i], cache=cache, **args)
    return args, features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes features for all the input data.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Method arguments file.')
    parser.add_argument('-l', '--list', dest='list', required=True, action='store', default=None, help='List file.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args(sys.argv[1:])
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    config = tsh.read_config(opts, __file__)
    meta, data = read_listfile(opts.list)
    args = meta
    if opts.args != None:
        args.update(read_argsfile(opts.args))
    args, features = compute_features(opts.method, args, data, output_dir=outdir)
    args['feature_method'] = opts.method
    for name in args['unserialized']:
        del args[name]
    del args['unserialized']
    write_featurefile(os.path.join(outdir, 'feats.csv'), data['id'], features, **args)
