#!/usr/bin/env python

from joblib import Parallel, delayed
import numpy as np
import os
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_argsfile, read_listfile, write_listfile, clean_args

method_table = {}

def get_pregenerated_features(sample, feature_names=None, **kwargs):
    assert feature_names is not None
    return np.array([ sample[f] for f in feature_names ], dtype=np.float64)


def prepare_pregenerated_features(data, features=None, **kwargs):
    return {}


method_table['pregenerated'] = {
    'function': get_pregenerated_features,
    'prepare': prepare_pregenerated_features
}

try:
    from features_chaincode import get_chaincode_features,\
        prepare_chaincode_features
    method_table['chaincode'] = {
        'function': get_chaincode_features,
        'prepare': prepare_chaincode_features
    }
except ImportError:
    logger.exception('Missing packages - chaincode features will not be available')
    pass


def compute_features(method_name, method_args, data, n_jobs=None, input_name=None, output_dir=None):
    cache = {}
    args = method_args.copy()
    if method_name not in method_table:
        raise NotImplementedError
    additional_args = method_table[method_name]['prepare'](data, input_name=input_name, output_dir=output_dir, **args)
    args.update(additional_args)
    feature_names = args['feature_names']
    #del args['feature_names']
    compute_fn = method_table[method_name]['function']
    N = len(data)
    d = len(feature_names)
    if n_jobs == None:
        _f = np.zeros((N, d), dtype=np.float64)
        for i in range(N):
            _f[i, :] = compute_fn(data[i], cache=cache, input_name=input_name, output_dir=output_dir, **args)
    else:
        results = Parallel(n_jobs=n_jobs, verbose=True, pre_dispatch='2*n_jobs')(
                delayed(compute_fn)(di, cache=cache, input_name=input_name, output_dir=output_dir, **args)
                for di in [dict(zip(data.dtype.names, i)) for i in data])
        _f = np.r_[results]

    features = np.core.records.fromarrays(
            [data['id']] + [_f[:, i] for i in range(_f.shape[1])],
            dtype=zip(['id'] + feature_names, [data.dtype['id']] + [np.float64] * d))
    args['feature_method'] = method_name
    return args, features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes features for all the input data.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Method arguments file.')
    parser.add_argument('-l', '--list', dest='list', required=True, action='store', default=None, help='List file.')
    parser.add_argument('-j', '--jobs', dest='jobs', required=False, action='store', default=None, type=int, help='Number of parallel processes.')
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
    meta, data = read_listfile(opts.list)
    args = meta
    if opts.args != None:
        args.update(read_argsfile(opts.args))
    args, features = compute_features(opts.method, args, data, input_name=inputname, n_jobs=opts.jobs, output_dir=outdir)
    clean_args(args)
    write_listfile(os.path.join(outdir, inputname + '-feats.csv.gz'), features, input_name=inputname, **args)
