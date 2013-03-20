#!/usr/bin/env python

import numpy as np

import tsh; logger = tsh.create_logger(__name__)
from utils import read_argsfile, write_listfile, clean_args

def generate(N=None, priors=None, means=None, sigmas=None, feature_dims=None, weight_dims=None, noise_dims=None, **kwargs):
    args = kwargs.copy()
    n_feats = len(means)
    assert n_feats == len(sigmas)
    if noise_dims == None:
        noise_dims = 0
    assert n_feats == feature_dims + weight_dims + noise_dims
    feature_names = ['f%02d' % d for d in range(n_feats)]
    data = np.zeros(N, dtype=[('id', int), ('class', int)] + zip(feature_names, [np.float64]*n_feats))
    data['id'] = range(N)

    c = np.cumsum(priors).astype(float)
    c /= c[-1]
    targets = np.random.rand(N)
    n_classes = len(priors)
    data['class'] = 1
    for i in range(n_classes):
        data['class'][targets > c[i]] = i+2

    k = data['class']-1
    for d in range(n_feats):
        mu = np.array(means)[d, k]
        si = np.array(sigmas)[d, k]
        data[feature_names[d]] = np.random.normal(mu, si)

    args['priors'] = priors
    args['means'] = means
    args['sigmas'] = sigmas
    args['class_labels'] = dict(zip(range(1, n_classes+1), [ 'C%02d' % i for i in range(1, n_classes+1) ]))
    args['feature_names'] = feature_names[:feature_dims] + feature_names[feature_dims+weight_dims:]
    args['weight_names'] = feature_names[feature_dims:feature_dims+weight_dims] + feature_names[feature_dims+weight_dims:]
    args['truth'] = 'class'
    return args, data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic data.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-a', '--args', dest='args', required=True, action='store', default=None, help='Method arguments file.')
    parser.add_argument('output', action='store', default=None, help='Output file.')
    opts = parser.parse_args()
    config = tsh.read_config(opts, __file__)
    args = read_argsfile(opts.args)
    args, data = generate(**args)
    clean_args(args)
    write_listfile(opts.output, data, **args)
