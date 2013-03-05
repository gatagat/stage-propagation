#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

import tsh; logger = tsh.create_logger(__name__)
from utils import read_propagatorfile

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot cross-validation results for a given model.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--model', dest='model', required=True, action='store', default=None, help='Propagator file.')
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
    basename = os.path.splitext(os.path.basename(opts.model))[0]
    propagator = read_propagatorfile(opts.model)
    cv_results = propagator['meta']['cv_results']
    data = {}
    for score, params in cv_results:
        key = tuple(params.items())
        if key not in data:
            data[key] = []
        data[key] += [ score ]
    all_param_keys = sorted(data.keys())
    data = tsh.dict_values(data, all_param_keys)
    plt.errorbar(range(len(all_param_keys)), np.mean(data, axis=1), fmt='ro', yerr=np.std(data, axis=1))
    plt.xlabel('Hyper-parameters')
    plt.ylabel('CV score')
    plt.xticks(range(len(all_param_keys)), [dict(p) for p in all_param_keys], rotation='90')
    plt.subplots_adjust(0.10, 0.30, 0.94, 0.92)
    plt.savefig(os.path.join(outdir, basename + '-cv.svg'))
    plt.close()
