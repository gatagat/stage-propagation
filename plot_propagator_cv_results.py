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

    param_names = cv_results[0][1].keys()
    assert len(param_names) == 2
    param_values = []
    for name in param_names:
        param_values += [ np.unique([ params[name] for _, params in cv_results ]) ]
    param_index = []
    for values in param_values:
        param_index += [ dict(zip(values, range(len(values)))) ]
    data = np.zeros((len(param_names[0]), len(param_names[1])))
    for score, params in cv_results:
        data[param_index[0][params[param_names[0]]], param_index[1][params[param_names[1]]]] = score
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(data, cmap=plt.cm.jet, origin='lower', interpolation='nearest')
    for y in xrange(len(param_values[0])):
        for x in xrange(len(param_values[1])):
            ax.annotate('%.2f' % data[y][x], xy=(x, y),
                        horizontalalignment='center',
                        verticalalignment='center', size='small')
    plt.ylabel(param_names[0])
    plt.xlabel(param_names[1])
    plt.yticks(range(len(param_values[0])), param_values[0])
    plt.xticks(range(len(param_values[1])), param_values[1])
    plt.ylim(-.5, len(param_values[0])-.5)
    plt.xlim(-.5, len(param_values[1])-.5)
    plt.savefig(os.path.join(outdir, basename + '-cv2.svg'))
    plt.close()
