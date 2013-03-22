#!/usr/bin/env python

import numpy as np
import os
import tempfile
import time

import tsh; logger = tsh.create_logger(__name__)
from utils import read_argsfile, read_listfile, read_truthfile, read_weightsfile, write_propagatorfile, clean_args, select
from semisupervised import propagate_labels

method_table = {
        'harmonic': { 'function': lambda p, w, **kw: propagate_labels(p, w, method_name='harmonic', **kw) },
        'general': { 'function': lambda p, w, **kw: propagate_labels(p, w, method_name='general', **kw) }
        }

import sklearn
import sklearn.cross_validation
import sklearn.grid_search

from joblib import Parallel, delayed

scoring_table = {
        'accuracy': sklearn.metrics.accuracy_score
        }

def train(method_name, method_args, data, output_dir=None):
    args = method_args.copy()
    hyper_params = method_args['hyper_params']
    grid = sklearn.grid_search.ParameterGrid(dict(zip(hyper_params, [ args[p] for p in hyper_params ])))
    verbose = True

    cv_results = Parallel(n_jobs=1, verbose=verbose,
        pre_dispatch='2*n_jobs')(
        delayed(fit_and_score)(
        d, method_name, args, propagate_params, verbose, output_dir)
        for propagate_params in grid for d in data)
        
    n_grid_points = len(list(grid))
    n_fits = len(cv_results)
    n_folds = n_fits // n_grid_points
    scores_mean = np.zeros(n_grid_points)
    scores_std = np.zeros(n_grid_points)
    params = []
    for i in range(n_grid_points):
        grid_start = i * n_folds
        scores = [score for score, _ in cv_results[grid_start:grid_start + n_folds]]
        scores_mean[i] = np.mean(scores)
        scores_std[i] = np.std(scores)
        params += [cv_results[grid_start][1]]
    best_n = np.argmax(scores_mean) 
    # XXX: take model with smallest variance amongst the best

    args['cv_results'] = cv_results
    args['mean_score'] = scores_mean[best_n]
    args['std_score'] = scores_std[best_n]
    model = { 'method_name': method_name }
    logger.info('Best model: %s', str(params[best_n]))
    model.update(params[best_n])
    return args, model

def get_weights(dissim, distance_factor):
    weights = np.exp(-dissim * distance_factor)
    weights[np.diag_indices_from(weights)] = 0.
    return weights

def fit_and_score(data, method_name, method_args, propagate_params, verbose, output_dir):
    args = method_args.copy()
    args.update(data['meta'])
    args.update(propagate_params)
    label_name = args['truth'] + '_labels'
    labels = args[label_name]
    weights = get_weights(data['dissim'], propagate_params['bandwidth'])
    propagate_fn = method_table[method_name]['function']
    propagated = propagate_fn(data['pred'], weights, labels=labels, output_dir=output_dir, **args)
    propagated = select(propagated, 'id', data['truth_ids'])
    logger.info('Using %d propagated samples with ground truth to evaluate', len(propagated))
    score = scoring_table[method_args['scoring']](propagated['pred'], data['target'])
    return score, propagate_params


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train label propagation on all the given data.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Method arguments file.')
    parser.add_argument('-d', '--dissimilarities', dest='dissim', nargs='*', required=True, action='store', default=None, help='Dissimilarities file(s).')
    parser.add_argument('-p', '--predictions', dest='predictions', nargs='*', required=True, action='store', default=None, help='Predictions file(s).')
    parser.add_argument('-t', '--truth', dest='truth', nargs='*', required=True, action='store', default=None, help='Truth file(s).')
    parser.add_argument('--random-seed', dest='seed', required=False, action='store', type=int, default=-1, help='Random seed, by default use time.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    config = tsh.read_config(opts, __file__)
    method_args = {}
    if opts.args != None:
        method_args.update(read_argsfile(opts.args))
    if opts.seed == -1:
        seed = int(time.time()*1024*1024)
    else:
        seed = opts.seed
    np.random.seed(seed)

    data = []
    for prediction_name, dissim_name, truth_name in zip(opts.predictions, opts.dissim, opts.truth):
        meta = {}
        m, predictions = read_listfile(prediction_name)
        meta.update(m)
        truth_meta, truth_ids, target = read_truthfile(truth_name)
        assert np.in1d(predictions['id'], np.array(truth_ids)).sum() > 0
        m, dissim_ids, dissim = read_weightsfile(dissim_name)
        assert (predictions['id'] == np.array(dissim_ids)).all()
        meta.update(m)
        data += [{
            'meta': meta,
            'pred': predictions,
            'dissim': dissim,
            'truth_ids': truth_ids,
            'target': target
        }]
    args, model = train(opts.method, method_args, data, output_dir=outdir)
    args['random_generator_seed'] = seed
    clean_args(args)
    write_propagatorfile(os.path.join(outdir, 'propagator.dat'), {
        'propagator': model,
        'meta': args,
        'data': data
    })
