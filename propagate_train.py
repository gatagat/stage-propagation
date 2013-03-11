#!/usr/bin/env python

import numpy as np
import os
import tempfile
import time

import tsh; logger = tsh.create_logger(__name__)
from utils import read_argsfile, read_listfile, read_truthfile, read_weightsfile, write_propagatorfile, clean_args
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

def train(method_name, method_args, ids, predictions, weights, target, output_dir=None):
    args = method_args.copy()
    label_name = args['truth'] + '_labels'
    labels = args[label_name]
    propagate_fn = method_table[method_name]['function']
    cv = sklearn.cross_validation.StratifiedKFold(y=target, n_folds=args['folds'])
    hyper_params = args['hyper_params']
    grid = sklearn.grid_search.ParameterGrid(dict(zip(hyper_params, [ args[p] for p in hyper_params ])))
    verbose = True
    scoring = args['scoring']

    cv_results = Parallel(n_jobs=1, verbose=verbose,
        pre_dispatch='2*n_jobs')(
        delayed(fit_and_score)(
        predictions, weights, target, labels, propagate_fn, propagate_params, train, scoring_table[scoring], verbose, output_dir, **args)
        for propagate_params in grid for train, _ in cv)

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

def fit_and_score(predictions, weights, target, labels, propagate_fn, propagate_params, indices, scoring_fn, verbose, output_dir, **kwargs):
    kwargs.update(propagate_params)
    propagated = propagate_fn(predictions[indices], weights[np.ix_(indices, indices)], labels=labels, output_dir=output_dir, **kwargs)
    score = scoring_fn(propagated['pred'], target[indices])
    return score, propagate_params


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train label propagation on all the given data.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Method arguments file.')
    parser.add_argument('-w', '--weights', dest='weights', required=True, action='store', default=None, help='Weights file.')
    parser.add_argument('-p', '--predictions', dest='predictions', required=True, action='store', default=None, help='Predictions file.')
    parser.add_argument('-t', '--truth', dest='truth', required=True, action='store', default=None, help='Truth file.')
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
    args = {}
    predictions_meta, predictions = read_listfile(opts.predictions)
    args.update(predictions_meta)
    truth_meta, truth_ids, target = read_truthfile(opts.truth)
    assert (predictions['id'] == np.array(truth_ids)).all()
    weights_meta, weights_ids, weights = read_weightsfile(opts.weights)
    assert (predictions['id'] == np.array(weights_ids)).all()
    args.update(weights_meta)
    if opts.args != None:
        args.update(read_argsfile(opts.args))
    if opts.seed == -1:
        seed = int(time.time()*1024*1024)
    else:
        seed = opts.seed
    np.random.seed(seed)
    args, model = train(opts.method, args, truth_ids, predictions, weights, target, output_dir=outdir)
    args['random_generator_seed'] = seed
    clean_args(args)
    data = {
            'propagator': model,
            'meta': args,
            'truth': { 'meta': truth_meta, 'ids': truth_ids, 'target': target },
            'pred': { 'meta': predictions_meta, 'pred': predictions },
            'weights': { 'meta': weights_meta, 'ids': weights_ids, 'weights': weights }
            }
    write_propagatorfile(os.path.join(outdir, 'propagator.dat'), data)
