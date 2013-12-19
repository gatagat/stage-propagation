#!/usr/bin/env python

from collections import Counter
import numpy as np
import os
import sklearn
import sklearn.grid_search
import sklearn.svm
import sklearn.metrics
import sys
import tempfile
import time

import tsh.obsolete as tsh; logger = tsh.create_logger(__name__)
from utils import read_argsfile, read_truthfile, read_featurefile, write_classifierfile


def train_svm(ids, features, target, labels, n_jobs=None, balance=None, coarse_C=None, scoring=None, folds=None, output_dir=None, **kwargs):
    assert coarse_C != None
    assert balance != None
    assert scoring != None
    assert folds != None

    if n_jobs == None:
        n_jobs = 1
    logger.info(Counter(target))
    if balance == False:
        indices = np.arange(len(target))
    elif balance == True:
        indices = np.sort(tsh.stratified_indices(target))
    else:
        indices = np.sort(tsh.stratified_indices(target, min_count=balance))
    features = features[indices]
    target = target[indices]
    logger.info(Counter(target))

    logger.info('SVM grid search...')
    class_weight = 'auto'
    logger.info('Coarse grid search')
    grid = sklearn.grid_search.GridSearchCV(
            #sklearn.svm.LinearSVC(class_weight=class_weight),
            sklearn.svm.SVC(class_weight=class_weight, kernel='linear', probability=False, verbose=False, max_iter=10000000),
            n_jobs=n_jobs,
            param_grid=dict(C=coarse_C),
            scoring=scoring,
            cv=sklearn.cross_validation.StratifiedKFold(y=target, n_folds=folds),
            verbose=True)
            #refit=False)
    grid.fit(features, target)
    cv_score = grid.best_score_
    logger.info('Best C-parameter: %f', grid.best_estimator_.C)
    logger.info('CV-score (%s): %f', scoring, cv_score)

    logger.info('Learning coarse SVM...')
    best_estimator = grid.best_estimator_
    best_estimator.probability = True
    best_estimator = best_estimator.fit(features, target)

    logger.info('Evaluating...')
    pred = best_estimator.predict(features)
    train_confusion = sklearn.metrics.confusion_matrix(target, pred, labels=sorted(labels.keys()))
    train_accuracy = tsh.supervised_accuracy(target, pred)
    logger.info('Confusion (train): %s', train_confusion)
    logger.info('Accuracy (train): %s', train_accuracy)

    args['indices'] = indices
    args['cv_score'] = cv_score
    args['train_confusion'] = train_confusion
    args['train_accuracy'] = train_accuracy
    return args, best_estimator


method_table = {
        'svm': { 'function': train_svm },
        }


def train_classifier(method_name, method_args, ids, features, target, output_dir=None, n_jobs=None):
    args = method_args.copy()
    labels = None
    truth_name = args['truth']
    label_name = truth_name + '_labels'
    labels = args[label_name]
    logger.info(dict(zip(labels.keys(), [ (np.array(target) == s).sum() for s in labels.keys() ])))
    return method_table[method_name]['function'](ids, features, target, labels, output_dir=output_dir, n_jobs=n_jobs, **args)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trains the classifier on all the given data.')
    parser.add_argument('-f', '--feature', dest='feature', required=True, action='store', default=False, help='Feature file.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file.')
    parser.add_argument('-m', '--method', dest='method', required=True, action='store', choices=method_table.keys(), default=None, help='Method name.')
    parser.add_argument('-a', '--args', dest='args', required=False, action='store', default=None, help='Method arguments file.')
    parser.add_argument('-t', '--truth', dest='truth', required=True, action='store', default=None, help='Truth file.')
    parser.add_argument('-j', '--jobs', dest='jobs', required=False, action='store', default=None, type=int, help='Number of parallel processes.')
    parser.add_argument('--random-seed', dest='seed', required=False, action='store', type=int, default=-1, help='Random seed, by default use time.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args(sys.argv[1:])
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    config = tsh.read_config(opts, __file__)
    truth_meta, truth_ids, target = read_truthfile(opts.truth)
    feature_meta, feature_ids, features = read_featurefile(opts.feature)
    args = truth_meta
    if opts.args != None:
        args.update(read_argsfile(opts.args))
    assert (np.array(feature_ids) == np.array(truth_ids)).all()
    if opts.seed == -1:
        seed = int(time.time()*1024*1024)
    else:
        seed = opts.seed
    np.random.seed(seed)
    args, classifier = train_classifier(opts.method, args, truth_ids, features, target, output_dir=outdir, n_jobs=opts.jobs)
    args['random_generator_seed'] = seed
    data = {
            'classifier': classifier,
            'meta': args,
            'truth': { 'meta': truth_meta, 'ids': truth_ids, 'target': target },
            'features': { 'meta': feature_meta, 'ids': feature_ids, 'data': features }
            }
    write_classifierfile(os.path.join(outdir, 'classifier.dat'), data)
