#!/usr/bin/env python

from collections import Counter
import numpy as np
import os
import sklearn
import sklearn.grid_search
import sklearn.svm
import sys
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_truthfile, read_featurefile, write_classifierfile

def train_classifier(args, ids, features, target, truth_meta, output_dir=None):
    labels = None
    truth_name = truth_meta['truth']
    label_name = truth_name + '_labels'
    if label_name in truth_meta:
        labels = truth_meta[label_name]
        logger.info(dict(zip(labels.keys(), [ (np.array(target) == s).sum() for s in labels.keys() ])))
    else:
        labels = dict([ (t, str(t)) for t in np.unique(target) ])

    logger.info(Counter(target))
    if 'balance' in args and args['balance'] != False:
        if args['balance'] == True:
            indices = np.sort(tsh.stratified_indices(target))
        else:
            indices = np.sort(tsh.stratified_indices(target, min_count=args['balance']))
        features = features[indices]
        target = target[indices]
        logger.info(Counter(target))
    else:
        indices = None

    logger.info('SVM grid search...')
    scoring = ('accuracy', sklearn.metrics.zero_one_score)
    class_weight = 'auto'
    logger.info('Coarse grid search')
    coarse_C_range = (10. ** np.arange(-5, 10)).tolist()
    grid = sklearn.grid_search.GridSearchCV(
            #sklearn.svm.LinearSVC(class_weight=class_weight),
            sklearn.svm.SVC(class_weight=class_weight, kernel='linear', probability=False),
            param_grid=dict(C=coarse_C_range),
            score_func=scoring[1],
            cv=sklearn.cross_validation.StratifiedKFold(y=target, k=5))
    grid.fit(features, target)
    cv_score = grid.best_score_
    logger.info('CV-score: %f', cv_score)

    logger.info('Learning coarse SVM...')
    best_estimator = grid.best_estimator_
    best_estimator.probability = True
    best_estimator = best_estimator.fit(features, target)

    logger.info('Evaluating...')
    pred = best_estimator.predict(features)
    train_confusion = sklearn.metrics.confusion_matrix(target, pred, labels=sorted(labels.keys()))
    train_accuracy = tsh.supervised_accuracy(target, pred)
    logger.info('Coarse SVM C-parameter: %f', best_estimator.C)
    logger.info('Coarse confusion (train): %s', train_confusion)
    logger.info('Coarse accuracy (train): %s', train_accuracy)

    return {
            'model': best_estimator,
            'indices': indices,
            'truth': truth_name,
            'labels': labels,
            'scoring': scoring[0],
            'cv_score': cv_score,
            'confusion': train_confusion,
            'accuracy': train_accuracy
           }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes features for all the input data.')
    parser.add_argument('-f', '--feature', dest='feature', required=True, action='store', default=False, help='Feature file.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file.')
    parser.add_argument('-t', '--truth', dest='truth', required=True, action='store', default=None, help='Truth file.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args(sys.argv[1:])
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    config = tsh.read_config(opts, __file__)
    args = { 'balance': 20 }
    truth_meta, truth_ids, target = read_truthfile(opts.truth)
    feature_meta, feature_ids, features = read_featurefile(opts.feature)
    assert (np.array(feature_ids) == np.array(truth_ids)).all()
    classifier = {
            'classifier': train_classifier(args, truth_ids, features, target, truth_meta, output_dir=outdir),
            'truth': { 'ids': truth_ids, 'target': target },
            'features': { 'meta': feature_meta, 'data': features }
            }
    write_classifierfile(os.path.join(outdir, 'classifier.dat'), classifier)
