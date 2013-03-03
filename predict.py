#!/usr/bin/env python

import numpy as np
import os
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, read_classifierfile, write_listfile, clean_args
from features import compute_features

def predict(model, classes, features, output_dir=None):
    _f = features[[n for n in features.dtype.names if n != 'id']].view(np.float64).reshape(len(features), -1)
    pred = model.predict(_f).astype(int)
    proba = model.predict_proba(_f)
    pred_argmax = np.array(classes)[proba.argmax(axis=1)]
    n_classes = proba.shape[1]
    ret = np.core.records.fromarrays(
            [features['id']] + [pred, pred_argmax] + [proba[:, n].flat for n in range(n_classes)],
            names=['id', 'pred', 'pred_argmax'] + ['prob%d' % int(n) for n in classes])
    return ret

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predicts classes.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-l', '--list', dest='list', required=True, action='store', default=None, help='List file.')
    parser.add_argument('-m', '--model', dest='model', required=True, action='store', default=None, help='Model file.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    config = tsh.read_config(opts, __file__)
    meta, data = read_listfile(opts.list)
    classifier = read_classifierfile(opts.model)
    feature_method = classifier['features']['meta']['feature_method']
    feature_args = meta.copy()
    feature_args.update(classifier['features']['meta'])
    args, features = compute_features(feature_method, feature_args, data, output_dir=outdir)
    assert (data['id'] == features['id']).all()
    clean_args(args)
    write_listfile(os.path.join(outdir, 'feats-test.csv'), features, **args)
    labels_name = classifier['meta']['truth'] + '_labels'
    labels = classifier['meta'][labels_name]
    pred = predict(classifier['classifier'], sorted(labels.keys()), features, output_dir=outdir)
    write_listfile(os.path.join(outdir, 'predictions.csv'), pred, classifier_name=opts.model, truth=classifier['meta']['truth'], labels_name=labels)
