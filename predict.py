#!/usr/bin/env python

import numpy as np
import os
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, read_classifierfile, write_featurefile, write_predfile, clean_args
from features import compute_features

def predict(model, features, output_dir=None):
    pred = model.predict(features).astype(int)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)
        n_classes = proba.shape[1]
        ret = np.core.records.fromarrays(
                [pred] + [proba[:, n].flat for n in range(n_classes)],
                names=['pred'] + ['prob%d' % n for n in range(n_classes)])
    else:
        ret = np.core.records.fromarrays([pred], names=['pred'])
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
    feature_meta, features = compute_features(feature_method, feature_args, data, output_dir=outdir)
    clean_args(feature_meta)
    write_featurefile(os.path.join(outdir, 'feats-test.csv'), data['id'], features, **feature_meta)
    pred = predict(classifier['classifier']['model'], features, output_dir=outdir)
    write_predfile(os.path.join(outdir, 'prediction.csv'), data['id'], pred, column_names=pred.dtype.names, classifier_name=opts.model, labels=classifier['classifier']['labels']) 
