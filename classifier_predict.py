#!/usr/bin/env python

import numpy as np
import os
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, read_classifierfile, write_listfile, clean_args
from features import compute_features

def predict(model, classes, features, output_dir=None):
    cols = [n for n in features.dtype.names if n != 'id']
    _f = np.zeros((len(features), len(cols)), dtype=np.float64)
    for i in range(len(cols)):
        _f[:, i] = features[cols[i]]
    #_f = features[].view(np.float64).reshape(len(features), -1)
    pred = model.predict(_f).astype(int)
    proba = model.predict_proba(_f)
    pred_argmax = np.array(classes)[proba.argmax(axis=1)]
    n_classes = proba.shape[1]
    ret = np.core.records.fromarrays(
            [features['id']] + [pred, pred_argmax] + \
                    [proba[:, n].flat for n in range(n_classes)],
            names=['id', 'pred', 'pred_argmax'] + \
                    ['prob%d' % int(n) for n in classes])
    return ret


def classifier_predict(listname, modelname, outdir=None, n_jobs=None):
    if outdir == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
    else:
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    inputname = os.path.splitext(os.path.basename(listname))[0]
    if listname.endswith('.gz'):
        inputname = os.path.splitext(inputname)[0]
    meta, data = read_listfile(listname)
    classifier = read_classifierfile(modelname)
    feature_method = classifier['features']['meta']['feature_method']
    feature_args = meta.copy()
    # Training input_name would shadow the current one.
    del classifier['features']['meta']['input_name']
    feature_args.update(classifier['features']['meta'])
    args, features = compute_features(feature_method, feature_args, data,
            input_name=inputname, n_jobs=n_jobs, output_dir=outdir)
    assert (data['id'] == features['id']).all()
    clean_args(args)
    write_listfile(os.path.join(outdir, inputname + '-feats.csv.gz'), features,
            input_name=inputname, **args)
    labels_name = classifier['meta']['truth'] + '_labels'
    labels = classifier['meta'][labels_name]
    pred = predict(classifier['classifier'], sorted(labels.keys()), features,
            output_dir=outdir)
    write_listfile(os.path.join(outdir, inputname + '-predictions.csv.gz'), pred,
            classifier_name=modelname, truth=classifier['meta']['truth'],
            labels_name=labels, input_name=inputname)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predicts classes.')
    parser.add_argument('-l', '--list', dest='list', required=True, action='store', default=None, help='List file.')
    parser.add_argument('-j', '--jobs', dest='jobs', required=False, action='store', default=None, type=int, help='Number of parallel processes.')
    parser.add_argument('-m', '--model', dest='model', required=True, action='store', default=None, help='Model file.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    classifier_predict(opts.list, opts.model, outdir=opts.output, n_jobs=opts.jobs)
