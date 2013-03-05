#!/usr/bin/env python

import numpy as np
import os
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, read_weightsfile, read_propagatorfile, write_listfile, clean_args

from propagate_train import method_table

def propagate(method_name, method_args, predictions, weights, output_dir=None):
    args = method_args.copy()
    label_name = args['truth'] + '_labels'
    labels = args[label_name]
    propagated = method_table[method_name]['function'](predictions, weights, labels=labels, output_dir=output_dir, **args)
    args['propagate_method'] = method_name
    return args, propagated


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Propagates labels using predictions and weights.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-m', '--model', dest='model', required=True, action='store', default=None, help='Model file.')
    parser.add_argument('-w', '--weights', dest='weights', required=True, action='store', default=None, help='Weights file.')
    parser.add_argument('-p', '--predictions', dest='predictions', required=True, action='store', default=None, help='Predictions file.')
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
    weights_meta, sample_ids, weights = read_weightsfile(opts.weights)
    assert (predictions['id'] == np.array(sample_ids)).all()
    assert predictions_meta['input_name'] == weights_meta['input_name'], \
            'Expecting same input names (%s x %s)' % (predictions_meta['input_name'], weights_meta['input_name'])
    inputname = predictions_meta['input_name']
    args.update(weights_meta)
    model = read_propagatorfile(opts.model)['propagator']
    method_name = model['method_name']
    del model['method_name']
    args.update(model)
    args, prop = propagate(method_name, args, predictions, weights, output_dir=outdir)
    clean_args(args)
    write_listfile(os.path.join(outdir, inputname + '-propagated.csv'), prop, **args)
