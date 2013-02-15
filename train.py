#!/usr/bin/env python

import numpy as np
import os
import sys
import tempfile

import tsh; logger = tsh.create_logger(__name__)
from utils import read_truthfile, write_classifierfile

def train_classifier(features, truth):
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes features for all the input data.')
    parser.add_argument('-f', '--feature', dest='feature', required=False, action='store', default=False, help='Feature file.')
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
    truth_meta, truth = read_truthfile(opts.truth)
    feature_meta, feature = read_featurefile(opts.feature)
    args, features = compute_features(opts.method, args, data, output_dir=outdir)
    args['feature_method'] = opts.method
    write_classifierfile(os.path.join(outdir, 'classifier.dat'), feature_mweta, **args)
