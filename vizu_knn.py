#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import numpy as np
import heapq
import os
import tempfile
from jinja2 import Environment, FileSystemLoader

import tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile, read_truthfile, read_weightsfile

def get_samples_data(listname, dissimname, predname, propname, truthname, only_errors):
    meta, data = read_listfile(listname)
    dissim_meta, dissim_ids, dissim = read_weightsfile(dissimname)
    assert (data['id'] == dissim_ids).all()
    if 'truth' in meta:
        truth_name = meta['truth']
        labels = meta[truth_name + '_labels']
    if predname != None:
        pred_meta, pred = read_listfile(predname)
        assert (data['id'] == pred['id']).all()
    if propname != None:
        prop_meta, prop = read_listfile(propname)
        assert (data['id'] == prop['id']).all()
    if truthname != None:
        truth_meta, truth_ids, truth = read_truthfile(truthname)
        truth_name = truth_meta['truth']
        labels = truth_meta[truth_name + '_labels']

    samples = []
    for j in range(len(data)):
        d = data[j]
        sample = {
            'id': d['id'], 
            'image': os.path.join('image', os.path.relpath(
                os.path.join(meta['image_prefix'], d['image']),
                '/home/imp/kazmar/vt_project/Segmentation/Fine/MetaSys/')),
            'mask': os.path.join('image', os.path.relpath(
                os.path.join(meta['mask_prefix'], d['mask']),
                '/home/imp/kazmar/vt_project/Segmentation/Fine/MetaSys/')),
            'expr': os.path.join('expr', 'expr%d.png' % d['id']) }

        neighbor_ids = heapq.nsmallest(5, range(len(data)), key=lambda ind: dissim[j, ind])
        neighbors = []
        for n in neighbor_ids:
            neighbors += [{'id': data['id'][n],
                'dissim': dissim[j, n],
                'expr': os.path.join('expr', 'expr%d.png' % data['id'][n]),
                'prediction': pred[pred['id'] == data['id'][n]][0]['pred'] if predname != None else 'N/A',
                'propagated': prop[prop['id'] == data['id'][n]][0]['pred'] if propname != None else 'N/A',
                'truth': truth[np.array(truth_ids) == data['id'][n]][0] if truthname != None and data['id'][n] in truth_ids else 'N/A'
                }]
        sample['neighbors'] = neighbors
        if predname != None:
            sample['prediction'] = pred[j]['pred']
        else:
            sample['prediction'] = 'N/A'
        if propname != None:
            sample['propagated'] = prop[j]['pred']
        else:
            sample['propagated'] = 'N/A'
        if truthname != None and d['id'] in truth_ids:
            sample['truth'] = truth[np.array(truth_ids) == d['id']][0]
            if only_errors:
                if (predname == None or sample['truth'] == sample['prediction']) and \
                   (propname == None or sample['truth'] == sample['propagated']):
                    continue
        else:
            sample['truth'] = 'N/A'
            if only_errors:
                continue
        samples += [sample]
    return samples

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Creates HTML with k nearest neighbors.')
    parser.add_argument('-c', '--config', dest='config', required=False, action='store', default=None, help='Path to the config file')
    parser.add_argument('-l', '--list', dest='list', nargs='*', required=True, action='store', default=None, help='List file.')
    parser.add_argument('-d', '--dissimilarities', dest='dissim', nargs='*', required=True, action='store', default=None, help='Dissimilarities file.')
    parser.add_argument('-e', '--only-errors', dest='errors', required=False, action='store_true', default=False, help='Keep only samples incorrectly predicted/propagated.')
    parser.add_argument('-p', '--predictions', dest='pred', nargs='*', required=False, action='store', default=None, help='Predictions file.')
    parser.add_argument('-q', '--propagated', dest='prop', nargs='*', required=False, action='store', default=None, help='Predictions file.')
    parser.add_argument('-t', '--truth', dest='truth', nargs='*', required=False, action='store', default=None, help='Truth file(s).')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    config = tsh.read_config(opts, __file__)
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
        logger.info('Output directory %s', outdir)
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)
    
    if len(opts.list) == 0:
        logger.error('Nothing to do')
    else:
        samples = []
        for n in range(len(opts.list)):
            logger.info('Processing %s', opts.list[n])
            if opts.truth != None:
                truthname = opts.truth[n]
            if opts.pred != None:
                predname = opts.pred[n]
            if opts.prop != None:
                propname = opts.prop[n]
            samples += get_samples_data(opts.list[n], opts.dissim[n], predname, propname, truthname, opts.errors)

        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        env = Environment(loader=FileSystemLoader(template_dir))

        if len(opts.list) == 1:
            inputname = os.path.basename(os.path.splitext(opts.list[0])[0])
            suffix = 'errors' if opts.errors else 'neighbors'
            open(os.path.join(outdir, inputname + '-' + suffix + '.html'), 'w').write(env.get_template('neighbors.html').render(
                    title='Nearest neighbors for ' + inputname, k=5, samples=samples))
        else:
            inputname = 'all'
            suffix = 'errors' if opts.errors else 'neighbors'
            open(os.path.join(outdir, inputname + '-' + suffix + '.html'), 'w').write(env.get_template('neighbors.html').render(
                    title='Nearest neighbors for ' + inputname, k=5, samples=samples))

