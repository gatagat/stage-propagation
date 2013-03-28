#!/usr/bin/env python

import glob
import os
import numpy as np
from matplotlib.pylab import csv2rec
import tempfile
import time

import tsh; logger = tsh.create_logger(__name__)
from utils import write_listfile

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Converts CSV file into a listfile.')
    parser.add_argument('-p', '--prefix', dest='image_prefix', required=False, action='store', default=None, help='Path prefix of all image files.')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    parser.add_argument('-t', '--truth', dest='truth', required=False, action='store', default=None, choices=['stage'], help='Truth column.')
    parser.add_argument('--no-timestamp', dest='no_timestamp', required=False, action='store_true', default=False, help='Do not suffix timestamp to the output filename.')
    parser.add_argument('csv_pattern', action='store', help='Wildcard matching input CSV files.')
    opts = parser.parse_args()
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
        logger.info('Output directory %s', outdir)
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)

    converterd = {'dirname': lambda x: x}
    image_prefix = opts.image_prefix if opts.image_prefix != None else ''
    csv_pattern = opts.csv_pattern
    dtype = [('id', object), ('mask', object), ('image', object)]
    if opts.truth == 'stage':
        converterd[opts.truth] = lambda x: x
        truth_map = {
                None: None,
                'unknown': -1,
                'stuff': None,
                'damaged': None,
                'unfertilized': None,
                '': None,
                '1-3': None,
                '4-6': 1,
                '7-8': 2,
                '9-10': 3,
                '11-12': 4,
                '13-14': 5,
                '15-16': 6 }
        truth_labels = dict(zip(truth_map.values(), truth_map.keys()))
        del truth_labels[None]
        labels = truth_labels
        del truth_labels[-1]
        dtype += [(opts.truth, int)]
    rescan_map = { 'rescan': 1, 'rescan2': 2, 'rescan3': 3 }
    dst = np.array([], dtype=dtype)
    t = -1
    dirnames = []
    for csvname in sorted(glob.glob(csv_pattern)):
        label = os.path.splitext(os.path.basename(csvname))[0]
        if label.find('#') != -1:
            continue
        t = max(t, os.path.getmtime(csvname))
        src = csv2rec(csvname, delimiter=',', converterd=converterd)
        for sample in src:
            dirnames += [sample['dirname']]
            sample_dirname = sample['dirname'].split('/')
            sample_date = sample_dirname[-2].split('-')
            if len(sample_date) > 1:
                sample_date = int(sample_date[0]) * 10 + rescan_map[sample_date[1]]
            else:
                sample_date = int(sample_date[0]) * 10
            sample_vt = sample_dirname[-1].split('~')
            if len(sample_vt) > 1:
                sample_suffix = ord(sample_vt[1]) - ord('A') + 1
                sample_vt = int(sample_vt[0])
            else:
                sample_suffix = 0
                sample_vt = int(sample_vt[0])
            sample_id = int(sample['metasys_id']) + 10000 * (
                    sample_suffix + 100 * (sample_vt + 1000000 * sample_date))
            dst_sample = [sample_id,
                'obj%04d-mask.png' % sample['metasys_id'],
                'obj%04d-bri.png' % sample['metasys_id']]
            if opts.truth != None:
                sample_truth = truth_map[sample[opts.truth]]
                if sample_truth not in truth_labels.keys():
                    continue
                dst_sample += [sample_truth]
            dst = np.r_[dst, np.array([tuple(dst_sample)], dtype=dtype)]
    if len(np.unique(dirnames)) > 1:
        for i in range(len(dst)):
            dst[i]['mask'] = os.path.join(dirnames[i], dst[i]['mask'])
            dst[i]['image'] = os.path.join(dirnames[i], dst[i]['image'])
        out_name = ''
    else:
        out_name = dirnames[0].replace('/', '_') + '-'
        image_prefix += dirnames[0]
    dst = np.sort(dst, order='id')
    if not opts.no_timestamp:
        out_name += '%s-' % time.strftime('%y%m%d%H%M%S', time.localtime(t))
    if opts.truth == None:
        out_name += 'data.csv'
    else:
        out_name += 'truth.csv'
    meta = {
        'mask_prefix': image_prefix,
        'image_prefix': image_prefix }
    if opts.truth != None:
        meta['truth'] = opts.truth
        meta[opts.truth + '_labels'] = labels
    write_listfile(os.path.join(outdir, out_name), dst, **meta)
