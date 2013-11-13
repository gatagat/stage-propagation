#!/usr/bin/env python

import os
import tempfile
import shutil

import tsh.obsolete as tsh; logger = tsh.create_logger(__name__)
from utils import read_listfile

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Copies images for all the listfiles.')
    parser.add_argument('-r', '--relative', dest='relative', required=True, action='store', default=None, help='Make paths relative to this.')
    parser.add_argument('lists', nargs='*', action='store', help='List file(s).')
    parser.add_argument('-o', '--output', dest='output', required=False, action='store', default=None, help='Output directory.')
    opts = parser.parse_args()
    if opts.output == None:
        outdir = tempfile.mkdtemp(dir=os.curdir, prefix='out')
        logger.info('Output directory %s', outdir)
    else:
        outdir = opts.output
        if not os.path.exists(outdir):
            tsh.makedirs(outdir)

    for listname in opts.lists:
        meta, data = read_listfile(listname)
        image_prefix = meta['image_prefix']
        mask_prefix = meta['mask_prefix']
        for d in data:
            filename = os.path.join(image_prefix, d['image'])
            destdir = os.path.dirname(os.path.join(outdir, os.path.relpath(filename, opts.relative)))
            tsh.makedirs(destdir)
            shutil.copy(filename, destdir)
            filename = os.path.join(mask_prefix, d['mask'])
            destdir = os.path.dirname(os.path.join(outdir, os.path.relpath(filename, opts.relative)))
            tsh.makedirs(destdir)
            shutil.copy(filename, destdir)
